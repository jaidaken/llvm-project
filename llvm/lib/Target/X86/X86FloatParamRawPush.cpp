//===--- X86FloatParamRawPush.cpp - Float param as raw integer move --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: When forwarding a float parameter to another function, MSVC 6.0
// treats the float as a raw 4-byte integer:
//   mov edx, [esp+0x08]   ; load float param as raw 32-bit value
//   push edx               ; push as integer
//
// Clang knows the type is float and routes it through the FPU:
//   flds [esp+0x08]        ; load float into FPU
//   fstps [esp]            ; store back from FPU to stack
//
// The FPU round-trip produces 6+ bytes vs 5 bytes for the raw push, and uses
// different instructions entirely. For simple forwarding of single-precision
// floats, the FPU instructions are semantically identical to integer mov
// (no rounding occurs).
//
// This pass, gated on the "float_param_raw_push" string attribute, finds
// LD_F32m + ST_FP32m (fld+fstp) pairs and replaces them with MOV32rm + MOV32mr
// using a free GPR (preferring EDX, then EAX, then ECX).
//
// The fld and fstp need not be adjacent - the pass scans forward from each fld
// to find the matching fstp. It verifies no other FPU instruction appears
// between them (they must be the only FPU ops in the range).
//
// Additionally, when the fld loads from a constant pool entry (a float literal)
// and the fstp stores to an ESP-relative location (outgoing call argument),
// the pass replaces both with a single PUSH32i containing the float's raw
// 32-bit integer representation. This matches MSVC 6.0's pattern of
// "push 0x42480000" for float constants like 50.0f.
//
// Must run AFTER the FP stackifier (which converts FP pseudos to real x87
// instructions) and among the other bw1-decomp passes.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/IR/Constants.h"

using namespace llvm;

#define DEBUG_TYPE "x86-float-param-raw-push"
#define X86_FLOAT_PARAM_RAW_PUSH_NAME                                          \
  "X86 float param raw push (fld+fstp -> mov+mov)"

namespace {
class X86FloatParamRawPushPass : public MachineFunctionPass {
public:
  static char ID;
  X86FloatParamRawPushPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return X86_FLOAT_PARAM_RAW_PUSH_NAME;
  }
};
} // end anonymous namespace

char X86FloatParamRawPushPass::ID = 0;

/// Check if an instruction is a post-stackifier x87 FPU instruction.
/// After the FP stackifier, all real x87 instructions (FPI class) implicitly
/// define FPSW. We also exclude calls and inline asm which may touch x87 regs.
static bool isPostStackifierFPU(const MachineInstr &MI) {
  if (MI.isCall() || MI.isInlineAsm() || MI.isPseudo())
    return false;
  for (const MachineOperand &MO : MI.implicit_operands()) {
    if (!MO.isReg())
      continue;
    Register Reg = MO.getReg();
    if (Reg == X86::FPSW || Reg == X86::FPCW ||
        (Reg >= X86::ST0 && Reg <= X86::ST7))
      return true;
  }
  return false;
}

/// Try to extract a float constant from a constant pool reference in a
/// LD_F32m instruction. Returns true if the displacement operand references
/// a constant pool entry containing a 32-bit float, and fills in RawBits.
static bool getFloatConstantBits(const MachineInstr &Fld,
                                 const MachineFunction &MF,
                                 uint32_t &RawBits) {
  // LD_F32m operands: base(0), scale(1), index(2), disp(3), seg(4)
  // For a constant pool load: base=0, scale=1, index=0, disp=CPI, seg=0
  const MachineOperand &Disp = Fld.getOperand(3);
  if (!Disp.isCPI() || Disp.getOffset() != 0)
    return false;

  // Base and index should be NoRegister for a pure constant pool reference.
  if (Fld.getOperand(0).getReg() != X86::NoRegister)
    return false;
  if (Fld.getOperand(2).getReg() != X86::NoRegister)
    return false;

  ArrayRef<MachineConstantPoolEntry> Constants =
      MF.getConstantPool()->getConstants();
  const MachineConstantPoolEntry &Entry = Constants[Disp.getIndex()];
  if (Entry.isMachineConstantPoolEntry())
    return false;

  const Constant *C = Entry.Val.ConstVal;
  const auto *CFP = dyn_cast<ConstantFP>(C);
  if (!CFP)
    return false;

  const APFloat &FPVal = CFP->getValueAPF();
  // Must be a 32-bit float (IEEE single precision).
  if (&FPVal.getSemantics() != &APFloat::IEEEsingle())
    return false;

  RawBits = FPVal.bitcastToAPInt().getZExtValue();
  return true;
}

/// Check if an ST_FP32m stores to an ESP-relative address (outgoing call arg).
static bool isESPRelativeStore(const MachineInstr &Fstp) {
  // ST_FP32m operands: base(0), scale(1), index(2), disp(3), seg(4)
  return Fstp.getOperand(0).isReg() &&
         Fstp.getOperand(0).getReg() == X86::ESP;
}

bool X86FloatParamRawPushPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("float_param_raw_push"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Collect fld+fstp pairs by scanning forward from each LD_F32m.
    // We store: {Fld, Fstp} pairs for replacement.
    SmallVector<std::pair<MachineInstr *, MachineInstr *>, 4> ToReplace;

    for (auto I = MBB.begin(), E = MBB.end(); I != E; ++I) {
      if (I->getOpcode() != X86::LD_F32m)
        continue;

      MachineInstr *Fld = &*I;

      // Scan forward to find the matching ST_FP32m.
      // Bail if we hit another FPU instruction (other than the fstp itself).
      bool FoundFPUConflict = false;
      MachineInstr *Fstp = nullptr;

      for (auto J = std::next(I); J != E; ++J) {
        if (J->isPseudo())
          continue;

        if (J->getOpcode() == X86::ST_FP32m) {
          Fstp = &*J;
          break;
        }

        // Any other FPU instruction between fld and fstp means the FPU stack
        // is being used for something else - we cannot safely remove the fld.
        if (isPostStackifierFPU(*J)) {
          FoundFPUConflict = true;
          break;
        }
      }

      if (FoundFPUConflict || !Fstp)
        continue;

      ToReplace.push_back({Fld, Fstp});
    }

    // Apply replacements.
    for (auto &[Fld, Fstp] : ToReplace) {
      // Check if this is a constant pool load that can become PUSH32i.
      uint32_t RawBits = 0;
      bool IsConstPoolPush =
          getFloatConstantBits(*Fld, MF, RawBits) && isESPRelativeStore(*Fstp);

      if (IsConstPoolPush) {
        // Replace fld [constpool] + fstp [esp+N] with push imm32.
        // The fstp writes to [ESP + disp]. Since we are replacing the
        // sub esp,4 + fstp sequence with a push, we need to verify the
        // displacement is 0 (the top of the outgoing args area).
        int64_t FstpDisp = Fstp->getOperand(3).getImm();
        if (FstpDisp != 0) {
          // Non-zero ESP displacement - fall through to MOV32rm+MOV32mr path.
          goto mov_path;
        }

        DebugLoc DL = Fld->getDebugLoc();

        // Build PUSH32i with the raw float bits as immediate.
        // Use PUSH32i for values that don't fit in 8 bits, PUSH32i8 otherwise.
        // Float bit patterns rarely fit in sign-extended 8 bits, but check.
        int32_t SignedBits = static_cast<int32_t>(RawBits);
        unsigned PushOpc =
            (SignedBits >= -128 && SignedBits <= 127) ? X86::PUSH32i8
                                                     : X86::PUSH32i;
        BuildMI(MBB, *Fstp, DL, TII->get(PushOpc))
            .addImm(static_cast<int64_t>(SignedBits));

        // Remove the fld and fstp.
        Fld->eraseFromParent();
        Fstp->eraseFromParent();
        Changed = true;
        continue;
      }

    mov_path:
      // Standard path: replace fld+fstp with MOV32rm+MOV32mr using a free GPR.
      // Compute liveness at the fld point to pick a free GPR.
      // Prefer EDX (MSVC 6.0 convention for float forwarding), then EAX, ECX.
      LivePhysRegs LocalLive(*TRI);
      LocalLive.addLiveOuts(MBB);
      for (auto RI = MBB.rbegin(); &*RI != Fld; ++RI)
        LocalLive.stepBackward(*RI);

      Register IntermediateReg = 0;
      for (Register Candidate : {X86::EDX, X86::EAX, X86::ECX}) {
        if (!LocalLive.contains(Candidate)) {
          IntermediateReg = Candidate;
          break;
        }
      }
      if (!IntermediateReg)
        continue; // No free GPR, skip this pair

      DebugLoc DL = Fld->getDebugLoc();

      // Build MOV32rm: load float as integer from fld's source address.
      // LD_F32m operands: base(0), scale(1), index(2), disp(3), seg(4)
      auto MovLoad =
          BuildMI(MBB, *Fld, DL, TII->get(X86::MOV32rm), IntermediateReg);
      for (unsigned i = 0; i < Fld->getNumOperands(); ++i)
        MovLoad.add(Fld->getOperand(i));
      MovLoad.cloneMemRefs(*Fld);

      // Build MOV32mr: store integer to fstp's destination address.
      // ST_FP32m operands: base(0), scale(1), index(2), disp(3), seg(4)
      auto MovStore = BuildMI(MBB, *Fstp, DL, TII->get(X86::MOV32mr));
      for (unsigned i = 0; i < Fstp->getNumOperands(); ++i)
        MovStore.add(Fstp->getOperand(i));
      MovStore.addReg(IntermediateReg);
      MovStore.cloneMemRefs(*Fstp);

      // Remove the fld and fstp.
      Fld->eraseFromParent();
      Fstp->eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86FloatParamRawPushPass() {
  return new X86FloatParamRawPushPass();
}

// bw1-decomp: Fix CMP operand materialization order for MSVC 6.0.
//
// When comparing two memory values, the register allocator picks which
// operand gets loaded into a register. MSVC 6.0 loads the LHS (first
// memory operand of CMP), while Clang loads the RHS.
//
// Clang generates:
//   MOV reg, [mem_B]     ; load RHS into register
//   CMP [mem_A], reg     ; CMP mr (opcode 0x39): computes [mem_A] - reg
//
// MSVC 6.0 generates:
//   MOV reg, [mem_A]     ; load LHS into register
//   CMP reg, [mem_B]     ; CMP rm (opcode 0x3B): computes reg - [mem_B]
//
// Both compute A - B, so flags are identical and no Jcc condition swap
// is needed. Only the encoding and register contents change.
//
// This pass finds MOV+CMP pairs where the MOV loads the CMP's register
// operand from memory, and swaps the MOV's source to be the CMP's memory
// operand instead (the LHS), converting the CMP from mr to rm form.
//
// Gated on the "cmp_lhs_reg" function attribute.

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define DEBUG_TYPE "x86-cmp-lhs-reg"

namespace {

/// Map CMP mr opcodes to their rm equivalents.
static unsigned getCmpRmForCmpMr(unsigned Opc) {
  switch (Opc) {
  case X86::CMP8mr:  return X86::CMP8rm;
  case X86::CMP16mr: return X86::CMP16rm;
  case X86::CMP32mr: return X86::CMP32rm;
  case X86::CMP64mr: return X86::CMP64rm;
  default: return 0;
  }
}

/// Map CMP mr opcodes to the corresponding MOV rm opcodes.
static unsigned getMovRmForCmpMr(unsigned Opc) {
  switch (Opc) {
  case X86::CMP8mr:  return X86::MOV8rm;
  case X86::CMP16mr: return X86::MOV16rm;
  case X86::CMP32mr: return X86::MOV32rm;
  case X86::CMP64mr: return X86::MOV64rm;
  default: return 0;
  }
}

/// Check if two memory operand sequences refer to the same address.
/// CMP mr memory operands start at index 0: [base, scale, index, disp, seg].
/// MOV rm memory operands start at index 1: [base, scale, index, disp, seg].
static bool sameMemOperands(const MachineInstr &MI_A, unsigned StartA,
                            const MachineInstr &MI_B, unsigned StartB) {
  for (unsigned i = 0; i < 5; ++i) {
    const MachineOperand &A = MI_A.getOperand(StartA + i);
    const MachineOperand &B = MI_B.getOperand(StartB + i);
    if (A.getType() != B.getType())
      return false;
    if (A.isReg()) {
      if (A.getReg() != B.getReg())
        return false;
    } else if (A.isImm()) {
      if (A.getImm() != B.getImm())
        return false;
    } else if (A.isGlobal()) {
      if (A.getGlobal() != B.getGlobal() || A.getOffset() != B.getOffset())
        return false;
    } else if (A.isCPI()) {
      if (A.getIndex() != B.getIndex() || A.getOffset() != B.getOffset())
        return false;
    } else {
      return false;
    }
  }
  return true;
}

class X86CmpLhsRegPass : public MachineFunctionPass {
public:
  static char ID;
  X86CmpLhsRegPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 CMP LHS register materialization for MSVC 6.0";
  }
};
} // end anonymous namespace

char X86CmpLhsRegPass::ID = 0;

bool X86CmpLhsRegPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("cmp_lhs_reg"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E; ) {
      MachineInstr &CmpMI = *I;
      unsigned RmOpc = getCmpRmForCmpMr(CmpMI.getOpcode());
      if (!RmOpc) {
        ++I;
        continue;
      }

      // CMP mr operands: [base(0), scale(1), index(2), disp(3), seg(4), src_reg(5)]
      if (CmpMI.getNumOperands() < 6) {
        ++I;
        continue;
      }

      Register CmpReg = CmpMI.getOperand(5).getReg();
      unsigned ExpectedMovOpc = getMovRmForCmpMr(CmpMI.getOpcode());

      // Walk backward from CMP to find the MOV that loaded CmpReg.
      // Skip debug instructions. Stop if we see another def of CmpReg
      // or a use of CmpReg (other than the MOV def) that would make
      // the transformation unsafe.
      MachineInstr *MovMI = nullptr;
      auto Search = I;
      while (Search != MBB.begin()) {
        --Search;
        MachineInstr &Candidate = *Search;
        if (Candidate.isDebugInstr())
          continue;

        // Check if this instruction defines CmpReg.
        bool DefinesReg = false;
        for (const MachineOperand &MO : Candidate.operands()) {
          if (MO.isReg() && MO.isDef() && MO.getReg() == CmpReg) {
            DefinesReg = true;
            break;
          }
        }

        if (DefinesReg) {
          // Found a def. Check if it's the MOV we're looking for.
          if (Candidate.getOpcode() == ExpectedMovOpc &&
              Candidate.getOperand(0).getReg() == CmpReg) {
            MovMI = &Candidate;
          }
          // Whether or not it's our MOV, stop searching - we hit a def.
          break;
        }

        // If the candidate uses CmpReg in a way that would be affected by
        // changing the value in CmpReg, bail out.
        for (const MachineOperand &MO : Candidate.operands()) {
          if (MO.isReg() && MO.isUse() && MO.getReg() == CmpReg) {
            // CmpReg is used between the MOV and CMP. Unsafe to transform.
            goto next_cmp;
          }
        }
      }

      if (!MovMI) {
        ++I;
        continue;
      }

      {
        // Verify the MOV loads from a different memory location than the CMP's
        // memory operand. If they're the same, there's nothing to swap.
        // MOV rm operands: [dst(0), base(1), scale(2), index(3), disp(4), seg(5)]
        // CMP mr operands: [base(0), scale(1), index(2), disp(3), seg(4), src_reg(5)]
        if (sameMemOperands(CmpMI, 0, *MovMI, 1)) {
          ++I;
          continue;
        }

        // Safety check: the CMP's memory operands must not use CmpReg as a
        // base or index register (since we're changing what's in CmpReg, the
        // address computation would break). This is unlikely for the patterns
        // we care about (both operands use the same base like ECX for `this`),
        // but we check the CMP's memory base/index against CmpReg.
        // CMP mr: base(0), scale(1), index(2), disp(3), seg(4)
        if (CmpMI.getOperand(0).isReg() &&
            CmpMI.getOperand(0).getReg() == CmpReg) {
          ++I;
          continue;
        }
        if (CmpMI.getOperand(2).isReg() &&
            CmpMI.getOperand(2).getReg() == CmpReg) {
          ++I;
          continue;
        }

        // Also check the MOV's memory base/index - these form the new CMP's
        // memory operand, and they must not use CmpReg either.
        // MOV rm: base(1), index(3)
        if (MovMI->getOperand(1).isReg() &&
            MovMI->getOperand(1).getReg() == CmpReg) {
          ++I;
          continue;
        }
        if (MovMI->getOperand(3).isReg() &&
            MovMI->getOperand(3).getReg() == CmpReg) {
          ++I;
          continue;
        }

        // All checks passed. Build the new instructions.
        DebugLoc DL = CmpMI.getDebugLoc();

        // Build new MOV: load from the CMP's memory operand (LHS = mem_A).
        // MOV rm: [dst_reg, base, scale, index, disp, seg]
        auto NewMov = BuildMI(MBB, *MovMI, MovMI->getDebugLoc(),
                              TII->get(ExpectedMovOpc))
            .addReg(CmpReg, RegState::Define);
        // Copy the 5 memory operands from the CMP mr (indices 0-4).
        for (unsigned i = 0; i < 5; ++i)
          NewMov.add(CmpMI.getOperand(i));
        // Memory refs from CMP (the address we're now loading from).
        NewMov.cloneMemRefs(CmpMI);

        // Build new CMP rm: compare reg against old MOV's memory (RHS = mem_B).
        // CMP rm: [src_reg, base, scale, index, disp, seg]
        auto NewCmp = BuildMI(MBB, CmpMI, DL, TII->get(RmOpc))
            .addReg(CmpReg);
        // Copy the 5 memory operands from the old MOV rm (indices 1-5).
        for (unsigned i = 1; i <= 5; ++i)
          NewCmp.add(MovMI->getOperand(i));
        // Memory refs from MOV (the address we're now comparing against).
        NewCmp.cloneMemRefs(*MovMI);

        // Erase the old MOV and CMP.
        auto NextI = std::next(I);
        CmpMI.eraseFromParent();
        MovMI->eraseFromParent();
        I = NextI;
        Changed = true;
        continue;
      }

next_cmp:
      ++I;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86CmpLhsRegPass() {
  return new X86CmpLhsRegPass();
}

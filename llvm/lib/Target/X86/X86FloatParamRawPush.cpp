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
// Must run AFTER the FP stackifier (which converts FP pseudos to real x87
// instructions) and among the other bw1-decomp passes.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

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

/// Skip pseudo-instructions when walking forward.
static MachineBasicBlock::iterator
skipPseudos(MachineBasicBlock::iterator I, MachineBasicBlock::iterator E) {
  while (I != E && I->isPseudo())
    ++I;
  return I;
}

bool X86FloatParamRawPushPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("float_param_raw_push"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Compute liveness to find a free GPR for the intermediate mov.
    LivePhysRegs LiveRegs(*TRI);
    LiveRegs.addLiveOuts(MBB);

    // Walk backwards so liveness is tracked correctly.
    SmallVector<std::pair<MachineInstr *, MachineInstr *>, 4> ToReplace;

    for (auto I = MBB.rbegin(), E = MBB.rend(); I != E; ++I) {
      MachineInstr &MI = *I;
      LiveRegs.stepBackward(MI);

      // Look for ST_FP32m (fstp dword ptr [mem])
      if (MI.getOpcode() != X86::ST_FP32m)
        continue;

      // Check if the previous non-pseudo instruction is LD_F32m
      auto PrevIt = std::next(I); // reverse iterator: next = earlier instruction
      while (PrevIt != E && PrevIt->isPseudo())
        ++PrevIt;
      if (PrevIt == E)
        continue;
      MachineInstr &Prev = *PrevIt;
      if (Prev.getOpcode() != X86::LD_F32m)
        continue;

      // Found fld+fstp pair. Record for replacement.
      ToReplace.push_back({&Prev, &MI});
    }

    // Apply replacements (forward order since we collected in reverse).
    for (auto &[Fld, Fstp] : ToReplace) {
      // Re-check liveness at the fld point to pick a free GPR.
      // Prefer EDX (MSVC 6.0 convention for float forwarding), then EAX, ECX.
      // We need to compute liveness at the fld instruction.
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

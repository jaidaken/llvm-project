//===--- X86PreferStackCmpLhs.cpp - CMP32rm -> CMP32mr for stack ops ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: MSVC 6.0 generates `cmp [esp+N], reg` (CMP32mr with stack
// source) while Clang generates `cmp reg, [esp+N]` (CMP32rm).
//
// This pass converts CMP32rm reg, [ESP+disp] to CMP32mr [ESP+disp], reg
// and flips any following Jcc/SETcc/CMOVcc condition code.
//
// Gated on the "prefer_stack_cmp_lhs" function attribute.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-stack-cmp-lhs"

namespace {

/// Swap condition code for reversed CMP operands.
static X86::CondCode swapConditionCode(X86::CondCode CC) {
  switch (CC) {
  case X86::COND_E:  return X86::COND_E;
  case X86::COND_NE: return X86::COND_NE;
  case X86::COND_L:  return X86::COND_G;
  case X86::COND_LE: return X86::COND_GE;
  case X86::COND_G:  return X86::COND_L;
  case X86::COND_GE: return X86::COND_LE;
  case X86::COND_B:  return X86::COND_A;
  case X86::COND_BE: return X86::COND_AE;
  case X86::COND_A:  return X86::COND_B;
  case X86::COND_AE: return X86::COND_BE;
  case X86::COND_S:  return X86::COND_S;
  case X86::COND_NS: return X86::COND_NS;
  case X86::COND_P:  return X86::COND_P;
  case X86::COND_NP: return X86::COND_NP;
  case X86::COND_O:  return X86::COND_O;
  case X86::COND_NO: return X86::COND_NO;
  default: return X86::COND_INVALID;
  }
}

/// Check if an instruction reads EFLAGS with a condition code that needs
/// swapping. Returns the operand index of the condition code, or -1.
static int getCondOperandIdx(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  if (Opc == X86::JCC_1 || Opc == X86::JCC_4)
    return 1;
  if (Opc == X86::SETCCr || Opc == X86::SETCCm)
    return MI.getDesc().getNumDefs();
  if (Opc == X86::CMOV32rr || Opc == X86::CMOV16rr || Opc == X86::CMOV64rr)
    return MI.getNumOperands() - 1;
  return -1;
}

/// Check if an instruction defines EFLAGS.
static bool definesEFLAGS(const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isReg() && MO.isDef() && MO.getReg() == X86::EFLAGS)
      return true;
  }
  return false;
}

/// Returns true if the CMP32rm uses ESP as its base register.
/// CMP32rm operands: [src_reg(0), base(1), scale(2), index(3), disp(4), seg(5)]
static bool cmpRmUsesEsp(const MachineInstr &MI) {
  return MI.getOperand(1).isReg() &&
         MI.getOperand(1).getReg() == X86::ESP;
}

class X86PreferStackCmpLhsPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferStackCmpLhsPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 prefer stack CMP LHS for MSVC 6.0";
  }
};
} // end anonymous namespace

char X86PreferStackCmpLhsPass::ID = 0;

bool X86PreferStackCmpLhsPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("prefer_stack_cmp_lhs"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E; ) {
      MachineInstr &MI = *I;

      // Match CMP32rm with ESP base.
      if (MI.getOpcode() != X86::CMP32rm || !cmpRmUsesEsp(MI)) {
        ++I;
        continue;
      }

      // CMP32rm operands: [src_reg(0), base(1), scale(2), index(3), disp(4), seg(5)]
      Register SrcReg = MI.getOperand(0).getReg();

      // Build CMP32mr: [base(0), scale(1), index(2), disp(3), seg(4), src_reg(5)]
      DebugLoc DL = MI.getDebugLoc();
      auto MIB = BuildMI(MBB, MI, DL, TII->get(X86::CMP32mr));
      // Copy 5 memory operands from positions 1-5 of CMP32rm.
      for (unsigned i = 1; i <= 5; ++i)
        MIB.add(MI.getOperand(i));
      MIB.addReg(SrcReg);
      MIB.cloneMemRefs(MI);

      // Flip condition codes on following instructions that read EFLAGS.
      auto Next = std::next(I);
      while (Next != E) {
        MachineInstr &NextMI = *Next;
        if (NextMI.isDebugInstr()) {
          ++Next;
          continue;
        }
        if (definesEFLAGS(NextMI))
          break;
        int CondIdx = getCondOperandIdx(NextMI);
        if (CondIdx >= 0) {
          auto OldCC = static_cast<X86::CondCode>(
              NextMI.getOperand(CondIdx).getImm());
          X86::CondCode NewCC = swapConditionCode(OldCC);
          if (NewCC != X86::COND_INVALID)
            NextMI.getOperand(CondIdx).setImm(NewCC);
        }
        ++Next;
      }

      // Remove original CMP32rm and advance.
      auto NextI = std::next(I);
      MI.eraseFromParent();
      I = NextI;
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferStackCmpLhsPass() {
  return new X86PreferStackCmpLhsPass();
}

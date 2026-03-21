//===------- X86PreferMovImm.cpp - Convert XOR+INC to MOV imm -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: This pass converts XOR32rr self-xor + INC32r on the same
// register into MOV32ri reg, 1. This matches MSVC 6.0's code generation
// pattern of "mov eax, 1" instead of Clang -Oz's "xor edi, edi; inc edi"
// for small constant loads.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-mov-imm"
#define X86_PREFER_MOV_IMM_NAME "X86 prefer MOV immediate pass"

namespace {
class X86PreferMovImmPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferMovImmPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return X86_PREFER_MOV_IMM_NAME; }
};
} // end anonymous namespace

char X86PreferMovImmPass::ID = 0;

static bool isSelfXor32(const MachineInstr &MI) {
  if (MI.getOpcode() != X86::XOR32rr && MI.getOpcode() != X86::XOR32rr_REV)
    return false;
  Register Dst = MI.getOperand(0).getReg();
  Register Src1 = MI.getOperand(1).getReg();
  Register Src2 = MI.getOperand(2).getReg();
  return Dst == Src1 && Dst == Src2;
}

bool X86PreferMovImmPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("prefer_mov_imm"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    SmallVector<std::pair<MachineInstr *, MachineInstr *>, 4> ToReplace;

    for (auto I = MBB.begin(), E = MBB.end(); I != E; ++I) {
      MachineInstr &XorMI = *I;
      if (!isSelfXor32(XorMI))
        continue;

      Register Reg = XorMI.getOperand(0).getReg();

      auto Next = std::next(I);
      while (Next != E && Next->isDebugInstr())
        ++Next;
      if (Next == E)
        continue;

      MachineInstr &NextMI = *Next;
      if (NextMI.getOpcode() == X86::INC32r &&
          NextMI.getOperand(0).getReg() == Reg) {
        ToReplace.push_back({&XorMI, &NextMI});
      }
    }

    for (auto &[XorMI, IncMI] : ToReplace) {
      Register Reg = XorMI->getOperand(0).getReg();
      DebugLoc DL = XorMI->getDebugLoc();

      BuildMI(MBB, *XorMI, DL, TII->get(X86::MOV32ri), Reg)
          .addImm(1);

      IncMI->eraseFromParent();
      XorMI->eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferMovImmPass() {
  return new X86PreferMovImmPass();
}

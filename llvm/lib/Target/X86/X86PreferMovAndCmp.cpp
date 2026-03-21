//===--- X86PreferMovAndCmp.cpp - Expand test [mem],imm to mov+and+cmp ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: MSVC 6.0 generates bitfield checks as:
//   mov al, byte ptr [ecx+N]; and al, mask; cmp al, mask; je/jne
//
// The compiler generates:
//   test byte ptr [ecx+N], mask; je/jne
//
// This pass expands TEST8mi (test byte [mem], imm) into:
//   MOV8rm AL, [mem]; AND8ri AL, imm; CMP8ri AL, imm; JCC with inverted CC
//
// The condition inversion is needed because:
//   test [mem], mask; jne = "jump if any bit set"
//   and al, mask; cmp al, mask; je = "jump if all tested bits match"
// These are equivalent when the mask has a single bit set.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-mov-and-cmp"

namespace {
class X86PreferMovAndCmpPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferMovAndCmpPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 prefer mov+and+cmp over test [mem],imm";
  }
};
} // end anonymous namespace

char X86PreferMovAndCmpPass::ID = 0;

bool X86PreferMovAndCmpPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::Msvc6RegAlloc))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E; ) {
      MachineInstr &MI = *I;

      // Match TEST8mi: test byte ptr [mem], imm8
      if (MI.getOpcode() != X86::TEST8mi) {
        ++I;
        continue;
      }

      // TEST8mi operands: base, scale, index, disp, segment, imm
      // Operands 0-4 are the memory address, operand 5 is the immediate
      if (MI.getNumOperands() < 6 || !MI.getOperand(5).isImm()) {
        ++I;
        continue;
      }

      int64_t Mask = MI.getOperand(5).getImm();
      DebugLoc DL = MI.getDebugLoc();

      // Build: MOV8rm AL, [mem]
      auto MovMI = BuildMI(MBB, MI, DL, TII->get(X86::MOV8rm), X86::AL);
      for (unsigned i = 0; i < 5; i++)
        MovMI.add(MI.getOperand(i));
      MovMI.cloneMemRefs(MI);

      // Build: AND8ri AL, mask (implicit def EFLAGS)
      BuildMI(MBB, MI, DL, TII->get(X86::AND8ri), X86::AL)
          .addReg(X86::AL)
          .addImm(Mask);

      // Build: CMP8ri AL, mask
      BuildMI(MBB, MI, DL, TII->get(X86::CMP8ri))
          .addReg(X86::AL)
          .addImm(Mask);

      // The TEST sets ZF=1 when (mem & mask)==0.
      // AND+CMP sets ZF=1 when (mem & mask)==mask.
      // For single-bit masks these are different conditions:
      //   test+jne ("any bit set") = and+cmp+je ("bits match mask")
      // So we need to invert the following JCC condition.
      auto NextI = std::next(I);
      if (NextI != E && NextI->getOpcode() == X86::JCC_1) {
        int64_t OldCC = NextI->getOperand(1).getImm();
        int64_t NewCC = OldCC;
        if (OldCC == X86::COND_NE) NewCC = X86::COND_E;
        else if (OldCC == X86::COND_E) NewCC = X86::COND_NE;
        NextI->getOperand(1).setImm(NewCC);
      }

      // Remove original TEST8mi
      auto ToRemove = I;
      ++I;
      ToRemove->eraseFromParent();

      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferMovAndCmpPass() {
  return new X86PreferMovAndCmpPass();
}

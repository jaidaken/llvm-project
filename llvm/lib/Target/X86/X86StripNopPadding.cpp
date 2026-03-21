//===--- X86StripNopPadding.cpp - Remove NOP alignment padding -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: Removes NOP alignment padding from functions with the
// msvc6_regalloc attribute. MSVC 6.0 does not align loop headers, but
// LLVM inserts NOP padding to align blocks to 16-byte boundaries.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
using namespace llvm;

#define DEBUG_TYPE "x86-strip-nop-padding"
#define X86_STRIP_NOP_PADDING_NAME "X86 strip NOP padding pass"

namespace {
class X86StripNopPaddingPass : public MachineFunctionPass {
public:
  static char ID;
  X86StripNopPaddingPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return X86_STRIP_NOP_PADDING_NAME; }
};
} // end anonymous namespace

char X86StripNopPaddingPass::ID = 0;

bool X86StripNopPaddingPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::Msvc6RegAlloc))
    return false;

  bool Changed = false;

  // Remove alignment from all basic blocks. This prevents the MC layer
  // from inserting NOP padding before block labels.
  for (MachineBasicBlock &MBB : MF) {
    if (MBB.getAlignment() != Align(1)) {
      MBB.setAlignment(Align(1));
      Changed = true;
    }
  }

  // Also remove any explicit NOOP instructions
  SmallVector<MachineInstr *, 32> ToRemove;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.getOpcode() == X86::NOOP || MI.getOpcode() == X86::NOOPL ||
          MI.getOpcode() == X86::NOOPW)
        ToRemove.push_back(&MI);
    }
  }
  for (MachineInstr *MI : ToRemove) {
    MI->eraseFromParent();
    Changed = true;
  }

  return Changed;
}

FunctionPass *llvm::createX86StripNopPaddingPass() {
  return new X86StripNopPaddingPass();
}

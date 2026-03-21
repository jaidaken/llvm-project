//===--- X86MergeReturnZero.cpp - Merge bare ret into xor eax; ret --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: When PreferBranchBool creates separate "false" blocks, multiple
// return-0 paths may exist: a bare "ret" block and an "xor eax,eax; ret"
// block. MSVC shares a single xor+ret epilogue for all return-0 paths.
//
// This pass finds bare-ret blocks and redirects their predecessors to the
// nearest xor-eax+ret block, then removes the bare-ret block.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
using namespace llvm;

#define DEBUG_TYPE "x86-merge-return-zero"

namespace {
class X86MergeReturnZeroPass : public MachineFunctionPass {
public:
  static char ID;
  X86MergeReturnZeroPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 merge bare ret into xor eax,eax; ret";
  }
};
} // end anonymous namespace

char X86MergeReturnZeroPass::ID = 0;

bool X86MergeReturnZeroPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::Msvc6RegAlloc))
    return false;

  bool Changed = false;

  // Find a "return-zero" block: xor eax,eax; ret (exactly 2 instructions)
  MachineBasicBlock *RetZeroMBB = nullptr;
  for (MachineBasicBlock &MBB : MF) {
    if (MBB.size() != 2) continue;
    auto First = MBB.begin();
    auto Second = std::next(First);
    bool isXorEax = (First->getOpcode() == X86::XOR32rr ||
                     First->getOpcode() == X86::XOR32rr_REV) &&
                    First->getOperand(0).getReg() == X86::EAX;
    bool isRet = Second->isReturn();
    if (isXorEax && isRet) {
      RetZeroMBB = &MBB;
      break;
    }
  }

  if (!RetZeroMBB) return false;

  // Find bare-ret blocks: exactly 1 instruction (ret) with no other code
  SmallVector<MachineBasicBlock *, 4> BareRetBlocks;
  for (MachineBasicBlock &MBB : MF) {
    if (&MBB == RetZeroMBB) continue;
    if (MBB.size() != 1) continue;
    if (MBB.front().isReturn())
      BareRetBlocks.push_back(&MBB);
  }

  for (MachineBasicBlock *BareRet : BareRetBlocks) {
    // Redirect all predecessors to the return-zero block
    SmallVector<MachineBasicBlock *, 4> Preds(BareRet->predecessors());
    for (MachineBasicBlock *Pred : Preds) {
      // Update branch targets
      for (MachineInstr &MI : *Pred) {
        for (unsigned i = 0; i < MI.getNumOperands(); i++) {
          if (MI.getOperand(i).isMBB() &&
              MI.getOperand(i).getMBB() == BareRet)
            MI.getOperand(i).setMBB(RetZeroMBB);
        }
      }
      Pred->replaceSuccessor(BareRet, RetZeroMBB);
    }

    // Remove the bare-ret block
    while (!BareRet->succ_empty())
      BareRet->removeSuccessor(BareRet->succ_begin());
    BareRet->eraseFromParent();
    Changed = true;
  }

  return Changed;
}

FunctionPass *llvm::createX86MergeReturnZeroPass() {
  return new X86MergeReturnZeroPass();
}

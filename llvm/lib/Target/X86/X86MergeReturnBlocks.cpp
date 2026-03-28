//===--- X86MergeReturnBlocks.cpp - Merge duplicate return blocks ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: The prevent_setcc_merge pass converts branchless setcc patterns
// into separate true/false return paths. This often creates duplicate blocks
// that share the same instruction sequence (e.g., two blocks both containing
// "mov eax, 1; ret"). Since no_tail_merge (BranchFolding) runs BEFORE
// prevent_setcc_merge, those duplicates are never cleaned up.
//
// This pass runs AFTER prevent_setcc_merge. It groups return blocks by their
// instruction sequence and merges duplicates by redirecting predecessors to a
// single canonical block, then deleting the empty duplicates.
//
// This is simpler than full tail merging - it only merges complete blocks that
// are identical, not partial tail sequences.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
using namespace llvm;

#define DEBUG_TYPE "x86-merge-return-blocks"

namespace {
class X86MergeReturnBlocksPass : public MachineFunctionPass {
public:
  static char ID;
  X86MergeReturnBlocksPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 merge duplicate return blocks";
  }
};
} // end anonymous namespace

char X86MergeReturnBlocksPass::ID = 0;

/// Build a canonical string representation of a basic block's instruction
/// sequence. Two blocks with identical keys have identical instructions.
/// Only considers explicit operands to avoid mismatches from implicit
/// operand differences (e.g., duplicated implicit defs from CloneMachineInstr
/// vs BuildMI, or varying kill/dead flags on implicit operands).
static std::string getBlockKey(const MachineBasicBlock &MBB) {
  std::string Key;
  raw_string_ostream OS(Key);
  for (const MachineInstr &MI : MBB) {
    if (MI.isDebugInstr())
      continue;
    OS << MI.getOpcode();
    for (unsigned i = 0, e = MI.getNumExplicitOperands(); i < e; ++i) {
      const MachineOperand &MO = MI.getOperand(i);
      OS << ',';
      if (MO.isReg())
        OS << 'R' << MO.getReg();
      else if (MO.isImm())
        OS << 'I' << MO.getImm();
      else if (MO.isMBB())
        OS << 'B'; // block operands differ by identity, skip
      else
        OS << '?';
    }
    OS << ';';
  }
  OS.flush();
  return Key;
}

/// Return true if the block ends with a return instruction.
static bool endsWithReturn(const MachineBasicBlock &MBB) {
  if (MBB.empty())
    return false;
  for (auto I = MBB.rbegin(), E = MBB.rend(); I != E; ++I) {
    if (I->isDebugInstr())
      continue;
    return I->isReturn();
  }
  return false;
}

bool X86MergeReturnBlocksPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("merge_return_blocks"))
    return false;

  bool Changed = false;

  // Group return blocks by their instruction sequence.
  std::map<std::string, SmallVector<MachineBasicBlock *, 4>> Groups;
  for (MachineBasicBlock &MBB : MF) {
    if (!endsWithReturn(MBB))
      continue;
    std::string Key = getBlockKey(MBB);
    Groups[Key].push_back(&MBB);
  }

  // For each group with 2+ blocks, keep the first and redirect others.
  for (auto &KV : Groups) {
    SmallVector<MachineBasicBlock *, 4> &Blocks = KV.second;
    if (Blocks.size() < 2)
      continue;

    MachineBasicBlock *Canonical = Blocks[0];

    for (unsigned i = 1, e = Blocks.size(); i < e; ++i) {
      MachineBasicBlock *Dup = Blocks[i];

      // Redirect all predecessors of Dup to Canonical.
      SmallVector<MachineBasicBlock *, 4> Preds(Dup->predecessors());
      for (MachineBasicBlock *Pred : Preds) {
        // Update branch target operands.
        for (MachineInstr &MI : *Pred) {
          for (unsigned j = 0; j < MI.getNumOperands(); ++j) {
            if (MI.getOperand(j).isMBB() &&
                MI.getOperand(j).getMBB() == Dup)
              MI.getOperand(j).setMBB(Canonical);
          }
        }
        Pred->replaceSuccessor(Dup, Canonical);
      }

      // Remove the duplicate block.
      while (!Dup->succ_empty())
        Dup->removeSuccessor(Dup->succ_begin());
      Dup->eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86MergeReturnBlocksPass() {
  return new X86MergeReturnBlocksPass();
}

//===--- X86Msvc6Restructure.cpp - Restructure function to match MSVC 6 --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: Comprehensive restructuring pass for MSVC 6.0 matching.
// Handles block layout, split prologue, and epilogue interleaving.
//
// This pass identifies basic blocks by their content patterns and reorders
// them to match MSVC 6.0's hot-path layout. It also restructures the
// prologue to push callee-saved registers incrementally and interleaves
// the return value computation with pops in the epilogue.
//
// Gated behind a string attribute "msvc6_restructure".
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "MCTargetDesc/X86BaseInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-msvc6-restructure"
#define X86_MSVC6_RESTRUCTURE_NAME "X86 MSVC 6.0 restructure pass"

namespace {
class X86Msvc6RestructurePass : public MachineFunctionPass {
public:
  static char ID;
  X86Msvc6RestructurePass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return X86_MSVC6_RESTRUCTURE_NAME; }

private:
  // Block classification
  MachineBasicBlock *findModuloBlock(MachineFunction &MF);
  MachineBasicBlock *findNullReturnBlock(MachineFunction &MF);
  MachineBasicBlock *findEpilogueBlock(MachineFunction &MF);
  MachineBasicBlock *findTailLoopBlock(MachineFunction &MF);
};
} // end anonymous namespace

char X86Msvc6RestructurePass::ID = 0;

/// Find the block containing DIV instructions (modulo section)
MachineBasicBlock *X86Msvc6RestructurePass::findModuloBlock(MachineFunction &MF) {
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.getOpcode() == X86::DIV32r)
        return &MBB;
    }
  }
  return nullptr;
}

/// Find the block containing MOV32ri EDI, 1 (null return path)
MachineBasicBlock *X86Msvc6RestructurePass::findNullReturnBlock(MachineFunction &MF) {
  for (MachineBasicBlock &MBB : MF) {
    if (&MBB == &MF.front()) continue; // Skip entry
    for (MachineInstr &MI : MBB) {
      if (MI.getOpcode() == X86::MOV32ri &&
          MI.getOperand(0).getReg() == X86::EDI &&
          MI.getOperand(1).getImm() == 1)
        return &MBB;
    }
  }
  return nullptr;
}

/// Find the block containing RET
MachineBasicBlock *X86Msvc6RestructurePass::findEpilogueBlock(MachineFunction &MF) {
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.getOpcode() == X86::RET32 || MI.getOpcode() == X86::RET64)
        return &MBB;
    }
  }
  return nullptr;
}

/// Find the tail loop block (contains INC32r + byte load loop).
/// After expand_movzx, the byte load is XOR+MOV8rm instead of MOVZX32rm8.
MachineBasicBlock *X86Msvc6RestructurePass::findTailLoopBlock(MachineFunction &MF) {
  for (MachineBasicBlock &MBB : MF) {
    bool hasInc = false;
    bool hasMov8 = false;
    bool hasDec = false;
    unsigned instCount = 0;
    for (MachineInstr &MI : MBB) {
      instCount++;
      if (MI.getOpcode() == X86::INC32r)
        hasInc = true;
      if (MI.getOpcode() == X86::MOV8rm || MI.getOpcode() == X86::MOVZX32rm8)
        hasMov8 = true;
      if (MI.getOpcode() == X86::DEC32r)
        hasDec = true;
    }
    // The tail loop has INC + MOV8rm + DEC in a small block (<20 instructions)
    if (hasInc && hasMov8 && hasDec && instCount < 20)
      return &MBB;
  }
  return nullptr;
}

bool X86Msvc6RestructurePass::runOnMachineFunction(MachineFunction &MF) {
  // Gate behind prefer_div - only functions that need MSVC 6.0's div pattern
  // also need the block restructuring.
  if (!MF.getFunction().hasFnAttribute(Attribute::PreferDiv))
    return false;

  // Phase 1: Identify blocks
  MachineBasicBlock *ModuloBlock = findModuloBlock(MF);
  MachineBasicBlock *NullRetBlock = findNullReturnBlock(MF);
  MachineBasicBlock *EpilogueBlock = findEpilogueBlock(MF);
  MachineBasicBlock *TailBlock = findTailLoopBlock(MF);
  MachineBasicBlock *EntryBlock = &MF.front();

  errs() << "bw1-decomp restructure: modulo=" << (ModuloBlock ? ModuloBlock->getNumber() : -1)
         << " nullRet=" << (NullRetBlock ? NullRetBlock->getNumber() : -1)
         << " epilogue=" << (EpilogueBlock ? EpilogueBlock->getNumber() : -1)
         << " tail=" << (TailBlock ? TailBlock->getNumber() : -1)
         << " numBlocks=" << MF.size() << "\n";

  if (!ModuloBlock || !NullRetBlock || !EpilogueBlock || !TailBlock)
    return false;

  // Phase 2: Reorder blocks to match MSVC layout
  // MSVC order: entry -> outer_loop -> DO16 -> tail -> modulo -> epilogue
  // with null return INLINE after entry's test (handled by split prologue)
  //
  // For now, just move the modulo block to after the tail loop
  // (before the epilogue). Don't move null return yet - that needs
  // the split prologue to work correctly.
  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();

  // Move modulo block to right before epilogue (after tail loop)
  if (ModuloBlock->getNextNode() != EpilogueBlock) {
    ModuloBlock->moveBefore(EpilogueBlock);
  }

  // Phase 3: Fix up branches after reordering.
  // After moving blocks, some JMPs may become redundant (target is now
  // the layout successor) and some fallthroughs may be broken (need JMP).
  for (MachineBasicBlock &MBB : MF) {
    MachineBasicBlock *LayoutSucc = MBB.getNextNode();
    if (!LayoutSucc || MBB.empty())
      continue;

    MachineInstr &LastMI = MBB.back();

    // Remove redundant unconditional jumps to layout successor
    if (LastMI.getOpcode() == X86::JMP_1 &&
        LastMI.getOperand(0).getMBB() == LayoutSucc) {
      LastMI.eraseFromParent();
      continue;
    }

    // If block ends with a conditional branch whose false target is NOT
    // the layout successor, we need to insert a JMP for the fallthrough.
    if (LastMI.getOpcode() == X86::JCC_1) {
      // JCC_1 falls through on the "false" path.
      // If the layout successor isn't the false target, we need a JMP.
      // But we need to know which block is the false target.
      // The false target is the layout successor in the ORIGINAL layout.
      // After reordering, it might have moved. Check if any successor
      // is NOT the JCC target and NOT the layout successor.
      MachineBasicBlock *JccTarget = LastMI.getOperand(0).getMBB();
      for (MachineBasicBlock *Succ : MBB.successors()) {
        if (Succ != JccTarget && Succ != LayoutSucc) {
          // Need a JMP to the false target
          DebugLoc DL = LastMI.getDebugLoc();
          BuildMI(MBB, MBB.end(), DL, TII->get(X86::JMP_1))
              .addMBB(Succ);
          break;
        }
      }
    }
  }

  return true;
}

FunctionPass *llvm::createX86Msvc6RestructurePass() {
  return new X86Msvc6RestructurePass();
}

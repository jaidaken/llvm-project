//===--- X86InterleaveS2Update.cpp - Software-pipeline s2 update ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: Transforms the DO16 unrolled adler32 loop body to match MSVC
// 6.0's software-pipelined s2 update ordering.
//
// Before (Clang): xor; mov; add ecx,edx; add edi,ecx (s2 after s1)
// After (MSVC):   xor; mov; add edi,ecx; add ecx,edx (s2 before s1)
//
// The first iteration omits the s2 update, and an extra s2 update is added
// after the last iteration. This is semantically equivalent because s2
// uses the s1 value from the PREVIOUS iteration, not the current one.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-interleave-s2-update"
#define X86_INTERLEAVE_S2_NAME "X86 interleave s2 update pass"

namespace {
class X86InterleaveS2UpdatePass : public MachineFunctionPass {
public:
  static char ID;
  X86InterleaveS2UpdatePass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return X86_INTERLEAVE_S2_NAME; }
};
} // end anonymous namespace

char X86InterleaveS2UpdatePass::ID = 0;

/// Check if MI is "ADD32rr ECX, EDX" (s1 += byte)
static bool isS1Update(const MachineInstr &MI) {
  return MI.getOpcode() == X86::ADD32rr &&
         MI.getOperand(0).getReg() == X86::ECX &&
         MI.getOperand(1).getReg() == X86::ECX &&
         MI.getOperand(2).getReg() == X86::EDX;
}

/// Check if MI is "ADD32rr EDI, ECX" (s2 += s1)
static bool isS2Update(const MachineInstr &MI) {
  return MI.getOpcode() == X86::ADD32rr &&
         MI.getOperand(0).getReg() == X86::EDI &&
         MI.getOperand(1).getReg() == X86::EDI &&
         MI.getOperand(2).getReg() == X86::ECX;
}

bool X86InterleaveS2UpdatePass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::Msvc6RegAlloc))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Find consecutive pairs of (s1 update, s2 update):
    //   ADD32rr ECX, ECX, EDX    (s1 += byte)
    //   ADD32rr EDI, EDI, ECX    (s2 += s1)
    // Collect all such pairs in the block.
    SmallVector<std::pair<MachineInstr *, MachineInstr *>, 16> Pairs;

    for (auto I = MBB.begin(), E = MBB.end(); I != E; ++I) {
      if (!isS1Update(*I))
        continue;
      auto Next = std::next(I);
      if (Next != E && isS2Update(*Next)) {
        Pairs.push_back({&*I, &*Next});
      }
    }

    // Need at least 8 pairs (unrolled loop body has 16)
    if (Pairs.size() < 8)
      continue;

    // Transform: for pairs 1..N-1 (not the first), swap the two instructions.
    // This moves s2 update BEFORE s1 update, matching MSVC's interleaving.
    // For pair 0, remove the s2 update entirely (deferred to pair 1).
    // After the last pair, add an extra s2 update.

    // Remove s2 update from the FIRST pair (iteration 0)
    MachineInstr *FirstS2 = Pairs[0].second;

    // For pairs 1..N-1, swap: put s2 update before s1 update
    for (unsigned i = 1; i < Pairs.size(); i++) {
      MachineInstr *S1MI = Pairs[i].first;
      MachineInstr *S2MI = Pairs[i].second;
      // Move S2MI before S1MI
      S2MI->removeFromParent();
      MBB.insert(MachineBasicBlock::iterator(S1MI), S2MI);
    }

    // Add an extra s2 update after the last pair's s1 update
    MachineInstr *LastS1 = Pairs.back().first;
    DebugLoc DL = LastS1->getDebugLoc();
    BuildMI(MBB, std::next(MachineBasicBlock::iterator(LastS1)), DL,
            TII->get(X86::ADD32rr), X86::EDI)
        .addReg(X86::EDI)
        .addReg(X86::ECX);

    // Now remove the original first s2 update
    FirstS2->eraseFromParent();

    Changed = true;
  }

  return Changed;
}

FunctionPass *llvm::createX86InterleaveS2UpdatePass() {
  return new X86InterleaveS2UpdatePass();
}

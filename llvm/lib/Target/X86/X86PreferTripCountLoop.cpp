//===--- X86PreferTripCountLoop.cpp - Rewrite loop to use trip count ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: Rewrites loops from "add reg,-16; cmp reg,15; ja loop" to
// "dec ebp; jne loop" with trip count precomputation. Matches MSVC 6.0's
// loop counter pattern for unrolled loops.
//
// MSVC 6.0 precomputes: trip_count = k/16, remainder = k - trip_count*16
// Then loops with: dec ebp; jne loop_top
// Then uses remainder for the tail loop.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-trip-count-loop"
#define X86_PREFER_TRIP_COUNT_LOOP_NAME "X86 prefer trip count loop pass"

namespace {
class X86PreferTripCountLoopPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferTripCountLoopPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return X86_PREFER_TRIP_COUNT_LOOP_NAME;
  }
};
} // end anonymous namespace

char X86PreferTripCountLoopPass::ID = 0;

bool X86PreferTripCountLoopPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::PreferDiv))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Scan backwards from the end of the block looking for the pattern:
    //   ADD32ri8 REG, -16    (or ADD32ri REG, -16)
    //   CMP32ri8 REG, 15     (or CMP32ri REG, 15)
    //   JCC_1 target, COND_A
    MachineInstr *AddMI = nullptr;
    MachineInstr *CmpMI = nullptr;
    MachineInstr *JccMI = nullptr;
    MCPhysReg LoopReg = 0;
    MachineBasicBlock *LoopTarget = nullptr;

    for (auto I = MBB.rbegin(), E = MBB.rend(); I != E; ++I) {
      if (I->isDebugInstr())
        continue;

      if (!JccMI) {
        if (I->getOpcode() == X86::JCC_1 &&
            I->getOperand(1).getImm() == X86::COND_A) {
          JccMI = &*I;
          LoopTarget = I->getOperand(0).getMBB();
          continue;
        }
        break; // Not the pattern we're looking for
      }

      if (!CmpMI) {
        if ((I->getOpcode() == X86::CMP32ri8 ||
             I->getOpcode() == X86::CMP32ri) &&
            I->getOperand(1).getImm() == 15) {
          CmpMI = &*I;
          LoopReg = CmpMI->getOperand(0).getReg();
          continue;
        }
        break;
      }

      if (!AddMI) {
        if ((I->getOpcode() == X86::ADD32ri8 ||
             I->getOpcode() == X86::ADD32ri) &&
            I->getOperand(0).getReg() == LoopReg &&
            I->getOperand(2).getImm() == -16) {
          AddMI = &*I;
          break;
        }
        break;
      }
    }

    if (!AddMI || !CmpMI || !JccMI || !LoopTarget)
      continue;

    // Found the pattern. Now find the preheader - the block that falls
    // through to this loop body. The preheader should set up LoopReg.
    // Look for the predecessor that isn't LoopTarget (the backedge).
    MachineBasicBlock *Preheader = nullptr;
    for (MachineBasicBlock *Pred : MBB.predecessors()) {
      if (Pred != &MBB && Pred != LoopTarget) {
        // Check if this pred is the preheader (not the backedge source)
        // The backedge source is the current MBB itself or LoopTarget
        Preheader = Pred;
        break;
      }
    }

    // LoopTarget is the top of the loop. If we're in a single-block loop,
    // the backedge goes back to the same block. The preheader is a different
    // predecessor.
    if (!Preheader) {
      // If the loop target IS this MBB (single-block loop), find
      // the predecessor that isn't this MBB.
      if (LoopTarget == &MBB) {
        for (MachineBasicBlock *Pred : MBB.predecessors()) {
          if (Pred != &MBB) {
            Preheader = Pred;
            break;
          }
        }
      }
      if (!Preheader)
        continue;
    }

    // Check that EBP is not live in the loop body (we need it for trip count).
    // After the swap pass, EBP should be available.
    // Simple check: EBP shouldn't appear in any instruction in the loop body.
    bool ebpUsed = false;
    for (MachineInstr &MI : MBB) {
      for (const MachineOperand &MO : MI.operands()) {
        if (MO.isReg() && (MO.getReg() == X86::EBP || MO.getReg() == X86::BP))
          ebpUsed = true;
      }
    }
    if (ebpUsed)
      continue;

    // Insert trip count precomputation at the end of the preheader,
    // before the terminator.
    MachineBasicBlock::iterator InsertPt = Preheader->getFirstTerminator();
    if (InsertPt == Preheader->end())
      InsertPt = Preheader->end();
    DebugLoc DL = AddMI->getDebugLoc();

    // mov ebp, eax (or whatever LoopReg is - but we want EAX in the output)
    // After the swap pass, the loop counter should be in a specific register.
    // The trip count goes in EBP; the remainder stays in LoopReg (EAX ideally).
    //
    // MSVC pattern:
    //   mov ebp, eax
    //   shr ebp, 4
    //   mov edx, ebp
    //   neg edx
    //   shl edx, 4
    //   add eax, edx    ; eax = remainder
    //
    // We insert equivalent using LoopReg instead of EAX if different.

    // mov ebp, LoopReg
    BuildMI(*Preheader, InsertPt, DL, TII->get(X86::MOV32rr), X86::EBP)
        .addReg(LoopReg);
    // shr ebp, 4
    BuildMI(*Preheader, InsertPt, DL, TII->get(X86::SHR32ri), X86::EBP)
        .addReg(X86::EBP)
        .addImm(4);
    // mov edx, ebp
    BuildMI(*Preheader, InsertPt, DL, TII->get(X86::MOV32rr), X86::EDX)
        .addReg(X86::EBP);
    // neg edx
    BuildMI(*Preheader, InsertPt, DL, TII->get(X86::NEG32r), X86::EDX)
        .addReg(X86::EDX);
    // shl edx, 4
    BuildMI(*Preheader, InsertPt, DL, TII->get(X86::SHL32ri), X86::EDX)
        .addReg(X86::EDX)
        .addImm(4);
    // add LoopReg, edx (remainder = k + (-trip_count * 16))
    BuildMI(*Preheader, InsertPt, DL, TII->get(X86::ADD32rr), LoopReg)
        .addReg(LoopReg)
        .addReg(X86::EDX);

    // Replace the loop latch:
    //   Remove: add LoopReg, -16
    //   Remove: cmp LoopReg, 15
    //   Remove: ja LoopTarget
    //   Insert: dec ebp
    //   Insert: jne LoopTarget
    BuildMI(MBB, *AddMI, DL, TII->get(X86::DEC32r), X86::EBP)
        .addReg(X86::EBP);
    BuildMI(MBB, *AddMI, DL, TII->get(X86::JCC_1))
        .addMBB(LoopTarget)
        .addImm(X86::COND_NE);

    JccMI->eraseFromParent();
    CmpMI->eraseFromParent();
    AddMI->eraseFromParent();

    Changed = true;
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferTripCountLoopPass() {
  return new X86PreferTripCountLoopPass();
}

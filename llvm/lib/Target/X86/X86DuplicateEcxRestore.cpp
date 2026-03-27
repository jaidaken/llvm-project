//===--- X86DuplicateEcxRestore.cpp - Duplicate ECX restore into branches --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: Post-regalloc pass that duplicates a hoisted MOV ECX, ESI/EDI
// into each successor block of a conditional branch.
//
// LLVM hoists the ECX restore before the branch when both paths need it:
//
//   mov ecx, esi       ; eager ECX restore
//   test eax, eax
//   je .path2
// .path1:
//   push 0x6a
//   call [edx+0x8e8]
// .path2:
//   push 0x65
//   call [edx+0x8e8]
//
// MSVC 6.0 keeps the MOV ECX independently in each path:
//
//   test eax, eax
//   je .path2
// .path1:
//   push 0x6a
//   mov ecx, esi       ; ECX setup in path 1
//   call [edx+0x8e8]
// .path2:
//   push 0x65
//   mov ecx, esi       ; ECX setup in path 2
//   call [edx+0x8e8]
//
// This pass finds the pattern: MOV ECX, ESI/EDI as the last real instruction
// before a conditional branch (Jcc), where both the fall-through and branch
// target blocks contain CALL instructions. It removes the MOV from the
// pre-branch position and inserts a copy into each successor block just
// before the first CALL (after any PUSHes), matching the push-before-ecx
// logic from X86PreferPushBeforeEcx.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-duplicate-ecx-restore"

namespace {
class X86DuplicateEcxRestorePass : public MachineFunctionPass {
public:
  static char ID;
  X86DuplicateEcxRestorePass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 duplicate ECX restore across branches";
  }
};
} // end anonymous namespace

char X86DuplicateEcxRestorePass::ID = 0;

/// Check if MI is a MOV32rr/MOV32rr_REV that sets ECX from ESI, EDI, or EBX.
static bool isMovEcxFromCalleeSaved(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  if (Opc != X86::MOV32rr && Opc != X86::MOV32rr_REV)
    return false;
  if (MI.getNumOperands() < 2)
    return false;
  if (!MI.getOperand(0).isReg() || MI.getOperand(0).getReg() != X86::ECX)
    return false;
  if (!MI.getOperand(1).isReg())
    return false;
  Register SrcReg = MI.getOperand(1).getReg();
  return SrcReg == X86::ESI || SrcReg == X86::EDI || SrcReg == X86::EBX;
}

/// Check if MI is a conditional branch (JCC_1).
static bool isConditionalJump(const MachineInstr &MI) {
  return MI.getOpcode() == X86::JCC_1;
}

/// Check if MI is a PUSH instruction (any variant used for argument passing).
static bool isArgPush(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case X86::PUSH32i8:
  case X86::PUSH32i:
  case X86::PUSH32r:
  case X86::PUSH32rmm:
    return true;
  default:
    return false;
  }
}

/// Check if MI is any CALL instruction.
static bool isAnyCall(const MachineInstr &MI) {
  return MI.isCall();
}

/// Check if a block contains any CALL instruction.
static bool blockHasCall(const MachineBasicBlock &MBB) {
  for (const MachineInstr &MI : MBB) {
    if (isAnyCall(MI))
      return true;
  }
  return false;
}

/// Find the insertion point just before the first CALL in the block,
/// after any preceding PUSHes. This matches the push-before-ecx logic:
/// the MOV ECX goes after all PUSHes but before the CALL.
static MachineBasicBlock::iterator
findInsertionPoint(MachineBasicBlock &MBB) {
  for (auto I = MBB.begin(), E = MBB.end(); I != E; ++I) {
    if (!isAnyCall(*I))
      continue;

    // Found a CALL. The insertion point is just before it.
    // But we want to be after all PUSHes, which means right before CALL
    // is correct - the PUSHes precede the CALL and we insert between them.
    return I;
  }
  return MBB.end();
}

bool X86DuplicateEcxRestorePass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("duplicate_ecx_restore"))
    return false;

  const X86InstrInfo *TII =
      MF.getSubtarget<X86Subtarget>().getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Scan backwards from the end of the block looking for:
    //   MOV ECX, ESI/EDI  (last non-pseudo before branch)
    //   [pseudos]
    //   JCC_1 target
    //
    // The block must end with a conditional branch.

    // Find the terminator (conditional branch).
    MachineBasicBlock::iterator TermIt = MBB.end();
    for (auto I = MBB.end(), B = MBB.begin(); I != B;) {
      --I;
      if (I->isPseudo())
        continue;
      if (isConditionalJump(*I)) {
        TermIt = I;
        break;
      }
      // If the first real instruction from the end is not a Jcc, stop.
      break;
    }
    if (TermIt == MBB.end())
      continue;

    // Walk backwards from the Jcc to find the last real (non-pseudo) instruction.
    MachineBasicBlock::iterator MovIt = MBB.end();
    for (auto I = TermIt, B = MBB.begin(); I != B;) {
      --I;
      if (I->isPseudo())
        continue;
      if (isMovEcxFromCalleeSaved(*I)) {
        MovIt = I;
      }
      // Whether it's the MOV or not, stop - we only want the last real instr.
      break;
    }
    if (MovIt == MBB.end())
      continue;

    // The block must have exactly two successors: fall-through and branch target.
    if (MBB.succ_size() != 2)
      continue;

    // Get the two successor blocks.
    MachineBasicBlock *Succ0 = *MBB.succ_begin();
    MachineBasicBlock *Succ1 = *std::next(MBB.succ_begin());

    // Both successors must contain CALL instructions.
    if (!blockHasCall(*Succ0) || !blockHasCall(*Succ1))
      continue;

    // Find insertion points in both successors (just before the first CALL).
    MachineBasicBlock::iterator InsertPt0 = findInsertionPoint(*Succ0);
    MachineBasicBlock::iterator InsertPt1 = findInsertionPoint(*Succ1);
    if (InsertPt0 == Succ0->end() || InsertPt1 == Succ1->end())
      continue;

    // Capture the MOV details before erasing.
    unsigned MovOpc = MovIt->getOpcode();
    Register SrcReg = MovIt->getOperand(1).getReg();
    DebugLoc DL = MovIt->getDebugLoc();

    // Remove the original MOV ECX from the pre-branch block.
    MovIt->eraseFromParent();

    // Insert MOV ECX, SrcReg in each successor just before the CALL.
    BuildMI(*Succ0, InsertPt0, DL, TII->get(MovOpc), X86::ECX)
        .addReg(SrcReg);
    BuildMI(*Succ1, InsertPt1, DL, TII->get(MovOpc), X86::ECX)
        .addReg(SrcReg);

    Changed = true;
  }

  return Changed;
}

FunctionPass *llvm::createX86DuplicateEcxRestorePass() {
  return new X86DuplicateEcxRestorePass();
}

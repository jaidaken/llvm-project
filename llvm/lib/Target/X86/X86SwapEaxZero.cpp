//===--- X86SwapEaxZero.cpp - Swap EAX away from pre-materialized zero ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: Post-regalloc pass that fixes EAX pre-zeroing for return-0 paths.
//
// Clang's register allocator pre-materializes `xor eax, eax` early for
// `return 0` paths. This forces the first loaded value (flags, pointers) into
// EDX/ECX, breaking `test ah` patterns and changing instruction encodings.
//
// MSVC 6.0 keeps the first loaded value in EAX and only zeroes EAX at the
// actual `return 0` point.
//
// This pass detects the pattern in the entry block:
//   $rx = MOV32rm [mem]       ; first load (should be EAX)
//   $eax = XOR32rr            ; pre-zero for return 0
//   TEST/CMP $rx, imm         ; test the loaded value
//   JCC_1 bb.ret, cc          ; conditional branch to return block
//
// And transforms it by:
//   1. Changing the load destination to EAX
//   2. Removing the XOR
//   3. Replacing uses of $rx with EAX in all blocks until $rx is redefined
//   4. Redirecting "return 0" branches to new blocks with XOR EAX,EAX + RET
//
// Gated on the "swap_eax_zero" function attribute.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-swap-eax-zero"
#define PASS_NAME "X86 swap EAX away from pre-materialized zero"

namespace {
class X86SwapEaxZeroPass : public MachineFunctionPass {
public:
  static char ID;
  X86SwapEaxZeroPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return PASS_NAME; }
};
} // end anonymous namespace

char X86SwapEaxZeroPass::ID = 0;

/// Return true if MI is a return instruction.
static bool isRetInstr(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  return Opc == X86::RET || Opc == X86::RET32 || Opc == X86::RETI32 ||
         MI.isReturn();
}

/// Map a 32-bit register to its 16-bit, 8-bit low, and 8-bit high variants.
/// Returns false if the register has no sub-register variants.
static bool getSubRegs(unsigned Reg32, unsigned &Reg16, unsigned &RegLo,
                       unsigned &RegHi) {
  switch (Reg32) {
  case X86::EAX: Reg16 = X86::AX; RegLo = X86::AL; RegHi = X86::AH; return true;
  case X86::EDX: Reg16 = X86::DX; RegLo = X86::DL; RegHi = X86::DH; return true;
  case X86::ECX: Reg16 = X86::CX; RegLo = X86::CL; RegHi = X86::CH; return true;
  case X86::EBX: Reg16 = X86::BX; RegLo = X86::BL; RegHi = X86::BH; return true;
  default: return false;
  }
}

/// Replace all occurrences of OldReg (and its sub-registers) with NewReg
/// (and the corresponding sub-registers) in MI.
static bool replaceRegInMI(MachineInstr &MI, unsigned OldReg32,
                           unsigned NewReg32, const TargetRegisterInfo *TRI) {
  unsigned Old16, OldLo, OldHi, New16, NewLo, NewHi;
  if (!getSubRegs(OldReg32, Old16, OldLo, OldHi))
    return false;
  if (!getSubRegs(NewReg32, New16, NewLo, NewHi))
    return false;

  bool Changed = false;
  for (MachineOperand &MO : MI.operands()) {
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (Reg == OldReg32) { MO.setReg(NewReg32); Changed = true; }
    else if (Reg == Old16) { MO.setReg(New16); Changed = true; }
    else if (Reg == OldLo) { MO.setReg(NewLo); Changed = true; }
    else if (Reg == OldHi) { MO.setReg(NewHi); Changed = true; }
  }
  return Changed;
}

/// Check if MI defines (writes to) any register overlapping Reg32.
static bool definesReg(const MachineInstr &MI, unsigned Reg32,
                       const TargetRegisterInfo *TRI) {
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isReg() && MO.isDef() && MO.getReg() &&
        TRI->regsOverlap(MO.getReg(), Reg32))
      return true;
  }
  return false;
}

/// Check if MI uses (reads) any register overlapping Reg32.
static bool usesReg(const MachineInstr &MI, unsigned Reg32,
                    const TargetRegisterInfo *TRI) {
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isReg() && MO.isUse() && MO.getReg() &&
        TRI->regsOverlap(MO.getReg(), Reg32))
      return true;
  }
  return false;
}

bool X86SwapEaxZeroPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("swap_eax_zero"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();

  MachineBasicBlock &EntryMBB = MF.front();

  // Step 1: Find the first MOV32rm (non-EAX) and first XOR EAX,EAX in the
  // entry block before any branches.
  MachineInstr *FirstLoad = nullptr;
  MachineInstr *XorEax = nullptr;

  for (MachineInstr &MI : EntryMBB) {
    if (MI.isPseudo() || MI.isDebugInstr())
      continue;

    unsigned Opc = MI.getOpcode();

    // Look for XOR32rr/XOR32rr_REV self-xor zeroing EAX.
    if ((Opc == X86::XOR32rr || Opc == X86::XOR32rr_REV) &&
        MI.getOperand(1).getReg() == MI.getOperand(2).getReg() &&
        MI.getOperand(0).getReg() == X86::EAX) {
      XorEax = &MI;
      continue;
    }

    // Look for MOV32rm loading into a non-EAX register.
    if (Opc == X86::MOV32rm && !FirstLoad) {
      Register Dst = MI.getOperand(0).getReg();
      if (Dst != X86::EAX) {
        FirstLoad = &MI;
      }
      continue;
    }

    // Stop at first branch.
    if (MI.isBranch())
      break;
  }

  if (!FirstLoad || !XorEax)
    return false;

  Register LoadReg = FirstLoad->getOperand(0).getReg();
  unsigned XorOpcode = XorEax->getOpcode();
  DebugLoc DL = XorEax->getDebugLoc();

  // Step 2: Identify "return 0" branch targets from the entry block. These are
  // branches that go to blocks containing only POPs + RET.
  SmallVector<MachineInstr *, 4> ReturnZeroBranches;

  for (MachineInstr &MI : EntryMBB) {
    if (MI.getOpcode() != X86::JCC_1)
      continue;

    MachineBasicBlock *Target = MI.getOperand(0).getMBB();
    bool IsSimpleReturn = false;
    for (auto It = Target->begin(), End = Target->end(); It != End; ++It) {
      if (It->isPseudo() || It->isDebugInstr())
        continue;
      if (It->getOpcode() == X86::POP32r)
        continue;
      if (isRetInstr(*It)) {
        IsSimpleReturn = true;
        break;
      }
      break; // Non-trivial instruction found.
    }

    if (IsSimpleReturn)
      ReturnZeroBranches.push_back(&MI);
  }

  if (ReturnZeroBranches.empty())
    return false;

  // Step 3: Replace LoadReg with EAX in all instructions that use the loaded
  // flags value. Walk through each block in layout order. In each block,
  // replace LoadReg->EAX until we hit an instruction that defines LoadReg
  // (meaning LoadReg is being used for something else from here on).
  //
  // Also replace EAX->LoadReg in any instruction that defines EAX AND does
  // not use LoadReg, to avoid conflicts. But only do this in the entry block
  // for the XOR instruction itself (which we will remove).

  // First, handle the entry block.
  // Change the MOV32rm destination from LoadReg to EAX.
  FirstLoad->getOperand(0).setReg(X86::EAX);

  // Remove the XOR EAX,EAX.
  XorEax->eraseFromParent();

  // Replace LoadReg->EAX in remaining entry block instructions (after the
  // load, before branches or until LoadReg is redefined).
  bool PastLoad = false;
  for (MachineInstr &MI : EntryMBB) {
    if (&MI == FirstLoad) {
      PastLoad = true;
      continue;
    }
    if (!PastLoad)
      continue;
    if (MI.isPseudo() || MI.isDebugInstr())
      continue;

    // Replace LoadReg->EAX in this instruction.
    replaceRegInMI(MI, LoadReg, X86::EAX, TRI);

    // If this instruction defines LoadReg, stop replacing in this block.
    if (definesReg(MI, LoadReg, TRI))
      break;
  }

  // Now handle successor blocks: replace LoadReg->EAX until LoadReg is
  // redefined. This handles the case where the flags value flows into
  // successor blocks (e.g., bb.1 uses TEST dx, dx -> TEST ax, ax).
  for (MachineBasicBlock &MBB : MF) {
    if (&MBB == &EntryMBB)
      continue;

    for (MachineInstr &MI : MBB) {
      if (MI.isPseudo() || MI.isDebugInstr())
        continue;

      // If this instruction defines EAX (e.g., MOV32rm EAX, [mem] loading
      // the pointer), stop. From here on, EAX holds a different value.
      // But first check if it uses LoadReg - if so, replace before stopping.
      if (usesReg(MI, LoadReg, TRI)) {
        replaceRegInMI(MI, LoadReg, X86::EAX, TRI);
      }

      // Stop at the first definition of either LoadReg or EAX.
      if (definesReg(MI, LoadReg, TRI) || definesReg(MI, X86::EAX, TRI))
        break;
    }
  }

  // Step 4: Redirect "return 0" branches to new blocks with XOR EAX,EAX + RET.
  for (MachineInstr *Branch : ReturnZeroBranches) {
    MachineBasicBlock *OrigTarget = Branch->getOperand(0).getMBB();
    MachineBasicBlock *BranchParent = Branch->getParent();

    // Create a new return-0 block at the end of the function to avoid
    // disrupting fall-through paths between existing blocks.
    MachineBasicBlock *Ret0MBB = MF.CreateMachineBasicBlock();
    Ret0MBB->setLabelMustBeEmitted();
    MF.push_back(Ret0MBB);

    // Build XOR EAX, EAX.
    BuildMI(*Ret0MBB, Ret0MBB->end(), DL, TII->get(XorOpcode), X86::EAX)
        .addReg(X86::EAX, RegState::Undef)
        .addReg(X86::EAX, RegState::Undef);

    // Clone POP instructions and RET from the original target.
    for (MachineInstr &MI : *OrigTarget) {
      if (MI.isPseudo() || MI.isDebugInstr())
        continue;
      MachineInstr *Clone = MF.CloneMachineInstr(&MI);
      // Fix the RET to use EAX.
      if (isRetInstr(*Clone)) {
        for (MachineOperand &MO : Clone->operands()) {
          if (MO.isReg() && MO.getReg() && TRI->regsOverlap(MO.getReg(), LoadReg))
            MO.setReg(X86::EAX);
        }
      }
      Ret0MBB->insert(Ret0MBB->end(), Clone);
      if (isRetInstr(MI))
        break;
    }

    // Redirect the branch.
    Branch->getOperand(0).setMBB(Ret0MBB);
    BranchParent->replaceSuccessor(OrigTarget, Ret0MBB);
  }

  return true;
}

FunctionPass *llvm::createX86SwapEaxZeroPass() {
  return new X86SwapEaxZeroPass();
}

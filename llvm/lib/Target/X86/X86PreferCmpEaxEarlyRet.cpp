//===--- X86PreferCmpEaxEarlyRet.cpp - CMP EAX + early return pattern ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: Clang generates a pattern where it saves a call result in a
// scratch register before overwriting EAX with the return value, then
// compares the scratch:
//
//   mov ecx, eax       ; save call result to scratch
//   mov eax, 1         ; preload return value
//   cmp ecx, 1         ; compare saved result
//   je .Lreturn         ; branch to shared return block
//
// MSVC 6.0 compares directly against EAX, then conditionally returns inline:
//
//   cmp eax, 1         ; compare directly
//   jne .Lskip         ; if NOT equal, skip
//   mov eax, 1         ; return value (only in taken path)
//   pop esi            ; inline epilogue
//   ret
//   .Lskip:
//
// This pass, gated on the "prefer_cmp_eax_early_ret" string attribute,
// rewrites:
//   MOV32rr scratch, EAX; MOV32ri EAX, imm; CMP32ri8 scratch, imm; JCC_1 je
// into:
//   CMP32ri8 EAX, imm; JCC_1 jne (inverted, swapped targets)
//
// The MOV32ri EAX is sunk into the fall-through block (where the branch
// would have gone).
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-cmp-eax-early-ret"

namespace {
class X86PreferCmpEaxEarlyRetPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferCmpEaxEarlyRetPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 prefer CMP EAX + early return";
  }
};
} // end anonymous namespace

char X86PreferCmpEaxEarlyRetPass::ID = 0;

/// Get the inverted condition code for a JCC.
static X86::CondCode invertCC(X86::CondCode CC) {
  switch (CC) {
  case X86::COND_E:  return X86::COND_NE;
  case X86::COND_NE: return X86::COND_E;
  case X86::COND_A:  return X86::COND_BE;
  case X86::COND_BE: return X86::COND_A;
  case X86::COND_B:  return X86::COND_AE;
  case X86::COND_AE: return X86::COND_B;
  case X86::COND_G:  return X86::COND_LE;
  case X86::COND_LE: return X86::COND_G;
  case X86::COND_L:  return X86::COND_GE;
  case X86::COND_GE: return X86::COND_L;
  default:           return X86::COND_INVALID;
  }
}

bool X86PreferCmpEaxEarlyRetPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("prefer_cmp_eax_early_ret"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E;) {
      MachineInstr &MovRR = *I;

      // Step 1: Match MOV32rr scratch, EAX (or MOV32rr_REV).
      if (MovRR.getOpcode() != X86::MOV32rr &&
          MovRR.getOpcode() != X86::MOV32rr_REV) {
        ++I;
        continue;
      }

      Register ScratchReg = MovRR.getOperand(0).getReg();
      Register SrcReg = MovRR.getOperand(1).getReg();
      if (!TRI->regsOverlap(SrcReg, X86::EAX)) {
        ++I;
        continue;
      }

      // Step 2: Next non-debug must be MOV32ri EAX, imm.
      auto MovImmIt = std::next(I);
      while (MovImmIt != E && MovImmIt->isDebugInstr())
        ++MovImmIt;
      if (MovImmIt == E) { ++I; continue; }

      MachineInstr &MovImm = *MovImmIt;
      if (MovImm.getOpcode() != X86::MOV32ri ||
          MovImm.getOperand(0).getReg() != X86::EAX) {
        ++I;
        continue;
      }
      int64_t ReturnImm = MovImm.getOperand(1).getImm();

      // Step 3: Next non-debug must be CMP32ri8 or CMP32ri scratch, imm.
      auto CmpIt = std::next(MovImmIt);
      while (CmpIt != E && CmpIt->isDebugInstr())
        ++CmpIt;
      if (CmpIt == E) { ++I; continue; }

      MachineInstr &CmpMI = *CmpIt;
      unsigned CmpOpc = CmpMI.getOpcode();
      if (CmpOpc != X86::CMP32ri8 && CmpOpc != X86::CMP32ri) {
        ++I;
        continue;
      }
      if (CmpMI.getOperand(0).getReg() != ScratchReg) {
        ++I;
        continue;
      }
      int64_t CmpImm = CmpMI.getOperand(1).getImm();

      // Step 4: Next non-debug must be JCC_1 or JCC_4.
      auto JccIt = std::next(CmpIt);
      while (JccIt != E && JccIt->isDebugInstr())
        ++JccIt;
      if (JccIt == E) { ++I; continue; }

      MachineInstr &JccMI = *JccIt;
      unsigned JccOpc = JccMI.getOpcode();
      if (JccOpc != X86::JCC_1 && JccOpc != X86::JCC_4) {
        ++I;
        continue;
      }

      // Get the condition code and invert it.
      X86::CondCode OrigCC =
          static_cast<X86::CondCode>(JccMI.getOperand(1).getImm());
      X86::CondCode InvCC = invertCC(OrigCC);
      if (InvCC == X86::COND_INVALID) {
        ++I;
        continue;
      }

      // Get the branch target and fall-through.
      MachineBasicBlock *BranchTarget = JccMI.getOperand(0).getMBB();
      MachineBasicBlock *FallThrough = MBB.getFallThrough();
      if (!FallThrough || !BranchTarget) {
        ++I;
        continue;
      }

      LLVM_DEBUG(dbgs() << "PreferCmpEaxEarlyRet: rewriting pattern in "
                        << MF.getName() << "\n");

      DebugLoc DL = MovRR.getDebugLoc();

      // Build: CMP32ri8 EAX, imm (compare directly against EAX).
      BuildMI(MBB, MovRR, DL, TII->get(CmpOpc))
          .addReg(X86::EAX)
          .addImm(CmpImm);

      // Build: JCC with inverted condition, targeting the old fall-through.
      // The old branch target (shared return block) becomes the new
      // fall-through, and the old fall-through becomes the new branch target.
      BuildMI(MBB, MovRR, DL, TII->get(JccOpc))
          .addMBB(FallThrough)
          .addImm(InvCC);

      // Sink the MOV32ri EAX, ReturnImm into the old branch target
      // (the shared return block that is now the fall-through).
      BuildMI(*BranchTarget, BranchTarget->begin(), DL,
              TII->get(X86::MOV32ri), X86::EAX)
          .addImm(ReturnImm);

      // Remove the four original instructions.
      auto NextI = std::next(JccIt);
      JccMI.eraseFromParent();
      CmpMI.eraseFromParent();
      MovImm.eraseFromParent();
      MovRR.eraseFromParent();

      // Update block successors: swap targets.
      // The old branch target is now the fall-through, and the old
      // fall-through is now the branch target.
      // We need to update MBB's successor list to reflect the new layout.
      // The successors should still be the same two blocks - just the
      // branch condition changed. The successor list stays the same.

      I = NextI;
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferCmpEaxEarlyRetPass() {
  return new X86PreferCmpEaxEarlyRetPass();
}

//===--- X86PreferIntFloatForward.cpp - Float param as integer push --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: When forwarding a float parameter to another function, LLVM
// uses FPU instructions (sub esp,4; fld [esp+N]; fstp [esp]; call; sub esp,4).
// MSVC 6.0 treats the float as a raw 4-byte integer (mov eax,[esp+N]; push eax;
// call). This pass detects the FPU forwarding pattern and replaces it with
// the integer move+push sequence.
//
// Pattern:
//   SUB32ri/SUB32ri8 ESP, 4     ; ADJCALLSTACKDOWN
//   LD_F32m [ESP + N]           ; fld dword ptr [esp+N]
//   ST_FP32m [ESP + 0]          ; fstp dword ptr [esp]
//   CALL ...                    ; call
//   SUB32ri/SUB32ri8 ESP, 4     ; callee-pops compensation
//
// Becomes:
//   MOV32rm EAX, [ESP + (N-4)]  ; load float as integer (adjust for no sub)
//   PUSH32r EAX                 ; push (combines sub+store)
//   CALL ...                    ; call
//   ; compensation removed (push already grew stack)
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-int-float-forward"

namespace {
class X86PreferIntFloatForwardPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferIntFloatForwardPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 prefer integer float forward (fld+fstp -> mov+push)";
  }
};
} // end anonymous namespace

char X86PreferIntFloatForwardPass::ID = 0;

/// Skip CFI and other pseudo-instructions when walking forward.
static MachineBasicBlock::iterator
skipPseudosForward(MachineBasicBlock::iterator I,
                   MachineBasicBlock::iterator E) {
  while (I != E && I->isPseudo())
    ++I;
  return I;
}

bool X86PreferIntFloatForwardPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("no_tail_call") &&
      !MF.getFunction().hasFnAttribute(Attribute::NoCalleeSaves))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E; ) {
      MachineInstr &FldMI = *I;

      // Pattern: LLVM may fold the SUB ESP,4 into addressing modes.
      // Look for: LD_F32m [ESP+N]; ST_FP32m [ESP+0]; CALL; SUB ESP,4
      //
      // The fld reads param from [ESP+N] where N = actual_offset + 4
      // (because the stack was pre-adjusted by the frame lowering).
      // The fstp writes to [ESP+0] (the call argument slot).
      //
      // Replace with: MOV32rm EAX, [ESP+(N-4)]; PUSH32r EAX; CALL
      // Remove the post-call SUB ESP,4 compensation.

      // Step 1: Find LD_F32m [ESP + N]
      if (FldMI.getOpcode() != X86::LD_F32m) {
        ++I;
        continue;
      }
      if (!FldMI.getOperand(0).isReg() ||
          FldMI.getOperand(0).getReg() != X86::ESP) {
        ++I;
        continue;
      }
      int64_t FldDisp = FldMI.getOperand(3).getImm();

      // Step 2: Find ST_FP32m [ESP + 0] immediately after
      auto FstpIt = skipPseudosForward(std::next(I), E);
      if (FstpIt == E || FstpIt->getOpcode() != X86::ST_FP32m) {
        ++I;
        continue;
      }
      MachineInstr &FstpMI = *FstpIt;
      if (!FstpMI.getOperand(0).isReg() ||
          FstpMI.getOperand(0).getReg() != X86::ESP ||
          FstpMI.getOperand(3).getImm() != 0) {
        ++I;
        continue;
      }

      // Step 3: Find CALL after fstp
      auto CallIt = skipPseudosForward(std::next(FstpIt), E);
      if (CallIt == E || !CallIt->isCall()) {
        ++I;
        continue;
      }

      // Step 4: Find post-call SUB ESP, 4 (callee-pops compensation)
      auto NextI = std::next(MachineBasicBlock::iterator(*CallIt));
      auto PostCallIt = skipPseudosForward(NextI, E);
      bool HasPostSub = false;
      if (PostCallIt != E) {
        unsigned PostOpc = PostCallIt->getOpcode();
        HasPostSub = (PostOpc == X86::SUB32ri || PostOpc == X86::SUB32ri8) &&
                     PostCallIt->getOperand(0).getReg() == X86::ESP &&
                     PostCallIt->getOperand(2).getImm() == 4;
      }
      if (!HasPostSub) {
        ++I;
        continue;
      }

      // Pattern matched! Build replacement.
      DebugLoc DL = FldMI.getDebugLoc();

      // MOV32rm EAX, [ESP + (FldDisp - 4)]
      // The -4 adjusts for removing the pre-allocated call arg space.
      // With push instead of pre-allocated space, ESP is 4 higher at
      // the point of the load.
      BuildMI(MBB, FldMI, DL, TII->get(X86::MOV32rm), X86::EAX)
          .addReg(X86::ESP)
          .addImm(FldMI.getOperand(1).getImm())  // scale
          .addReg(FldMI.getOperand(2).getReg())   // index
          .addImm(FldDisp - 4)                    // adjusted displacement
          .addReg(FldMI.getOperand(4).getReg());  // segment

      // PUSH32r EAX
      BuildMI(MBB, FldMI, DL, TII->get(X86::PUSH32r))
          .addReg(X86::EAX);

      // Remove FLD, FSTP, and post-call SUB
      NextI = std::next(MachineBasicBlock::iterator(*PostCallIt));
      FldMI.eraseFromParent();
      FstpMI.eraseFromParent();
      PostCallIt->eraseFromParent();

      Changed = true;
      I = NextI;
      continue;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferIntFloatForwardPass() {
  return new X86PreferIntFloatForwardPass();
}

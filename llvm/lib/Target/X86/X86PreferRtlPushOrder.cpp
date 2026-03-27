//===--- X86PreferRtlPushOrder.cpp - RTL argument push reordering ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp / deusex-decomp: MSVC 6.0 evaluates function arguments
// right-to-left. When a function call has both constant arguments and
// arguments computed by sub-calls, MSVC pushes the constants first:
//
//   push 0x1              ; rightmost arg (constant) pushed first
//   mov ecx, esi
//   call GetFootball      ; compute leftmost arg
//   mov ecx, eax
//   call GetBall
//   push eax              ; push computed leftmost arg
//   mov ecx, esi
//   call LookAtObject     ; target call
//
// Clang evaluates left-to-right, producing:
//
//   call GetFootball      ; compute leftmost arg first
//   call GetBall
//   push eax              ; push computed leftmost arg
//   push 0x1              ; then push rightmost constant arg
//   call LookAtObject     ; target call
//
// This pass scans backward from each CALL to find the argument PUSH
// sequence. If a PUSH-imm appears after a CALL+PUSH-reg sequence
// (meaning the constant was evaluated after the sub-call), it moves
// the PUSH-imm to before the argument-computing sub-call chain.
//
// Gate: function attribute "prefer_rtl_push_order".
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-rtl-push-order"

namespace {
class X86PreferRtlPushOrderPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferRtlPushOrderPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 prefer RTL push order for MSVC 6.0";
  }
};
} // end anonymous namespace

char X86PreferRtlPushOrderPass::ID = 0;

/// Check if MI is a PUSH of an immediate value.
static bool isPushImm(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  return Opc == X86::PUSH32i8 || Opc == X86::PUSH32i;
}

/// Check if MI is a PUSH of a register.
static bool isPushReg(const MachineInstr &MI) {
  return MI.getOpcode() == X86::PUSH32r;
}

/// Check if MI is any kind of PUSH used for argument passing.
static bool isAnyPush(const MachineInstr &MI) {
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

/// Check if MI references ESP in any explicit operand.
static bool referencesEsp(const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.explicit_operands()) {
    if (MO.isReg() && MO.getReg() == X86::ESP)
      return true;
  }
  return false;
}

bool X86PreferRtlPushOrderPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("prefer_rtl_push_order"))
    return false;

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Scan forward looking for the target pattern:
    //   [sub-call chain]  CALL + MOV ECX + CALL ...
    //   PUSH reg          (push result of sub-call chain)
    //   PUSH imm          (constant arg that should come first)
    //   ...more PUSHes...
    //   CALL              (target function call)
    //
    // We want to move PUSH imm(s) to before the sub-call chain.

    for (auto I = MBB.begin(), E = MBB.end(); I != E; ++I) {
      // Find a target CALL instruction.
      if (!I->isCall())
        continue;

      MachineInstr *TargetCall = &*I;

      // Walk backwards from the CALL to collect the argument PUSHes.
      // Skip pseudos (ADJCALLSTACKDOWN, CFI, etc.) and MOV ECX setup.
      auto Scan = MachineBasicBlock::iterator(TargetCall);
      if (Scan == MBB.begin())
        continue;

      // Collect all PUSHes (and interleaved pseudos/MOV ECX) before the CALL.
      // We walk backwards to find the push sequence.
      struct ArgInfo {
        MachineInstr *Push;
        bool IsImm;
      };
      SmallVector<ArgInfo, 8> Args;

      // Also track any MOV ECX instructions between pushes and the call
      // (thiscall setup). We skip those during scanning.
      auto Pos = Scan;
      --Pos;

      // Skip pseudo instructions and MOV ECX right before the CALL
      while (Pos != MBB.begin()) {
        if (Pos->isPseudo() || Pos->isDebugInstr()) {
          --Pos;
          continue;
        }
        // Skip MOV ECX, REG (thiscall setup) right before CALL
        unsigned Opc = Pos->getOpcode();
        if ((Opc == X86::MOV32rr || Opc == X86::MOV32rr_REV) &&
            Pos->getOperand(0).isReg() &&
            Pos->getOperand(0).getReg() == X86::ECX) {
          --Pos;
          continue;
        }
        break;
      }

      // Now collect PUSHes walking backwards. Between PUSHes there may be
      // sub-call sequences (CALL + MOV ECX) that compute arguments, or
      // pseudos/CFI instructions.
      //
      // The pattern we look for (reading backwards from target CALL):
      //   PUSH imm   <- this should be moved earlier
      //   PUSH reg   <- result of a sub-call
      //   CALL       <- sub-call that computes an arg
      //   ...
      //
      // We collect consecutive PUSHes. If we see PUSH imm after PUSH reg
      // where the PUSH reg follows a CALL, that's our target.

      // Collect the immediate PUSHes that are closest to the target CALL.
      SmallVector<MachineInstr *, 4> TrailingPushImms;
      auto CollectPos = Pos;

      // Skip backwards past pseudo instructions
      while (CollectPos != MBB.begin() &&
             (CollectPos->isPseudo() || CollectPos->isDebugInstr()))
        --CollectPos;

      // Collect trailing PUSH imm instructions (these are the ones to move)
      while (CollectPos != MBB.begin() && isPushImm(*CollectPos)) {
        TrailingPushImms.push_back(&*CollectPos);
        --CollectPos;
        // Skip pseudos
        while (CollectPos != MBB.begin() &&
               (CollectPos->isPseudo() || CollectPos->isDebugInstr()))
          --CollectPos;
      }

      if (TrailingPushImms.empty())
        continue;

      // Now CollectPos should point to a PUSH reg (the computed argument).
      if (!isPushReg(*CollectPos))
        continue;

      MachineInstr *PushReg = &*CollectPos;

      // Walk further back to find the sub-call chain that computes this arg.
      // We look for a CALL instruction before the PUSH reg.
      auto SubCallPos = MachineBasicBlock::iterator(PushReg);
      if (SubCallPos == MBB.begin())
        continue;
      --SubCallPos;

      // Skip pseudos and MOV ECX
      while (SubCallPos != MBB.begin()) {
        if (SubCallPos->isPseudo() || SubCallPos->isDebugInstr()) {
          --SubCallPos;
          continue;
        }
        unsigned Opc = SubCallPos->getOpcode();
        if ((Opc == X86::MOV32rr || Opc == X86::MOV32rr_REV) &&
            SubCallPos->getOperand(0).isReg() &&
            SubCallPos->getOperand(0).getReg() == X86::ECX) {
          --SubCallPos;
          continue;
        }
        break;
      }

      // SubCallPos should now be at a CALL instruction (the sub-call).
      if (!SubCallPos->isCall())
        continue;

      MachineInstr *SubCall = &*SubCallPos;

      // Walk backwards from this sub-call to find the start of the sub-call
      // chain. There might be multiple chained calls (e.g., GetFootball then
      // GetBall). We want to find the earliest instruction in the chain.
      auto ChainStart = MachineBasicBlock::iterator(SubCall);

      // Keep walking back through CALL + MOV ECX chains
      while (ChainStart != MBB.begin()) {
        auto Prev = std::prev(ChainStart);
        // Skip pseudos
        while (Prev != MBB.begin() &&
               (Prev->isPseudo() || Prev->isDebugInstr()))
          --Prev;

        // Check for MOV ECX, REG before the CALL
        unsigned Opc = Prev->getOpcode();
        if ((Opc == X86::MOV32rr || Opc == X86::MOV32rr_REV) &&
            Prev->getOperand(0).isReg() &&
            Prev->getOperand(0).getReg() == X86::ECX) {
          ChainStart = Prev;
          if (ChainStart == MBB.begin())
            break;
          --Prev;
          // Skip pseudos
          while (Prev != MBB.begin() &&
                 (Prev->isPseudo() || Prev->isDebugInstr()))
            --Prev;
        }

        // Check for another CALL before the MOV ECX (chained calls)
        if (Prev->isCall()) {
          ChainStart = Prev;
          continue;
        }

        // Check for MOV EAX/ECX setup before a CALL (vtable load, etc.)
        // These are part of the sub-call chain too.
        // Pattern: MOV32rm EAX, [REG] (vtable load)
        if (Prev->getOpcode() == X86::MOV32rm &&
            !referencesEsp(*Prev)) {
          ChainStart = Prev;
          continue;
        }

        break;
      }

      // Safety check: make sure none of the instructions between ChainStart
      // and the PUSH imms reference ESP (other than implicit defs from PUSH/
      // CALL which are expected). If any explicit ESP reference exists, we
      // would need to adjust offsets and that's not safe to do generically.
      bool HasEspRef = false;
      for (auto Check = ChainStart;
           Check != MachineBasicBlock::iterator(TrailingPushImms.back());
           ++Check) {
        if (Check->isPseudo() || Check->isDebugInstr())
          continue;
        if (referencesEsp(*Check)) {
          HasEspRef = true;
          break;
        }
      }
      if (HasEspRef)
        continue;

      // Move all trailing PUSH imm instructions to before ChainStart.
      // TrailingPushImms is in reverse order (last push first), so we
      // insert them in reverse to maintain their original relative order.
      for (int i = TrailingPushImms.size() - 1; i >= 0; --i) {
        MBB.splice(ChainStart, &MBB,
                   MachineBasicBlock::iterator(TrailingPushImms[i]));
      }

      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferRtlPushOrderPass() {
  return new X86PreferRtlPushOrderPass();
}

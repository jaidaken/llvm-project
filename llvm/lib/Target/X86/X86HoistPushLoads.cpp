//===--- X86HoistPushLoads.cpp - Hoist MOVs before PUSH sequence -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// deusex-decomp: After X86PreferRegisterPush unfolds PUSH32rmm to MOV+PUSH,
// the loads are interleaved with pushes:
//
//   push 0                    ; imm push
//   mov eax, [GError]         ; load (interleaved)
//   push eax
//   push 0
//   mov ecx, [esp+0x1c]      ; load (interleaved, post-push ESP offset)
//   push ecx
//   mov edx, [esp+0x1c]      ; load (interleaved)
//   push edx
//
// MSVC 6.0 pre-loads from ORIGINAL ESP offsets before any pushes:
//
//   mov eax, [GError]         ; pre-loaded
//   mov ecx, [esp+0x10]      ; pre-loaded (original ESP offset)
//   mov edx, [esp+0x0c]      ; pre-loaded
//   push 0
//   push eax
//   ...push sequence using registers...
//
// This pass runs AFTER X86PreferRegisterPush and hoists MOV32rm instructions
// that precede their PUSH32r to before the first PUSH in the sequence.
// When hoisting a MOV that loads from ESP-relative address, the displacement
// is adjusted: subtract 4 for each PUSH that the MOV was hoisted over.
//
// Gate: function attribute "prefer_register_push" (same gate as the unfold pass).
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-hoist-push-loads"

namespace {
class X86HoistPushLoadsPass : public MachineFunctionPass {
public:
  static char ID;
  X86HoistPushLoadsPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 hoist MOV loads before PUSH sequence";
  }
};
} // end anonymous namespace

char X86HoistPushLoadsPass::ID = 0;

static bool isAnyPush(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case X86::PUSH32r:
  case X86::PUSH32i8:
  case X86::PUSH32i:
  case X86::PUSH32rmm:
    return true;
  default:
    return false;
  }
}

static bool isMovLoad32(const MachineInstr &MI) {
  return MI.getOpcode() == X86::MOV32rm;
}

/// Count pushes (each shifts ESP by 4) between two iterators.
static unsigned countPushesBetween(MachineBasicBlock::iterator Begin,
                                   MachineBasicBlock::iterator End) {
  unsigned Count = 0;
  for (auto I = Begin; I != End; ++I) {
    if (isAnyPush(*I))
      ++Count;
  }
  return Count;
}

bool X86HoistPushLoadsPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("prefer_register_push"))
    return false;


  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Find sequences: a run of PUSHes and MOV+PUSH pairs ending at a CALL.
    // We scan for the first PUSH or MOV-before-PUSH in the sequence,
    // then collect all MOVs that can be hoisted.

    for (auto I = MBB.begin(), E = MBB.end(); I != E;) {
      // Skip until we find the start of a push sequence.
      if (!isAnyPush(*I) && !(isMovLoad32(*I) && std::next(I) != E &&
                               isAnyPush(*std::next(I)))) {
        ++I;
        continue;
      }

      // Find the first push instruction in this sequence.
      auto SeqStart = I;
      MachineBasicBlock::iterator FirstPush = E;
      for (auto J = SeqStart; J != E; ++J) {
        if (isAnyPush(*J)) {
          FirstPush = J;
          break;
        }
        if (!isMovLoad32(*J))
          break;
      }

      if (FirstPush == E) {
        ++I;
        continue;
      }

      // Collect MOV+PUSH pairs after the first push that can be hoisted.
      // Walk forward through the sequence collecting MOVs to hoist.
      // Cap at 3: MSVC 6.0 pre-loads into at most 3 registers (EAX, ECX, EDX)
      // before the push sequence. The 4th+ values are loaded mid-sequence
      // by reusing a register after it has been pushed.
      SmallVector<MachineInstr *, 4> MovsToHoist;
      auto J = FirstPush;
      while (J != E && MovsToHoist.size() < 3) {
        // Skip CFI pseudo-instructions (stack frame tracking).
        if (J->getOpcode() == TargetOpcode::CFI_INSTRUCTION) {
          ++J;
          continue;
        }
        if (isAnyPush(*J)) {
          ++J;
          continue;
        }
        if (isMovLoad32(*J)) {
          auto Next = std::next(J);
          // Skip CFI between MOV and PUSH.
          while (Next != E && Next->getOpcode() == TargetOpcode::CFI_INSTRUCTION)
            ++Next;
          if (Next != E && Next->getOpcode() == X86::PUSH32r) {
            // This is a MOV+PUSH pair. The MOV can be hoisted.
            Register MovDst = J->getOperand(0).getReg();
            Register PushSrc = Next->getOperand(0).getReg();
            if (MovDst == PushSrc) {
              MovsToHoist.push_back(&*J);
              J = std::next(Next);
              continue;
            }
          }
        }
        // End of push sequence (hit a CALL, another instruction, etc.)
        break;
      }

      if (MovsToHoist.empty()) {
        I = J;
        continue;
      }

      // Hoist each MOV to just before FirstPush.
      // For each MOV, if it loads from ESP-relative, adjust displacement
      // by subtracting 4 * (number of pushes between FirstPush and the MOV's
      // original position).
      for (MachineInstr *Mov : MovsToHoist) {
        // Count pushes between FirstPush and this MOV's current position.
        unsigned PushCount = countPushesBetween(
            FirstPush, MachineBasicBlock::iterator(Mov));

        // Adjust ESP-relative displacement.
        // MOV32rm operands: dst(0), base(1), scale(2), index(3), disp(4), seg(5)
        if (PushCount > 0 &&
            Mov->getOperand(1).isReg() &&
            Mov->getOperand(1).getReg() == X86::ESP &&
            Mov->getOperand(4).isImm()) {
          int64_t OldDisp = Mov->getOperand(4).getImm();
          Mov->getOperand(4).setImm(OldDisp - 4 * PushCount);
        }

        // Splice the MOV to just before FirstPush.
        MBB.splice(FirstPush, &MBB, MachineBasicBlock::iterator(Mov));
        Changed = true;
      }

      I = J;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86HoistPushLoadsPass() {
  return new X86HoistPushLoadsPass();
}

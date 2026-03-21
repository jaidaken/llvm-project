//===--- X86PreferBatchPush.cpp - Batch stack param loads before pushes ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: After PreferMovPush expands PUSH32rmm to MOV32rm + PUSH32r,
// the compiler produces interleaved load-push pairs:
//   mov eax, [esp+8]; push eax; mov eax, [esp+8]; push eax
//
// MSVC 6.0 loads all params first, then pushes:
//   mov eax, [esp+8]; mov edx, [esp+4]; push eax; push edx
//
// This pass detects consecutive MOV32rm+PUSH32r pairs where the MOV
// loads from the same register (EAX), and rewrites to use separate
// registers (EAX, EDX) with all loads first.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-batch-push"

namespace {
class X86PreferBatchPushPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferBatchPushPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 prefer batched stack param push";
  }
};
} // end anonymous namespace

char X86PreferBatchPushPass::ID = 0;

bool X86PreferBatchPushPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("no_tail_call") &&
      !MF.getFunction().hasFnAttribute(Attribute::NoCalleeSaves))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    auto I = MBB.begin();
    while (I != MBB.end()) {
      // Look for: MOV32rm EAX, [mem]; PUSH32r EAX; MOV32rm EAX, [mem]; PUSH32r EAX
      struct LoadPushPair {
        MachineInstr *Load;
        MachineInstr *Push;
      };

      SmallVector<LoadPushPair, 4> Pairs;
      auto SeqStart = I;

      while (I != MBB.end()) {
        // Skip CFI instructions
        while (I != MBB.end() && I->getOpcode() == TargetOpcode::CFI_INSTRUCTION)
          ++I;
        if (I == MBB.end()) break;

        auto Next = std::next(I);
        // Skip CFI after MOV
        while (Next != MBB.end() && Next->getOpcode() == TargetOpcode::CFI_INSTRUCTION)
          ++Next;
        if (Next == MBB.end()) break;

        MachineInstr &Cand = *I;
        MachineInstr &CandPush = *Next;

        // Match: MOV32rm EAX, [any mem]
        if (Cand.getOpcode() != X86::MOV32rm) break;
        if (Cand.getOperand(0).getReg() != X86::EAX) break;

        // Match: PUSH32r EAX
        if (CandPush.getOpcode() != X86::PUSH32r) break;
        if (CandPush.getOperand(0).getReg() != X86::EAX) break;

        Pairs.push_back({&Cand, &CandPush});
        I = std::next(Next);
      }

      if (Pairs.size() < 2) {
        if (Pairs.empty()) ++I;
        continue;
      }

      // Only handle exactly 2 pairs for now
      if (Pairs.size() > 2) {
        continue;
      }

      DebugLoc DL = Pairs[0].Load->getDebugLoc();

      // The first load stays as MOV32rm EAX (same register)
      // Change the second load to use EDX and adjust its ESP offset.
      // The second load was compiled with ESP-4 (after the first push).
      // Since we're moving it before the push, subtract 4 from its disp.
      Pairs[1].Load->getOperand(0).setReg(X86::EDX);
      Pairs[1].Push->getOperand(0).setReg(X86::EDX);
      if (Pairs[1].Load->getOperand(1).isReg() &&
          Pairs[1].Load->getOperand(1).getReg() == X86::ESP &&
          Pairs[1].Load->getOperand(4).isImm()) {
        int64_t OldDisp = Pairs[1].Load->getOperand(4).getImm();
        Pairs[1].Load->getOperand(4).setImm(OldDisp - 4);
      }

      // Move the second load before the first push
      // Current: Load0, Push0, [CFI], Load1, Push1
      // Target:  Load0, Load1, Push0, [CFI], Push1
      MBB.splice(MachineBasicBlock::iterator(Pairs[0].Push),
                 &MBB,
                 MachineBasicBlock::iterator(Pairs[1].Load),
                 std::next(MachineBasicBlock::iterator(Pairs[1].Load)));

      Changed = true;
      // I is already past the sequence
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferBatchPushPass() {
  return new X86PreferBatchPushPass();
}

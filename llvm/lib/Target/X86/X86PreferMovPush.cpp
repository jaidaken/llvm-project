//===--- X86PreferMovPush.cpp - Unfold push [mem] to mov+push -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: MSVC 6.0 never generates push [mem]. It always does
// mov reg,[mem]; push reg. This pass expands PUSH32rmm back to
// MOV32rm + PUSH32r for functions with the no_tail_call attribute.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-mov-push"
#define X86_PREFER_MOV_PUSH_NAME "X86 prefer MOV+PUSH over PUSH [mem]"

namespace {
class X86PreferMovPushPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferMovPushPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return X86_PREFER_MOV_PUSH_NAME; }
};
} // end anonymous namespace

char X86PreferMovPushPass::ID = 0;

bool X86PreferMovPushPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("no_tail_call"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E; ) {
      MachineInstr &MI = *I++;

      // Match PUSH32rmm (push from memory)
      if (MI.getOpcode() != X86::PUSH32rmm)
        continue;

      // Get the memory operand components
      // PUSH32rmm format: (ops i32mem:$src)
      // Memory operand: base, scale, index, disp, segment
      MachineOperand &Base = MI.getOperand(0);
      MachineOperand &Scale = MI.getOperand(1);
      MachineOperand &Index = MI.getOperand(2);
      MachineOperand &Disp = MI.getOperand(3);
      MachineOperand &Seg = MI.getOperand(4);

      DebugLoc DL = MI.getDebugLoc();

      // Build: MOV32rm EAX, [mem]
      MachineInstr *MovMI = BuildMI(MBB, MI, DL, TII->get(X86::MOV32rm), X86::EAX)
          .add(Base).add(Scale).add(Index).add(Disp).add(Seg);

      // Build: PUSH32r EAX
      BuildMI(MBB, MI, DL, TII->get(X86::PUSH32r))
          .addReg(X86::EAX, RegState::Kill);

      // Remove original PUSH32rmm
      MI.eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferMovPushPass() {
  return new X86PreferMovPushPass();
}

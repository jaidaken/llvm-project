//===--- X86PreferRegisterPush.cpp - Pre-load args into regs before push ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// deusex-decomp: MSVC 6.0 never generates push [mem]. It loads into a register
// first, then pushes the register. This pass unfolds PUSH32rmm to
// MOV32rm + PUSH32r with rotating register assignment (EAX, ECX, EDX).
//
// Gate: function attribute "prefer_register_push".
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-register-push"
#define X86_PREFER_REGISTER_PUSH_NAME \
  "X86 prefer register pre-load before push sequence"

namespace {
class X86PreferRegisterPushPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferRegisterPushPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return X86_PREFER_REGISTER_PUSH_NAME;
  }
};
} // end anonymous namespace

char X86PreferRegisterPushPass::ID = 0;

bool X86PreferRegisterPushPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("prefer_register_push"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Build liveness snapshots for PUSH32rmm instructions.
    // Use the same compact struct pattern as X86ZeroViaXor.
    struct LiveInfo {
      bool EaxLive = false;
      bool EcxLive = false;
      bool EdxLive = false;
    };
    DenseMap<MachineInstr *, LiveInfo> LivenessMap;
    {
      LivePhysRegs LiveRegs(*TRI);
      LiveRegs.addLiveOuts(MBB);
      for (auto I = MBB.rbegin(), E = MBB.rend(); I != E; ++I) {
        MachineInstr &MI = *I;
        LiveRegs.stepBackward(MI);
        if (MI.getOpcode() == X86::PUSH32rmm) {
          LiveInfo LI;
          LI.EaxLive = LiveRegs.contains(X86::EAX);
          LI.EcxLive = LiveRegs.contains(X86::ECX);
          LI.EdxLive = LiveRegs.contains(X86::EDX);
          LivenessMap[&MI] = LI;
        }
      }
    }

    // Rotating register assignment for consecutive PUSH32rmm instructions.
    static const Register RegOrder[] = {X86::EAX, X86::ECX, X86::EDX};
    unsigned RegIdx = 0;

    for (auto I = MBB.begin(), E = MBB.end(); I != E;) {
      MachineInstr &MI = *I++;

      if (MI.getOpcode() != X86::PUSH32rmm)
        continue;

      // Memory operand: base(0), scale(1), index(2), disp(3), segment(4)
      Register BaseReg = MI.getOperand(0).getReg();
      Register IndexReg = MI.getOperand(2).isReg() ? MI.getOperand(2).getReg()
                                                    : Register();

      auto LiveIt = LivenessMap.find(&MI);
      if (LiveIt == LivenessMap.end())
        continue;
      const LiveInfo &LI = LiveIt->second;

      // Pick a scratch register from the rotation.
      Register ScratchReg;
      for (unsigned Attempt = 0; Attempt < 3; ++Attempt) {
        Register Cand = RegOrder[(RegIdx + Attempt) % 3];
        // Check liveness.
        bool IsLive = false;
        if (Cand == X86::EAX) IsLive = LI.EaxLive;
        else if (Cand == X86::ECX) IsLive = LI.EcxLive;
        else if (Cand == X86::EDX) IsLive = LI.EdxLive;
        if (IsLive)
          continue;
        // Check overlap with memory operand registers.
        if (BaseReg && TRI->regsOverlap(Cand, BaseReg))
          continue;
        if (IndexReg && TRI->regsOverlap(Cand, IndexReg))
          continue;
        ScratchReg = Cand;
        RegIdx = (RegIdx + Attempt + 1) % 3;
        break;
      }

      if (!ScratchReg)
        continue;

      DebugLoc DL = MI.getDebugLoc();

      // Build: MOV32rm ScratchReg, [mem]
      BuildMI(MBB, MI, DL, TII->get(X86::MOV32rm), ScratchReg)
          .add(MI.getOperand(0))  // base
          .add(MI.getOperand(1))  // scale
          .add(MI.getOperand(2))  // index
          .add(MI.getOperand(3))  // disp
          .add(MI.getOperand(4)); // segment

      // Build: PUSH32r ScratchReg
      BuildMI(MBB, MI, DL, TII->get(X86::PUSH32r))
          .addReg(ScratchReg, RegState::Kill);

      MI.eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferRegisterPushPass() {
  return new X86PreferRegisterPushPass();
}

//===---- X86PreferPopCleanup.cpp - Use pop ecx for stack cleanup ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: This pass replaces "add esp, 4" (3 bytes: 83 C4 04) with
// "pop ecx" (1 byte: 59) for cdecl call cleanup in functions with the
// PreferPopCleanup attribute.
//
// MSVC 6.0 uses pop ecx to clean up a single 4-byte argument from the
// stack after cdecl calls, saving 2 bytes per call site.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-pop-cleanup"
#define X86_PREFER_POP_CLEANUP_NAME "X86 prefer pop cleanup pass"

namespace {
class X86PreferPopCleanupPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferPopCleanupPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return X86_PREFER_POP_CLEANUP_NAME;
  }
};
} // end anonymous namespace

char X86PreferPopCleanupPass::ID = 0;

bool X86PreferPopCleanupPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::PreferPopCleanup))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Backward scan to track ECX liveness.
    SmallVector<MachineInstr *, 4> ToReplace;
    {
      LivePhysRegs LiveRegs(*TRI);
      LiveRegs.addLiveOuts(MBB);

      for (auto I = MBB.rbegin(), E = MBB.rend(); I != E; ++I) {
        MachineInstr &MI = *I;

        // Match: ADD32ri8 ESP, 4
        if (MI.getOpcode() == X86::ADD32ri8 &&
            MI.getOperand(0).getReg() == X86::ESP &&
            MI.getOperand(2).getImm() == 4) {
          // ECX must be dead — pop will clobber it.
          if (!LiveRegs.contains(X86::ECX))
            ToReplace.push_back(&MI);
        }

        LiveRegs.stepBackward(MI);
      }
    }

    for (MachineInstr *MI : ToReplace) {
      DebugLoc DL = MI->getDebugLoc();

      // Replace ADD32ri8 ESP, 4 with POP32r ECX.
      // pop ecx: opcode 0x59 (1 byte) vs add esp, 4: 0x83 0xC4 0x04 (3 bytes)
      BuildMI(MBB, *MI, DL, TII->get(X86::POP32r), X86::ECX);

      MI->eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferPopCleanupPass() {
  return new X86PreferPopCleanupPass();
}

//===------- X86OrMinusOne.cpp - Convert MOV32ri -1 to OR32ri8 -1 ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: This pass converts MOV32ri reg, 0xFFFFFFFF to OR32ri8 reg, -1
// when the function has the PreferOrMinusOne attribute. This matches MSVC 6.0's
// code generation pattern of "or eax, -1" (opcode 0x83 0xC8 0xFF, 3 bytes)
// instead of "mov eax, -1" (0xB8 0xFF 0xFF 0xFF 0xFF, 5 bytes).
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-or-minus-one"
#define X86_OR_MINUS_ONE_NAME "X86 OR minus one pass"

namespace {
class X86OrMinusOnePass : public MachineFunctionPass {
public:
  static char ID;
  X86OrMinusOnePass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return X86_OR_MINUS_ONE_NAME; }
};
} // end anonymous namespace

char X86OrMinusOnePass::ID = 0;

bool X86OrMinusOnePass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::PreferOrMinusOne))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Two-pass approach: first collect MOV32ri -1 instructions, then replace.
    // Use backward iteration to track EFLAGS liveness.
    SmallVector<MachineInstr *, 8> ToReplace;
    {
      LivePhysRegs LiveRegs(*TRI);
      LiveRegs.addLiveOuts(MBB);

      for (auto I = MBB.rbegin(), E = MBB.rend(); I != E; ++I) {
        MachineInstr &MI = *I;

        if (MI.getOpcode() == X86::MOV32ri) {
          // Check if the immediate is 0xFFFFFFFF (which is -1 as int32).
          int64_t Imm = MI.getOperand(1).getImm();
          if (Imm == -1 || Imm == 0xFFFFFFFF) {
            // OR clobbers EFLAGS, so only replace if EFLAGS is dead.
            if (!LiveRegs.contains(X86::EFLAGS)) {
              ToReplace.push_back(&MI);
            }
          }
        }

        LiveRegs.stepBackward(MI);
      }
    }

    // Pass 2: Replace collected MOV32ri instructions with OR32ri8.
    for (MachineInstr *MI : ToReplace) {
      Register DstReg = MI->getOperand(0).getReg();
      DebugLoc DL = MI->getDebugLoc();

      // OR32ri8 reg, -1: the -1 is sign-extended from 8 bits to 32 bits,
      // producing 0xFFFFFFFF. The OR with -1 sets all bits regardless of
      // the previous value of the register.
      BuildMI(MBB, *MI, DL, TII->get(X86::OR32ri8), DstReg)
          .addReg(DstReg, RegState::Undef)
          .addImm(-1);

      MI->eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86OrMinusOnePass() {
  return new X86OrMinusOnePass();
}

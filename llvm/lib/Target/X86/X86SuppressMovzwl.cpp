//===--- X86SuppressMovzwl.cpp - Convert MOVZX32rm16 to MOV16rm ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: Clang zero-extends 16-bit memory loads using MOVZX32rm16
// (movzx edx, word [mem], opcode 0F B7). MSVC 6.0 uses partial register
// writes: MOV16rm (mov dx, word [mem], 66 8B prefix). Same code size but
// different opcodes.
//
// This pass, gated on the "suppress_movzwl" string attribute, converts all
// MOVZX32rm16 instructions into MOV16rm (16-bit partial register load).
//
// Example:
//   Before: movzx edx, word ptr [ecx+0x10]   ; 0F B7 51 10
//   After:  mov dx, word ptr [ecx+0x10]       ; 66 8B 51 10
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "x86-suppress-movzwl"

namespace {
class X86SuppressMovzwlPass : public MachineFunctionPass {
public:
  static char ID;
  X86SuppressMovzwlPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 suppress MOVZX word-extend to partial MOV16";
  }
};
} // end anonymous namespace

char X86SuppressMovzwlPass::ID = 0;

bool X86SuppressMovzwlPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("suppress_movzwl"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E;) {
      MachineInstr &MI = *I;

      if (MI.getOpcode() != X86::MOVZX32rm16) {
        ++I;
        continue;
      }

      Register DstReg32 = MI.getOperand(0).getReg();
      Register DstReg16 = TRI->getSubReg(DstReg32, X86::sub_16bit);
      if (!DstReg16) {
        ++I;
        continue;
      }

      LLVM_DEBUG(dbgs() << "SuppressMovzwl: converting MOVZX32rm16 to MOV16rm"
                        << " in " << MF.getName() << "\n");

      // Build MOV16rm with the 16-bit subreg destination.
      MachineInstrBuilder NewMI =
          BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(X86::MOV16rm), DstReg16);
      // Copy memory operands (base, scale, index, disp, segment).
      for (unsigned i = 1; i < MI.getNumOperands(); ++i)
        NewMI.add(MI.getOperand(i));
      NewMI.setMemRefs(MI.memoperands());

      auto NextI = std::next(I);
      MI.eraseFromParent();
      I = NextI;
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86SuppressMovzwlPass() {
  return new X86SuppressMovzwlPass();
}

//===------- X86PreferXOR8.cpp - Convert XOR32rr self-xors to XOR8rr ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: This pass converts XOR32rr/XOR32rr_REV self-xor zeroing idioms
// to their 8-bit equivalents (XOR8rr/XOR8rr_REV) when the function has the
// PreferXOR8 attribute. This matches MSVC 6.0's code generation pattern of
// "xor al, al" (opcode 0x32 0xC0) instead of "xor eax, eax" (0x33 0xC0).
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-xor8"
#define X86_PREFER_XOR8_NAME "X86 prefer 8-bit XOR zeroing pass"

namespace {
class X86PreferXOR8Pass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferXOR8Pass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return X86_PREFER_XOR8_NAME; }
};
} // end anonymous namespace

char X86PreferXOR8Pass::ID = 0;

bool X86PreferXOR8Pass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::PreferXOR8))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      unsigned Opcode = MI.getOpcode();

      // Only handle XOR32rr and XOR32rr_REV.
      if (Opcode != X86::XOR32rr && Opcode != X86::XOR32rr_REV)
        continue;

      Register DstReg = MI.getOperand(0).getReg();
      Register Src1Reg = MI.getOperand(1).getReg();
      Register Src2Reg = MI.getOperand(2).getReg();

      // Must be a self-xor (zeroing idiom): xor reg, reg
      if (DstReg != Src1Reg || DstReg != Src2Reg)
        continue;

      // Get the 8-bit sub-register (AL for EAX, BL for EBX, etc.).
      // Only EAX/EBX/ECX/EDX have 8-bit sub-registers in 32-bit mode.
      Register SubReg = TRI->getSubReg(DstReg, X86::sub_8bit);
      if (!SubReg)
        continue;

      // Choose the 8-bit opcode matching the encoding direction.
      unsigned NewOpcode = (Opcode == X86::XOR32rr_REV)
                               ? X86::XOR8rr_REV
                               : X86::XOR8rr;

      DebugLoc DL = MI.getDebugLoc();
      BuildMI(MBB, MI, DL, TII->get(NewOpcode), SubReg)
          .addReg(SubReg, RegState::Undef)
          .addReg(SubReg, RegState::Undef);

      MI.eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferXOR8Pass() {
  return new X86PreferXOR8Pass();
}

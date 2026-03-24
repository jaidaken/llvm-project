//===-- X86Msvc6FastcallRegFix.cpp - Swap EAX<->ECX for fastcall ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: Post-regalloc pass that swaps EAX<->ECX throughout a function
// when LLVM's register allocator assigned them opposite to MSVC 6.0's pattern.
//
// MSVC 6.0 fastcall functions copy the this-pointer from ECX to EAX at entry,
// then use EAX as the base register for member stores. LLVM's greedy allocator
// sometimes assigns these the other way around. This pass detects the mismatch
// by counting store base registers in the entry block and swaps EAX<->ECX
// (plus sub-registers) in all instructions when the registers are reversed.
//
// Gated behind the "msvc6_fastcall_regfix" string function attribute.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
using namespace llvm;

#define DEBUG_TYPE "x86-msvc6-fastcall-regfix"
#define X86_MSVC6_FASTCALL_REGFIX_NAME \
  "X86 MSVC 6.0 fastcall EAX<->ECX register fix"

namespace {
class X86Msvc6FastcallRegFixPass : public MachineFunctionPass {
public:
  static char ID;
  X86Msvc6FastcallRegFixPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return X86_MSVC6_FASTCALL_REGFIX_NAME;
  }
};
} // end anonymous namespace

char X86Msvc6FastcallRegFixPass::ID = 0;

/// Map a register to its swapped counterpart (EAX<->ECX and sub-registers).
static unsigned swapReg(unsigned Reg) {
  switch (Reg) {
  case X86::EAX: return X86::ECX;
  case X86::ECX: return X86::EAX;
  case X86::AX:  return X86::CX;
  case X86::CX:  return X86::AX;
  case X86::AL:  return X86::CL;
  case X86::CL:  return X86::AL;
  case X86::AH:  return X86::CH;
  case X86::CH:  return X86::AH;
  default: return Reg;
  }
}

bool X86Msvc6FastcallRegFixPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("msvc6_fastcall_regfix"))
    return false;

  MachineBasicBlock &EntryMBB = MF.front();

  // Find the entry MOV: first non-pseudo, non-debug instruction that is
  // MOV32rr or MOV32rr_REV with dst=EAX, src=ECX (the this-ptr copy).
  MachineInstr *EntryMov = nullptr;
  for (MachineInstr &MI : EntryMBB) {
    if (MI.isPseudo() || MI.isDebugInstr())
      continue;
    unsigned Opc = MI.getOpcode();
    if ((Opc == X86::MOV32rr || Opc == X86::MOV32rr_REV) &&
        MI.getOperand(0).getReg() == X86::EAX &&
        MI.getOperand(1).getReg() == X86::ECX) {
      EntryMov = &MI;
    }
    break; // Only check the first real instruction.
  }

  if (!EntryMov)
    return false;

  // Count store instructions that use EAX vs ECX as base register
  // in the entry basic block.
  unsigned StoresViaEAX = 0;
  unsigned StoresViaECX = 0;

  for (MachineInstr &MI : EntryMBB) {
    unsigned Opc = MI.getOpcode();
    if (Opc != X86::MOV32mr && Opc != X86::MOV32mi &&
        Opc != X86::MOV16mr && Opc != X86::MOV8mr)
      continue;
    unsigned BaseReg = MI.getOperand(0).getReg();
    if (BaseReg == X86::EAX)
      StoresViaEAX++;
    else if (BaseReg == X86::ECX)
      StoresViaECX++;
  }

  // Only swap when ALL stores use the wrong register (ECX instead of EAX).
  // If both registers are used as bases, the situation is ambiguous - skip.
  if (StoresViaECX == 0)
    return false;
  if (StoresViaEAX > 0)
    return false;

  // The registers are swapped. Walk ALL instructions in the function,
  // swap every occurrence of EAX<->ECX (and sub-registers) in all operands.
  // Skip the entry MOV itself.
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (&MI == EntryMov)
        continue;
      for (MachineOperand &MO : MI.operands()) {
        if (!MO.isReg())
          continue;
        unsigned NewReg = swapReg(MO.getReg());
        if (NewReg != MO.getReg()) {
          MO.setReg(NewReg);
          Changed = true;
        }
      }
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86Msvc6FastcallRegFixPass() {
  return new X86Msvc6FastcallRegFixPass();
}

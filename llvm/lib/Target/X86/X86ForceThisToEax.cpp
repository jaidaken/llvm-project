//===--- X86ForceThisToEax.cpp - Swap EAX<->ECX for this-ptr functions -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// deusex-decomp: Post-regalloc pass that swaps EAX and ECX throughout a
// function and inserts MOV32rr EAX, ECX at entry.
//
// MSVC 6.0 copies `this` (ECX) to EAX at the start of small member functions
// that don't make sub-calls, then uses ECX for loading parameters:
//   mov.s eax, ecx      ; copy this to EAX
//   mov ecx, [esp+4]    ; load parameter into ECX (now free)
//   mov edx, [ecx+4]    ; read through parameter
//   mov [eax+4], edx    ; write through this (via EAX)
//
// Clang keeps `this` in ECX and uses EAX for the parameter, producing:
//   mov eax, [esp+4]    ; load parameter into EAX
//   mov edx, [eax+4]    ; read through parameter
//   mov [ecx+4], edx    ; write through this (stays in ECX)
//
// This pass swaps all EAX<->ECX references (and sub-registers) throughout
// the function, then inserts MOV EAX, ECX at entry. The swap turns Clang's
// allocation into MSVC's.
//
// Gate: function attribute "force_this_eax".
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
using namespace llvm;

#define DEBUG_TYPE "x86-force-this-to-eax"
#define X86_FORCE_THIS_TO_EAX_NAME "X86 force this pointer to EAX pass"

namespace {
class X86ForceThisToEaxPass : public MachineFunctionPass {
public:
  static char ID;
  X86ForceThisToEaxPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return X86_FORCE_THIS_TO_EAX_NAME; }
};
} // end anonymous namespace

char X86ForceThisToEaxPass::ID = 0;

/// Swap EAX<->ECX and sub-registers.
static unsigned swapRegAC(unsigned Reg) {
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

/// Check if an instruction is a prologue PUSH (callee-saved register save).
static bool isProloguePush(const MachineInstr &MI) {
  return MI.getOpcode() == X86::PUSH32r;
}

/// Check if a MOV instruction is a nop (source == dest).
static bool isNopMov(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  if (Opc != X86::MOV32rr && Opc != X86::MOV32rr_REV &&
      Opc != X86::MOV16rr && Opc != X86::MOV16rr_REV &&
      Opc != X86::MOV8rr && Opc != X86::MOV8rr_REV)
    return false;
  return MI.getNumOperands() >= 2 &&
         MI.getOperand(0).isReg() && MI.getOperand(1).isReg() &&
         MI.getOperand(0).getReg() == MI.getOperand(1).getReg();
}

bool X86ForceThisToEaxPass::runOnMachineFunction(MachineFunction &MF) {
  const Function &Fn = MF.getFunction();
  if (!Fn.hasFnAttribute("force_this_eax"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();

  MachineBasicBlock &EntryMBB = MF.front();

  // Find the insertion point: after all prologue PUSHes.
  MachineBasicBlock::iterator InsertPt = EntryMBB.begin();
  while (InsertPt != EntryMBB.end() && isProloguePush(*InsertPt))
    ++InsertPt;

  // Insert: MOV32rr EAX, ECX at function entry.
  DebugLoc DL;
  MachineInstr *InsertedMov =
      BuildMI(EntryMBB, InsertPt, DL, TII->get(X86::MOV32rr), X86::EAX)
          .addReg(X86::ECX);

  // Swap ALL EAX<->ECX references (both uses and defs) in all instructions
  // EXCEPT the MOV we just inserted. This turns Clang's
  //   "this in ECX, param in EAX"
  // into MSVC's
  //   "this in EAX, param in ECX".
  bool Changed = true;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (&MI == InsertedMov)
        continue;
      for (MachineOperand &MO : MI.operands()) {
        if (!MO.isReg())
          continue;
        unsigned NewReg = swapRegAC(MO.getReg());
        if (NewReg != MO.getReg()) {
          MO.setReg(NewReg);
          Changed = true;
        }
      }
    }
  }

  // Remove nop MOVs created by swapping (e.g., `MOV EAX, ECX` where both
  // were swapped becomes `MOV ECX, EAX`, which is a valid instruction but
  // check for self-moves like `MOV EAX, EAX`).
  SmallVector<MachineInstr *, 4> NopMovs;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (&MI == InsertedMov)
        continue;
      if (isNopMov(MI))
        NopMovs.push_back(&MI);
    }
  }
  for (MachineInstr *MI : NopMovs) {
    MI->eraseFromParent();
    Changed = true;
  }

  return Changed;
}

FunctionPass *llvm::createX86ForceThisToEaxPass() {
  return new X86ForceThisToEaxPass();
}

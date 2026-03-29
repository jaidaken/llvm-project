//===--- X86PreferFirstMemLoadOrder.cpp - Swap mem load order for CMP -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: In `if (a != b)`, the compiler loads `b` first then `a`.
// MSVC 6.0 loads `a` first then `b`.
//
// Pattern:
//   mov REG1, [mem_b]    ; load b
//   mov REG2, [mem_a]    ; load a
//   cmp REG2, REG1       ; cmp a, b
//
// Desired:
//   mov REG2, [mem_a]    ; load a (now first)
//   mov REG1, [mem_b]    ; load b (now second)
//   cmp REG2, REG1       ; cmp a, b (unchanged, since we swap destinations)
//
// The pass swaps the two MOV32rm instructions and swaps their destination
// registers, so the CMP operands remain correct.
//
// Gated on the "prefer_first_mem_load_order" function attribute.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-first-mem-load-order"

namespace {

class X86PreferFirstMemLoadOrderPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferFirstMemLoadOrderPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 prefer first memory load order for MSVC 6.0";
  }
};
} // end anonymous namespace

char X86PreferFirstMemLoadOrderPass::ID = 0;

bool X86PreferFirstMemLoadOrderPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("prefer_first_mem_load_order"))
    return false;

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E; ) {
      MachineInstr &MI1 = *I;

      // Look for first MOV32rm.
      if (MI1.getOpcode() != X86::MOV32rm) {
        ++I;
        continue;
      }

      // Find the next non-pseudo instruction.
      auto Next1 = std::next(I);
      while (Next1 != E && (Next1->isPseudo() || Next1->isDebugInstr()))
        ++Next1;

      if (Next1 == E) {
        ++I;
        continue;
      }

      MachineInstr &MI2 = *Next1;

      // Second instruction must also be MOV32rm.
      if (MI2.getOpcode() != X86::MOV32rm) {
        ++I;
        continue;
      }

      // Find the next non-pseudo after the second MOV.
      auto Next2 = std::next(Next1);
      while (Next2 != E && (Next2->isPseudo() || Next2->isDebugInstr()))
        ++Next2;

      if (Next2 == E) {
        ++I;
        continue;
      }

      MachineInstr &CmpMI = *Next2;

      // Third instruction must be CMP32rr.
      if (CmpMI.getOpcode() != X86::CMP32rr) {
        ++I;
        continue;
      }

      Register Reg1 = MI1.getOperand(0).getReg();
      Register Reg2 = MI2.getOperand(0).getReg();
      Register CmpL = CmpMI.getOperand(0).getReg();
      Register CmpR = CmpMI.getOperand(1).getReg();

      // The CMP should use both load destinations.
      // Pattern: mov Reg1, [b]; mov Reg2, [a]; cmp Reg2, Reg1
      // We want: mov Reg2, [a]; mov Reg1, [b]; cmp Reg2, Reg1
      // (swap the two MOVs' positions and swap their dest registers)
      if (!((CmpL == Reg2 && CmpR == Reg1) ||
            (CmpL == Reg1 && CmpR == Reg2))) {
        ++I;
        continue;
      }

      // Swap the destination registers of the two MOV instructions.
      MI1.getOperand(0).setReg(Reg2);
      MI2.getOperand(0).setReg(Reg1);

      // Swap the positions of the two MOV instructions.
      // Move MI2 before MI1.
      MBB.splice(MachineBasicBlock::iterator(MI1), &MBB,
                  MachineBasicBlock::iterator(MI2));

      Changed = true;
      // Advance past this group.
      I = std::next(Next2);
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferFirstMemLoadOrderPass() {
  return new X86PreferFirstMemLoadOrderPass();
}

//===--- X86PreferTest16Bit.cpp - Convert TEST32rr to TEST16rr -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: After X86ExpandMovzx converts MOVZX32rm16 to XOR32rr + MOV16rm,
// Clang emits TEST32rr EAX, EAX (opcode 85 C0, 2 bytes) to test the result.
// MSVC 6.0 emits TEST16rr AX, AX (opcode 66 85 C0, 3 bytes) instead.
//
// This pass, gated on the "prefer_test_16bit" function attribute, converts
// TEST32rr reg, reg to TEST16rr subreg, subreg when the value in reg was
// loaded via a 16-bit partial register write (MOV16rm after XOR32rr).
//
// The Jcc following the TEST does not need changing because both TEST32rr
// and TEST16rr set ZF/SF identically for 16-bit values zero-extended into
// a 32-bit register.
//
// Example:
//   Before: xor eax, eax; mov ax, [mem]; test eax, eax; je .LBB0_1
//   After:  xor eax, eax; mov ax, [mem]; test ax, ax;   je .LBB0_1
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-test-16bit"

namespace {
class X86PreferTest16BitPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferTest16BitPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 prefer TEST16rr over TEST32rr for 16-bit loads";
  }
};
} // end anonymous namespace

char X86PreferTest16BitPass::ID = 0;

bool X86PreferTest16BitPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("prefer_test_16bit"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      // Look for TEST32rr reg, reg (same register both operands).
      if (MI.getOpcode() != X86::TEST32rr)
        continue;
      Register TestReg = MI.getOperand(0).getReg();
      if (TestReg != MI.getOperand(1).getReg())
        continue;

      // Get the 16-bit sub-register for this 32-bit register.
      Register Sub16 = TRI->getSubReg(TestReg, X86::sub_16bit);
      if (!Sub16)
        continue;

      // Walk backwards from the TEST to find what defines TestReg.
      // We expect: XOR32rr reg, reg; MOV16rm subreg, [mem]; ...; TEST32rr
      // with no intervening defs of reg between MOV16rm and TEST.
      bool FoundPattern = false;
      MachineBasicBlock::iterator SearchIt(MI);

      while (SearchIt != MBB.begin()) {
        --SearchIt;
        MachineInstr &Prev = *SearchIt;

        if (Prev.isDebugInstr())
          continue;

        // Check if this instruction defines any register overlapping TestReg.
        bool DefsTestReg = false;
        for (const MachineOperand &MO : Prev.operands()) {
          if (MO.isReg() && MO.isDef() && MO.getReg() != 0 &&
              TRI->regsOverlap(MO.getReg(), TestReg)) {
            DefsTestReg = true;
            break;
          }
        }

        if (!DefsTestReg)
          continue;

        // Found the defining instruction. It must be MOV16rm into subreg.
        if (Prev.getOpcode() != X86::MOV16rm ||
            Prev.getOperand(0).getReg() != Sub16)
          break;

        // Now check that the instruction before this MOV16rm is XOR32rr
        // zeroing the same register.
        auto XorIt = SearchIt;
        while (XorIt != MBB.begin()) {
          --XorIt;
          if (XorIt->isDebugInstr())
            continue;

          unsigned XorOpc = XorIt->getOpcode();
          if ((XorOpc == X86::XOR32rr || XorOpc == X86::XOR32rr_REV) &&
              XorIt->getOperand(0).getReg() == TestReg &&
              XorIt->getOperand(1).getReg() == TestReg &&
              XorIt->getOperand(2).getReg() == TestReg) {
            FoundPattern = true;
          }
          break; // Only check the immediately preceding non-debug instr.
        }
        break; // Stop at the first defining instruction regardless.
      }

      if (!FoundPattern)
        continue;

      LLVM_DEBUG(dbgs() << "PreferTest16Bit: converting TEST32rr "
                        << printReg(TestReg, TRI) << " to TEST16rr "
                        << printReg(Sub16, TRI) << " in "
                        << MF.getName() << "\n");

      BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(X86::TEST16rr))
          .addReg(Sub16)
          .addReg(Sub16);

      MI.eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferTest16BitPass() {
  return new X86PreferTest16BitPass();
}

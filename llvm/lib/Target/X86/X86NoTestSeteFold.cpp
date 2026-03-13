//===---- X86NoTestSeteFold.cpp - Prevent test+sete folding ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: This pass converts TEST8ri + SETEr/SETNEr patterns back into
// NOT8r + SHR32ri + AND32ri8 for functions with the NoTestSeteFold attribute.
// MSVC 6.0 generates "not al; shr eax, N; and eax, 1" for (~byte >> N) & 1
// instead of LLVM's "test al, (1 << N); sete al".
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/MathExtras.h"
using namespace llvm;

#define DEBUG_TYPE "x86-no-test-sete-fold"
#define X86_NO_TEST_SETE_FOLD_NAME "X86 no test+sete fold pass"

namespace {
class X86NoTestSeteFoldPass : public MachineFunctionPass {
public:
  static char ID;
  X86NoTestSeteFoldPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return X86_NO_TEST_SETE_FOLD_NAME;
  }
};
} // end anonymous namespace

char X86NoTestSeteFoldPass::ID = 0;

/// Check if instruction is a SETcc that tests for equality (ZF=1).
static bool isSETEInstr(const MachineInstr &MI) {
  return (MI.getOpcode() == X86::SETCCr || MI.getOpcode() == X86::SETCCm) &&
         X86::getCondFromSETCC(MI) == X86::COND_E;
}

/// Check if instruction is a SETcc that tests for inequality (ZF=0).
static bool isSETNEInstr(const MachineInstr &MI) {
  return (MI.getOpcode() == X86::SETCCr || MI.getOpcode() == X86::SETCCm) &&
         X86::getCondFromSETCC(MI) == X86::COND_NE;
}

/// Get the 32-bit super-register for an 8-bit register.
static Register get32BitSuperReg(Register Reg8) {
  switch (Reg8) {
  case X86::AL: return X86::EAX;
  case X86::CL: return X86::ECX;
  case X86::DL: return X86::EDX;
  case X86::BL: return X86::EBX;
  default: return 0;
  }
}

bool X86NoTestSeteFoldPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::NoTestSeteFold))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Scan for TEST8ri + SETEr/SETNEr patterns.
    for (auto I = MBB.begin(), E = MBB.end(); I != E; /*incremented below*/) {
      MachineInstr &MI = *I;

      // Look for TEST8ri with a power-of-2 immediate.
      if (MI.getOpcode() != X86::TEST8ri) {
        ++I;
        continue;
      }

      Register TestReg = MI.getOperand(0).getReg();
      int64_t TestImm = MI.getOperand(1).getImm();

      // Immediate must be a power of 2 (single bit test).
      if (!isPowerOf2_64(TestImm & 0xFF)) {
        ++I;
        continue;
      }

      unsigned BitPos = Log2_64(TestImm & 0xFF);

      // Find the SETcc that uses this TEST's EFLAGS.
      auto NextI = std::next(I);
      // Skip over any intervening instructions that don't read/write EFLAGS.
      MachineInstr *SetccMI = nullptr;
      while (NextI != E) {
        if (isSETEInstr(*NextI) ||
            isSETNEInstr(*NextI)) {
          SetccMI = &*NextI;
          break;
        }
        // If this instruction modifies EFLAGS, the pattern is broken.
        if (NextI->modifiesRegister(X86::EFLAGS, /*TRI=*/nullptr))
          break;
        ++NextI;
      }

      // Only handle SETCCr with COND_E (register form, not memory) — computes (~x >> N) & 1.
      if (!SetccMI || SetccMI->getOpcode() != X86::SETCCr ||
          X86::getCondFromSETCC(*SetccMI) != X86::COND_E) {
        ++I;
        continue;
      }

      Register SeteReg = SetccMI->getOperand(0).getReg();
      Register TestSuperReg = get32BitSuperReg(TestReg);
      if (!TestSuperReg) {
        ++I;
        continue;
      }

      DebugLoc DL = MI.getDebugLoc();

      // Replace TEST + SETE with NOT + SHR + AND.
      // not al          — invert the byte
      BuildMI(MBB, MI, DL, TII->get(X86::NOT8r), TestReg)
          .addReg(TestReg);

      // shr eax, N      — shift the inverted bit into position 0
      if (BitPos > 0) {
        BuildMI(MBB, MI, DL, TII->get(X86::SHR32ri), TestSuperReg)
            .addReg(TestSuperReg)
            .addImm(BitPos);
      }

      // and eax, 1      — mask to single bit
      BuildMI(MBB, MI, DL, TII->get(X86::AND32ri8), TestSuperReg)
          .addReg(TestSuperReg)
          .addImm(1);

      // If the SETE destination is different from the TEST register,
      // we need to move the result.
      if (SeteReg != TestReg) {
        Register SeteSuperReg = get32BitSuperReg(SeteReg);
        if (SeteSuperReg && SeteSuperReg != TestSuperReg) {
          BuildMI(MBB, MI, DL, TII->get(X86::MOV32rr), SeteSuperReg)
              .addReg(TestSuperReg);
        }
      }

      // Erase the original TEST and SETE.
      // New instructions were inserted before MI (TEST8ri) via BuildMI.
      // Advance I past SETE before erasing, since both TEST and SETE
      // will be invalidated.
      auto EraseI = I;
      I = std::next(NextI);
      SetccMI->eraseFromParent();
      EraseI->eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86NoTestSeteFoldPass() {
  return new X86NoTestSeteFoldPass();
}

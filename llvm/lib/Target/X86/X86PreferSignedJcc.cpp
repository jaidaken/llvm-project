//===--- X86PreferSignedJcc.cpp - Rewrite unsigned JCC to signed ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: MSVC 6.0 uses signed conditional branches (jl, jge, jle, jg)
// where Clang uses unsigned (jb, jae, jbe, ja) for comparisons of values
// known to be non-negative. This pass rewrites unsigned to signed equivalents.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "MCTargetDesc/X86BaseInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-signed-jcc"
#define X86_PREFER_SIGNED_JCC_NAME "X86 prefer signed JCC pass"

namespace {
class X86PreferSignedJccPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferSignedJccPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return X86_PREFER_SIGNED_JCC_NAME; }
};
} // end anonymous namespace

char X86PreferSignedJccPass::ID = 0;

bool X86PreferSignedJccPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::PreferDiv)) {
    return false;
  }

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.getOpcode() != X86::JCC_1 && MI.getOpcode() != X86::JCC_4)
        continue;

      // JCC_1 format: operand 0 = MBB target, operand 1 = condition code
      MachineOperand &CondOp = MI.getOperand(1);
      if (!CondOp.isImm())
        continue;

      int64_t CC = CondOp.getImm();
      int64_t NewCC = CC;

      // MSVC 6.0 uses unsigned JCC after TEST, signed JCC after CMP.
      // Scan backward to find the flag-setting instruction.
      bool flagSetByTest = false;
      {
        auto It = MachineBasicBlock::iterator(&MI);
        while (It != MBB.begin()) {
          --It;
          unsigned Opc = It->getOpcode();
          if (Opc == X86::TEST32rr || Opc == X86::TEST16rr ||
              Opc == X86::TEST8rr || Opc == X86::TEST32ri ||
              Opc == X86::TEST8ri) {
            flagSetByTest = true;
            break;
          }
          if (It->modifiesRegister(X86::EFLAGS, /*TRI=*/nullptr))
            break;
        }
      }
      if (flagSetByTest)
        continue; // Keep unsigned JCC after TEST

      switch (CC) {
      case X86::COND_B:  NewCC = X86::COND_L;  break; // jb -> jl
      case X86::COND_AE: NewCC = X86::COND_GE; break; // jae -> jge
      case X86::COND_BE: NewCC = X86::COND_LE; break; // jbe -> jle
      case X86::COND_A:  NewCC = X86::COND_G;  break; // ja -> jg
      default: break;
      }

      if (NewCC != CC) {
        CondOp.setImm(NewCC);
        Changed = true;
      }
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferSignedJccPass() {
  return new X86PreferSignedJccPass();
}

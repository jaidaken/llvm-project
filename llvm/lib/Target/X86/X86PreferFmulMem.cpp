//===---- X86PreferFmulMem.cpp - Fold fld+fmulp into fmul [mem] -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: This pass folds fld+fmulp sequences into memory-form fmul
// for functions with the PreferFmulMem attribute.
//
// MSVC 6.0 generates "fld [a]; fmul dword ptr [b]" while LLVM generates
// "fld [a]; fld [b]; fmulp st(1), st(0)". This pass detects the
// LD_F32m/LD_F64m + MUL_FPrST0 pattern and folds it into MUL_F32m/MUL_F64m.
//
// Must run after X86FloatingPointStackifierPass (in addPreEmitPass).
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-fmul-mem"
#define X86_PREFER_FMUL_MEM_NAME "X86 prefer memory-form fmul pass"

namespace {
class X86PreferFmulMemPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferFmulMemPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return X86_PREFER_FMUL_MEM_NAME; }
};
} // end anonymous namespace

char X86PreferFmulMemPass::ID = 0;

/// Get the memory-form multiply opcode for a given FLD opcode.
/// Returns 0 if no folding is possible.
static unsigned getMulMemOpcodeForFld(unsigned FldOpcode) {
  switch (FldOpcode) {
  case X86::LD_F32m: return X86::MUL_F32m;
  case X86::LD_F64m: return X86::MUL_F64m;
  default: return 0;
  }
}

bool X86PreferFmulMemPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::PreferFmulMem))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E; /*below*/) {
      MachineInstr &MI = *I;

      // Look for LD_F32m or LD_F64m (fld dword/qword ptr [mem]).
      unsigned MulMemOpc = getMulMemOpcodeForFld(MI.getOpcode());
      if (!MulMemOpc) {
        ++I;
        continue;
      }

      // Check if the next instruction is MUL_FPrST0 (fmulp st(1), st(0)).
      auto NextI = std::next(I);
      if (NextI == E || NextI->getOpcode() != X86::MUL_FPrST0) {
        ++I;
        continue;
      }

      DebugLoc DL = MI.getDebugLoc();

      // Build memory-form fmul with the same memory operands as the fld.
      // fmul dword/qword ptr [addr] — multiplies ST(0) by memory.
      auto MIB = BuildMI(MBB, MI, DL, TII->get(MulMemOpc));
      // Copy memory operands from the FLD (operands 0..N are the address).
      for (unsigned i = 0; i < MI.getNumOperands(); ++i)
        MIB.add(MI.getOperand(i));
      MIB.cloneMemRefs(MI);

      // Erase both the FLD and FMULP.
      auto EraseI = I;
      I = std::next(NextI);
      NextI->eraseFromParent();
      EraseI->eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferFmulMemPass() {
  return new X86PreferFmulMemPass();
}

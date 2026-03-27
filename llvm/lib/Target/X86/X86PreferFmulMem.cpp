//===---- X86PreferFmulMem.cpp - Fold fld+fop into fop [mem] --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: This pass folds fld+fop sequences into memory-form FPU ops
// for functions with the PreferFmulMem attribute.
//
// MSVC 6.0 generates "fmul dword ptr [mem]", "fadd dword ptr [mem]",
// "fsubr dword ptr [mem]", etc. while LLVM generates "fld [mem]; fmulp"
// (load then operate). This pass detects LD_F32m/LD_F64m followed by
// MUL_FPrST0, ADD_FPrST0, SUB_FPrST0, or SUBR_FPrST0 and folds them
// into the corresponding memory-form instruction.
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

/// Get the memory-form opcode for a given FLD + FPU-op pair.
/// Returns 0 if no folding is possible.
static unsigned getMemFormOpcode(unsigned FldOpcode, unsigned FpOpcode) {
  // Determine the size suffix from the FLD opcode.
  bool IsF32;
  switch (FldOpcode) {
  case X86::LD_F32m: IsF32 = true;  break;
  case X86::LD_F64m: IsF32 = false; break;
  default: return 0;
  }

  // Map the FPU pop-form opcode to the corresponding memory-form opcode.
  switch (FpOpcode) {
  case X86::MUL_FPrST0:  return IsF32 ? X86::MUL_F32m  : X86::MUL_F64m;
  case X86::ADD_FPrST0:  return IsF32 ? X86::ADD_F32m  : X86::ADD_F64m;
  case X86::SUB_FPrST0:  return IsF32 ? X86::SUB_F32m  : X86::SUB_F64m;
  case X86::SUBR_FPrST0: return IsF32 ? X86::SUBR_F32m : X86::SUBR_F64m;
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

      // Skip non-FLD instructions early.
      unsigned FldOpc = MI.getOpcode();
      if (FldOpc != X86::LD_F32m && FldOpc != X86::LD_F64m) {
        ++I;
        continue;
      }

      // Check if the next instruction is a foldable FPU pop-form op.
      auto NextI = std::next(I);
      if (NextI == E) {
        ++I;
        continue;
      }

      unsigned MemFormOpc = getMemFormOpcode(FldOpc, NextI->getOpcode());
      if (!MemFormOpc) {
        ++I;
        continue;
      }

      DebugLoc DL = MI.getDebugLoc();

      // Build the memory-form instruction with the same memory operands
      // as the FLD.
      auto MIB = BuildMI(MBB, MI, DL, TII->get(MemFormOpc));
      for (unsigned i = 0; i < MI.getNumOperands(); ++i)
        MIB.add(MI.getOperand(i));
      MIB.cloneMemRefs(MI);

      // Erase both the FLD and the FPU pop-form instruction.
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

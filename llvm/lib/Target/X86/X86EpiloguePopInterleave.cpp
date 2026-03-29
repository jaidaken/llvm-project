//===--- X86EpiloguePopInterleave.cpp - Interleave MOV EAX with POPs ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: MSVC 6.0 interleaves `mov eax, imm` between epilogue pops:
//
//   pop edi
//   mov eax, 1
//   pop esi
//   ret
//
// Clang groups the mov before all pops:
//
//   mov eax, 1
//   pop edi
//   pop esi
//   ret
//
// This pass moves the MOV32ri EAX, imm from before the first POP to after
// the Nth POP (specified by the attribute parameter).
//
// Gated on the "epilogue_pop_interleave" function attribute with a parameter
// specifying after which POP to insert (e.g., "1" = after first POP).
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-epilogue-pop-interleave"

namespace {

/// Returns true if MI is a POP32r instruction.
static bool isPop32r(const MachineInstr &MI) {
  return MI.getOpcode() == X86::POP32r;
}

/// Returns true if MI is MOV32ri with EAX destination.
static bool isMovEaxImm(const MachineInstr &MI) {
  return MI.getOpcode() == X86::MOV32ri &&
         MI.getOperand(0).getReg() == X86::EAX;
}

class X86EpiloguePopInterleavePass : public MachineFunctionPass {
public:
  static char ID;
  X86EpiloguePopInterleavePass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 epilogue POP interleave for MSVC 6.0";
  }
};
} // end anonymous namespace

char X86EpiloguePopInterleavePass::ID = 0;

bool X86EpiloguePopInterleavePass::runOnMachineFunction(MachineFunction &MF) {
  const Function &F = MF.getFunction();
  if (!F.hasFnAttribute("epilogue_pop_interleave"))
    return false;

  StringRef AttrVal =
      F.getFnAttribute("epilogue_pop_interleave").getValueAsString();

  unsigned AfterPop = 0;
  if (AttrVal.getAsInteger(10, AfterPop) || AfterPop == 0)
    return false;

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Look for a MOV32ri EAX, imm followed by one or more POPs.
    // Walk the block looking for a return instruction, then scan backward.
    for (auto I = MBB.begin(), E = MBB.end(); I != E; ) {
      MachineInstr &MI = *I;

      // Find MOV32ri EAX, imm.
      if (!isMovEaxImm(MI)) {
        ++I;
        continue;
      }

      // Collect consecutive POP32r instructions after the MOV.
      SmallVector<MachineInstr *, 8> Pops;
      auto Next = std::next(I);
      while (Next != E) {
        if (Next->isPseudo() || Next->isDebugInstr()) {
          ++Next;
          continue;
        }
        if (isPop32r(*Next)) {
          Pops.push_back(&*Next);
          ++Next;
        } else {
          break;
        }
      }

      // Need enough pops to place the MOV after the Nth one.
      if (Pops.size() < AfterPop) {
        ++I;
        continue;
      }

      // The instruction after the Nth pop is where we insert.
      auto InsertPoint = std::next(MachineBasicBlock::iterator(*Pops[AfterPop - 1]));

      // Move the MOV to after the Nth POP.
      MBB.splice(InsertPoint, &MBB, I);

      Changed = true;
      // Restart scan from the moved instruction's new position
      // (we only do one rewrite per block).
      break;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86EpiloguePopInterleavePass() {
  return new X86EpiloguePopInterleavePass();
}

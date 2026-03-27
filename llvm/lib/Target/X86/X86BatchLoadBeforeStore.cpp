//===--- X86BatchLoadBeforeStore.cpp - Batch last loads before stores ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// deusex-decomp: MSVC 6.0 interleaves load-store pairs for struct copies,
// but batches the last two loads before their stores because the final load
// clobbers the source base register:
//
//   mov edx, [ecx+4]     ; L0   \
//   mov [eax+4], edx     ; S0   / interleaved pair
//   mov edx, [ecx+8]     ; L1   \
//   mov [eax+8], edx     ; S1   / interleaved pair
//   mov edx, [ecx+0xc]   ; L2   \
//   mov ecx, [ecx+0x10]  ; L3    | batched: last 2 loads
//   mov [eax+0xc], edx   ; S2    | then last 2 stores
//   mov [eax+0x10], ecx  ; S3   /
//
// Clang interleaves all pairs: L0 S0 L1 S1 L2 S2 L3 S3.
//
// This pass detects sequences of interleaved load-store pairs and reorders
// the last 2 pairs from (L S L S) to (L L S S). It also handles the case
// where a MOV32mi (immediate store like vtable set) appears between the
// load-store pairs.
//
// Gate: function attribute "batch_load_store".
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-batch-load-before-store"

namespace {
class X86BatchLoadBeforeStorePass : public MachineFunctionPass {
public:
  static char ID;
  X86BatchLoadBeforeStorePass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 batch last loads before stores (MSVC 6.0 eval order)";
  }
};
} // end anonymous namespace

char X86BatchLoadBeforeStorePass::ID = 0;

static bool isLoad32(const MachineInstr &MI) {
  return MI.getOpcode() == X86::MOV32rm;
}

static bool isStore32(const MachineInstr &MI) {
  return MI.getOpcode() == X86::MOV32mr;
}

static bool isImmStore32(const MachineInstr &MI) {
  return MI.getOpcode() == X86::MOV32mi;
}

/// Check if a load's destination register is the same as its base register
/// (e.g., mov ecx, [ecx+N] - clobbers the base).
static bool loadClobbersBase(const MachineInstr &MI) {
  if (!isLoad32(MI))
    return false;
  Register Dst = MI.getOperand(0).getReg();
  Register Base = MI.getOperand(1).getReg();
  return Dst == Base;
}

bool X86BatchLoadBeforeStorePass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("batch_load_store"))
    return false;

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    auto I = MBB.begin();
    while (I != MBB.end()) {
      // Collect interleaved load-store pairs.
      // Also allow an immediate store (MOV32mi like vtable set) to appear
      // adjacent to load-store pairs without breaking the sequence.
      if (!isLoad32(*I)) {
        ++I;
        continue;
      }

      struct LoadStorePair {
        MachineInstr *Load;
        MachineInstr *Store;
      };
      SmallVector<LoadStorePair, 8> Pairs;

      auto J = I;
      while (J != MBB.end()) {
        // Skip immediate stores (vtable sets) in the middle
        if (isImmStore32(*J)) {
          ++J;
          continue;
        }
        if (!isLoad32(*J))
          break;
        auto NextJ = std::next(J);
        if (NextJ == MBB.end() || !isStore32(*NextJ))
          break;
        Pairs.push_back({&*J, &*NextJ});
        J = std::next(NextJ);
      }

      if (Pairs.size() < 2) {
        ++I;
        continue;
      }

      // Check if the last load clobbers its base register.
      // If so, we MUST batch - the last load must happen before the
      // second-to-last store to preserve the base register value.
      // Even if it doesn't clobber, MSVC still batches the last 2 as a
      // stylistic choice, so we always batch.
      //
      // Current: ... L(n-2) S(n-2) L(n-1) S(n-1)
      // Target:  ... L(n-2) L(n-1) S(n-2) S(n-1)
      unsigned last = Pairs.size() - 1;
      unsigned prev = Pairs.size() - 2;

      MachineInstr *PrevStore = Pairs[prev].Store;
      MachineInstr *LastLoad = Pairs[last].Load;

      MBB.splice(MachineBasicBlock::iterator(PrevStore),
                 &MBB,
                 MachineBasicBlock::iterator(LastLoad));

      Changed = true;
      I = std::next(MachineBasicBlock::iterator(Pairs[last].Store));
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86BatchLoadBeforeStorePass() {
  return new X86BatchLoadBeforeStorePass();
}

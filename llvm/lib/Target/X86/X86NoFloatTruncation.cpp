//===--- X86NoFloatTruncation.cpp - Remove float return truncation --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: Clang inserts fstp dword [esp+N]; fld dword [esp+N] to truncate
// a double-precision FPU result to float precision on return. MSVC 6.0 does
// not truncate - it leaves the full-precision value in ST(0).
//
// This pass removes store+reload pairs at the same ESP-relative address when
// the function has the no_float_truncation attribute. Both instructions must
// be adjacent (fstp followed immediately by fld at the same address).
//
// Must run after the FP stackifier (which lowers FP pseudos to real x87).
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-no-float-truncation"
#define X86_NO_FLOAT_TRUNCATION_NAME \
  "X86 remove float return truncation (fstp+fld -> nop)"

namespace {
class X86NoFloatTruncationPass : public MachineFunctionPass {
public:
  static char ID;
  X86NoFloatTruncationPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return X86_NO_FLOAT_TRUNCATION_NAME;
  }
};
} // end anonymous namespace

char X86NoFloatTruncationPass::ID = 0;

/// Check whether two memory operand sequences refer to the same address.
/// Each x86 memory operand is 5 values: base, scale, index, disp, segment.
/// StoreStart/LoadStart are the operand indices where the memory ops begin.
static bool isSameMemAddress(const MachineInstr &Store, unsigned StoreStart,
                             const MachineInstr &Load, unsigned LoadStart) {
  // base register
  if (!Store.getOperand(StoreStart).isReg() ||
      !Load.getOperand(LoadStart).isReg())
    return false;
  if (Store.getOperand(StoreStart).getReg() !=
      Load.getOperand(LoadStart).getReg())
    return false;

  // scale (immediate)
  if (Store.getOperand(StoreStart + 1).getImm() !=
      Load.getOperand(LoadStart + 1).getImm())
    return false;

  // index register
  if (Store.getOperand(StoreStart + 2).getReg() !=
      Load.getOperand(LoadStart + 2).getReg())
    return false;

  // displacement
  const MachineOperand &StoreDisp = Store.getOperand(StoreStart + 3);
  const MachineOperand &LoadDisp = Load.getOperand(LoadStart + 3);
  if (StoreDisp.isImm() && LoadDisp.isImm()) {
    if (StoreDisp.getImm() != LoadDisp.getImm())
      return false;
  } else {
    return false; // non-immediate displacements - don't try to match
  }

  // segment register
  if (Store.getOperand(StoreStart + 4).getReg() !=
      Load.getOperand(LoadStart + 4).getReg())
    return false;

  return true;
}

bool X86NoFloatTruncationPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::NoFloatTruncation))
    return false;

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E; /*incremented below*/) {
      MachineInstr &StoreMI = *I;

      // Step 1: Find ST_FP32m (fstp dword ptr [addr])
      if (StoreMI.getOpcode() != X86::ST_FP32m) {
        ++I;
        continue;
      }

      // Step 2: Find LD_F32m (fld dword ptr [addr]) immediately after
      auto LoadI = std::next(I);
      if (LoadI == E || LoadI->getOpcode() != X86::LD_F32m) {
        ++I;
        continue;
      }

      // Step 3: Verify both access the same memory address.
      // ST_FP32m operands: 0=base, 1=scale, 2=index, 3=disp, 4=seg
      // LD_F32m  operands: 0=base, 1=scale, 2=index, 3=disp, 4=seg
      if (!isSameMemAddress(StoreMI, 0, *LoadI, 0)) {
        ++I;
        continue;
      }

      // Matched: fstp [addr]; fld [addr] - a truncation round-trip.
      // Remove both. The value stays in ST(0) at full precision.
      auto NextI = std::next(LoadI);
      LoadI->eraseFromParent();
      StoreMI.eraseFromParent();
      I = NextI;
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86NoFloatTruncationPass() {
  return new X86NoFloatTruncationPass();
}

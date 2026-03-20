//===--- X86PreferEarlyBufAdvance.cpp - Hoist pointer advance in loops ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: Moves the buffer pointer advance (add esi, 16) from after all
// loads to after the first load, converting remaining positive offsets to
// negative. Matches MSVC 6.0's instruction scheduling for unrolled loops.
//
// Before: mov dl,[esi]; mov dl,[esi+1]; ... mov dl,[esi+15]; add esi,16
// After:  mov dl,[esi]; add esi,16; mov dl,[esi-15]; ... mov dl,[esi-1]
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "MCTargetDesc/X86BaseInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-early-buf-advance"
#define X86_PREFER_EARLY_BUF_ADVANCE_NAME "X86 prefer early buf advance pass"

namespace {
class X86PreferEarlyBufAdvancePass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferEarlyBufAdvancePass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return X86_PREFER_EARLY_BUF_ADVANCE_NAME;
  }
};
} // end anonymous namespace

char X86PreferEarlyBufAdvancePass::ID = 0;

bool X86PreferEarlyBufAdvancePass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::Msvc6RegAlloc))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Find ADD32ri8 REG, 16 (or ADD32ri REG, 16) in this block.
    MachineInstr *AddMI = nullptr;
    MCPhysReg BaseReg = 0;

    for (MachineInstr &MI : MBB) {
      if ((MI.getOpcode() == X86::ADD32ri8 ||
           MI.getOpcode() == X86::ADD32ri) &&
          MI.getOperand(2).getImm() == 16) {
        MCPhysReg Reg = MI.getOperand(0).getReg();
        if (Reg == X86::ESI || Reg == X86::EDI || Reg == X86::EBX) {
          AddMI = &MI;
          BaseReg = Reg;
        }
      }
    }

    if (!AddMI)
      continue;

    // Find the first load using BaseReg as memory base with offset 0.
    // This is the first load in the unrolled sequence.
    MachineInstr *FirstLoad = nullptr;
    SmallVector<MachineInstr *, 16> SubsequentLoads;

    for (MachineInstr &MI : MBB) {
      if (&MI == AddMI)
        break; // Stop at the ADD instruction

      int MemOpIdx = X86II::getMemoryOperandNo(MI.getDesc().TSFlags);
      if (MemOpIdx < 0)
        continue;
      MemOpIdx += X86II::getOperandBias(MI.getDesc());

      unsigned BaseIdx = MemOpIdx + X86::AddrBaseReg;
      unsigned DispIdx = MemOpIdx + X86::AddrDisp;
      unsigned IdxIdx = MemOpIdx + X86::AddrIndexReg;
      unsigned ScaleIdx = MemOpIdx + X86::AddrScaleAmt;

      if (BaseIdx >= MI.getNumOperands() ||
          DispIdx >= MI.getNumOperands())
        continue;

      const MachineOperand &BaseMO = MI.getOperand(BaseIdx);
      const MachineOperand &DispMO = MI.getOperand(DispIdx);
      const MachineOperand &IdxMO = MI.getOperand(IdxIdx);

      if (!BaseMO.isReg() || BaseMO.getReg() != BaseReg)
        continue;
      if (IdxMO.isReg() && IdxMO.getReg() != 0)
        continue; // Has index register, skip
      if (!DispMO.isImm())
        continue;

      int64_t Disp = DispMO.getImm();

      if (!FirstLoad && Disp == 0) {
        FirstLoad = &MI;
      } else if (FirstLoad && Disp > 0 && Disp < 16) {
        SubsequentLoads.push_back(&MI);
      }
    }

    // Need at least the first load and some subsequent loads
    if (!FirstLoad || SubsequentLoads.size() < 10)
      continue;

    // Move the ADD right after the first load.
    // Find the instruction after FirstLoad.
    auto InsertPt = std::next(MachineBasicBlock::iterator(FirstLoad));

    // Remove AddMI from its current position and insert after first load.
    AddMI->removeFromParent();
    MBB.insert(InsertPt, AddMI);

    // Now adjust all subsequent loads: change offset from +N to -(16-N)
    for (MachineInstr *LoadMI : SubsequentLoads) {
      int MemOpIdx2 = X86II::getMemoryOperandNo(LoadMI->getDesc().TSFlags);
      MemOpIdx2 += X86II::getOperandBias(LoadMI->getDesc());
      unsigned DispIdx2 = MemOpIdx2 + X86::AddrDisp;

      MachineOperand &DispMO2 = LoadMI->getOperand(DispIdx2);
      int64_t OldDisp = DispMO2.getImm();
      int64_t NewDisp = OldDisp - 16; // +1 becomes -15, +15 becomes -1
      DispMO2.setImm(NewDisp);
    }

    Changed = true;
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferEarlyBufAdvancePass() {
  return new X86PreferEarlyBufAdvancePass();
}

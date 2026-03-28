//===--- X86FoldTestMem.cpp - Fold MOVZX+TEST into TEST [mem] ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: Clang loads a byte into a register then tests the register:
//   movzbl [mem], reg    ; MOVZX32rm8 (opcode 0F B6)
//   test reg, imm        ; TEST8ri
//
// MSVC 6.0 tests memory directly:
//   test byte [mem], imm ; TEST8mi (opcode F6 /0)
//
// This pass, gated on the "fold_test_mem" string attribute, folds the
// MOVZX32rm8 + TEST8ri pair into a single TEST8mi when the loaded register
// is dead after the TEST instruction.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "x86-fold-test-mem"

namespace {
class X86FoldTestMemPass : public MachineFunctionPass {
public:
  static char ID;
  X86FoldTestMemPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 fold MOVZX+TEST into TEST [mem]";
  }
};
} // end anonymous namespace

char X86FoldTestMemPass::ID = 0;

/// Check if a register (and all its super/sub registers) is dead after a
/// given point in the basic block.
static bool isRegDeadAfter(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator AfterPoint,
                           Register Reg, const TargetRegisterInfo *TRI) {
  // Walk forward from AfterPoint to end of block.
  for (auto I = AfterPoint, E = MBB.end(); I != E; ++I) {
    MachineInstr &MI = *I;
    if (MI.isDebugInstr())
      continue;

    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || MO.getReg() == 0)
        continue;

      // If any overlapping register is read before being defined, it's live.
      if (MO.isUse() && TRI->regsOverlap(MO.getReg(), Reg))
        return false;

      // If the full register is defined (overwritten), it's dead from here.
      if (MO.isDef() && TRI->regsOverlap(MO.getReg(), Reg))
        return true;
    }

    // At a terminator or call, assume the register may be live-out.
    if (MI.isTerminator() || MI.isCall())
      return false;
  }

  // Reached end of block without overwrite - conservatively assume live-out.
  return false;
}

bool X86FoldTestMemPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("fold_test_mem"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E;) {
      MachineInstr &LoadMI = *I;

      // Look for MOVZX32rm8: movzx reg32, byte ptr [mem]
      if (LoadMI.getOpcode() != X86::MOVZX32rm8) {
        ++I;
        continue;
      }

      Register DstReg32 = LoadMI.getOperand(0).getReg();
      Register DstReg8 = TRI->getSubReg(DstReg32, X86::sub_8bit);
      if (!DstReg8) {
        ++I;
        continue;
      }

      // Next non-debug instruction must be TEST8ri using the 8-bit subreg.
      auto TestIt = std::next(I);
      while (TestIt != E && TestIt->isDebugInstr())
        ++TestIt;
      if (TestIt == E) {
        ++I;
        continue;
      }

      MachineInstr &TestMI = *TestIt;
      if (TestMI.getOpcode() != X86::TEST8ri) {
        ++I;
        continue;
      }

      // TEST8ri operands: reg(0), imm(1)
      if (TestMI.getOperand(0).getReg() != DstReg8) {
        ++I;
        continue;
      }

      int64_t ImmVal = TestMI.getOperand(1).getImm();

      // Verify the register is dead after the TEST.
      auto AfterTest = std::next(TestIt);
      if (!isRegDeadAfter(MBB, AfterTest, DstReg32, TRI)) {
        ++I;
        continue;
      }

      LLVM_DEBUG(dbgs() << "FoldTestMem: folding MOVZX32rm8+TEST8ri into "
                        << "TEST8mi in " << MF.getName() << "\n");

      // Build TEST8mi: test byte ptr [mem], imm
      // TEST8mi operands: base, scale, index, disp, segment, imm
      MachineInstrBuilder NewMI =
          BuildMI(MBB, LoadMI, LoadMI.getDebugLoc(), TII->get(X86::TEST8mi));
      // Copy memory operands from the MOVZX (operands 1..N are mem ops).
      for (unsigned i = 1; i < LoadMI.getNumOperands(); ++i)
        NewMI.add(LoadMI.getOperand(i));
      NewMI.addImm(ImmVal);
      NewMI.setMemRefs(LoadMI.memoperands());

      // Remove both original instructions.
      auto NextI = std::next(TestIt);
      TestMI.eraseFromParent();
      LoadMI.eraseFromParent();
      I = NextI;
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86FoldTestMemPass() {
  return new X86FoldTestMemPass();
}

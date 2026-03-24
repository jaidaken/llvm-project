//===--- X86SplitCondJmp.cpp - Split near Jcc into Je+Jmp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: MSVC 6.0 generates a split branch pattern for null-check thunks:
//
//   test ecx, ecx         ; null check
//   je skip               ; 74 05 (short je, 2 bytes)
//   jmp target            ; e9 XX XX XX XX (near jmp, 5 bytes)
//   skip:
//   ret                   ; c3
//
// LLVM optimizes this to a single conditional jump:
//
//   jne target            ; 0f 85 XX XX XX XX (near jne, 6 bytes)
//   ret
//
// This pass converts the near conditional jump back to the split pattern.
// It finds a near Jcc (JCC_4) followed by RET, inverts the condition to a
// short Jcc (JCC_1) that skips over a near JMP (JMP_4) to the original target.
//
// Gated on the "split_cond_jmp" function attribute.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "MCTargetDesc/X86BaseInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-split-cond-jmp"
#define X86_SPLIT_COND_JMP_NAME "X86 split conditional jump into je+jmp"

namespace {
class X86SplitCondJmpPass : public MachineFunctionPass {
public:
  static char ID;
  X86SplitCondJmpPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return X86_SPLIT_COND_JMP_NAME; }
};
} // end anonymous namespace

char X86SplitCondJmpPass::ID = 0;

/// Check if the opcode is a RET instruction (with or without stack cleanup).
static bool isRetOpcode(unsigned Opc) {
  return Opc == X86::RET32 || Opc == X86::RET64 ||
         Opc == X86::RETI32 || Opc == X86::RETI64;
}

bool X86SplitCondJmpPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("split_cond_jmp"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E;) {
      MachineInstr &MI = *I++;

      // Match JCC_1 or JCC_4 (conditional jump to a distant target).
      // At this stage in the pipeline, branches may still be JCC_1 (short
      // form); the assembler relaxes them to JCC_4 (near form) later.
      if (MI.getOpcode() != X86::JCC_1 && MI.getOpcode() != X86::JCC_4)
        continue;

      // The JCC must be followed by a RET in the same basic block.
      if (I == E)
        continue;

      MachineInstr &NextMI = *I;
      if (!isRetOpcode(NextMI.getOpcode()))
        continue;

      // Extract the original condition code and target.
      // JCC_4 format: operand 0 = MBB target, operand 1 = condition code (imm)
      MachineOperand &TargetOp = MI.getOperand(0);
      MachineOperand &CondOp = MI.getOperand(1);
      if (!CondOp.isImm())
        continue;

      X86::CondCode OrigCC =
          static_cast<X86::CondCode>(CondOp.getImm());
      X86::CondCode InvCC = X86::GetOppositeBranchCondition(OrigCC);

      MachineBasicBlock *TargetMBB = TargetOp.getMBB();
      DebugLoc DL = MI.getDebugLoc();

      // We need a label for the "skip" target. The RET instruction that
      // follows is the skip target. We create a new MBB to hold the RET
      // so that the short Jcc can reference it as a branch target.
      //
      // Before: [... | JCC target | RET | ...]
      // After:  [... | JCC_1 skip | JMP_4 target | skip: RET | ...]
      //
      // Split the current block: everything from the RET onward goes into
      // a new "skip" block.
      MachineBasicBlock *SkipMBB = MF.CreateMachineBasicBlock();
      MF.insert(std::next(MachineFunction::iterator(&MBB)), SkipMBB);

      // Move the RET (and anything after it, though typically nothing) to SkipMBB.
      SkipMBB->splice(SkipMBB->end(), &MBB, I, MBB.end());

      // Fix up the successor list. The original MBB had TargetMBB as a
      // successor (from the JCC). Remove it and re-add with the correct
      // topology: MBB branches to SkipMBB (inverted Jcc) or TargetMBB (JMP).
      // SkipMBB contains only RET and has no successors.
      MBB.removeSuccessor(TargetMBB);
      MBB.addSuccessor(SkipMBB);
      MBB.addSuccessor(TargetMBB);

      // Build the inverted short conditional jump: JCC_1 skip
      BuildMI(MBB, MI, DL, TII->get(X86::JCC_1))
          .addMBB(SkipMBB)
          .addImm(InvCC);

      // Build the near unconditional jump: JMP_4 target
      BuildMI(MBB, MI, DL, TII->get(X86::JMP_4))
          .addMBB(TargetMBB);

      // Remove the original JCC_4.
      MI.eraseFromParent();

      Changed = true;
      break; // MBB has been split; stop iterating this block.
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86SplitCondJmpPass() {
  return new X86SplitCondJmpPass();
}

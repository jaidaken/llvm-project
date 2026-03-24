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

      // --- Pattern 1: TAILJMPd_CC (conditional tail jump pseudo) ---
      // LLVM uses this for: if (p) tailcall(p);
      // We convert: TAILJMPd_CC target, cc -> JCC_1 skip, !cc; JMP_4 target; skip: RET
      if (MI.getOpcode() == X86::TAILJMPd_CC) {
        // TAILJMPd_CC operands: 0=target(global), 1=condcode(imm), 2+=regmask/implicits
        MachineOperand &TargetOp = MI.getOperand(0);
        MachineOperand &CondOp = MI.getOperand(1);
        if (!CondOp.isImm())
          continue;

        X86::CondCode OrigCC =
            static_cast<X86::CondCode>(CondOp.getImm());
        X86::CondCode InvCC = X86::GetOppositeBranchCondition(OrigCC);
        DebugLoc DL = MI.getDebugLoc();

        // Find the fall-through block that contains the RET.
        // The TAILJMPd_CC block should have a successor with a RET.
        MachineBasicBlock *SkipMBB = nullptr;
        for (auto *Succ : MBB.successors()) {
          if (!Succ->empty() && isRetOpcode(Succ->begin()->getOpcode())) {
            SkipMBB = Succ;
            break;
          }
        }
        if (!SkipMBB) {
          // No existing RET block found. Create one.
          SkipMBB = MF.CreateMachineBasicBlock();
          MF.insert(std::next(MachineFunction::iterator(&MBB)), SkipMBB);
          BuildMI(*SkipMBB, SkipMBB->end(), DL, TII->get(X86::RET32));
          MBB.addSuccessor(SkipMBB);
        }

        // Build inverted short conditional jump over the JMP.
        BuildMI(MBB, MI, DL, TII->get(X86::JCC_1))
            .addMBB(SkipMBB)
            .addImm(InvCC);

        // Build unconditional near jump to the original target.
        BuildMI(MBB, MI, DL, TII->get(X86::JMP_4))
            .add(TargetOp);

        // Remove the TAILJMPd_CC.
        MI.eraseFromParent();
        Changed = true;
        break;
      }

      // --- Pattern 2: JCC + RET in same block ---
      if (MI.getOpcode() != X86::JCC_1 && MI.getOpcode() != X86::JCC_4)
        continue;

      if (I == E)
        continue;

      MachineInstr &NextMI = *I;
      if (!isRetOpcode(NextMI.getOpcode()))
        continue;

      MachineOperand &TargetOp = MI.getOperand(0);
      MachineOperand &CondOp = MI.getOperand(1);
      if (!CondOp.isImm())
        continue;

      X86::CondCode OrigCC =
          static_cast<X86::CondCode>(CondOp.getImm());
      X86::CondCode InvCC = X86::GetOppositeBranchCondition(OrigCC);

      MachineBasicBlock *TargetMBB = TargetOp.getMBB();
      DebugLoc DL = MI.getDebugLoc();

      MachineBasicBlock *SkipMBB = MF.CreateMachineBasicBlock();
      MF.insert(std::next(MachineFunction::iterator(&MBB)), SkipMBB);
      SkipMBB->splice(SkipMBB->end(), &MBB, I, MBB.end());

      MBB.removeSuccessor(TargetMBB);
      MBB.addSuccessor(SkipMBB);
      MBB.addSuccessor(TargetMBB);

      BuildMI(MBB, MI, DL, TII->get(X86::JCC_1))
          .addMBB(SkipMBB)
          .addImm(InvCC);
      BuildMI(MBB, MI, DL, TII->get(X86::JMP_4))
          .addMBB(TargetMBB);

      MI.eraseFromParent();
      Changed = true;
      break;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86SplitCondJmpPass() {
  return new X86SplitCondJmpPass();
}

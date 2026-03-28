// bw1-decomp: Convert CRT guard immediate forms to register-register forms.
//
// MSVC 6.0 CRT static initializer guards use a specific instruction sequence:
//   mov cl, byte ptr [guard]
//   mov al, 0x01
//   test al, cl
//   jne skip
//   or.s cl, al
//   mov byte ptr [guard], cl
//
// Clang generates the guard variable load via MOVZX32rm8 into EAX, then uses
// TEST8ri and OR8ri with AL. This pass recognizes the full pattern and
// restructures it:
//   1. Replace MOVZX32rm8 $eax with MOV8rm $cl (no zero-extend, use CL)
//   2. Insert MOV8ri AL, 1 to materialize the constant
//   3. Replace TEST8ri AL, 1 with TEST8rr AL, CL
//   4. Replace OR8ri AL, 1 with OR8rr CL, AL
//   5. Fix the store from MOV8mr [guard], AL to MOV8mr [guard], CL
//
// The TestRev pass then swaps TEST operand order, and the ReversedOps pass
// handles the OR8rr_REV encoding.
//
// Gated by the crt_guard_pattern function attribute.

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-crt-guard-pattern"

namespace {
class X86CrtGuardPatternPass : public MachineFunctionPass {
public:
  static char ID;
  X86CrtGuardPatternPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 CRT guard pattern (TEST8ri/OR8ri -> register forms)";
  }
};
} // end anonymous namespace

char X86CrtGuardPatternPass::ID = 0;

/// Replace a register in a MachineOperand if it matches OldReg.
static void replaceRegInOperand(MachineOperand &MO, Register OldReg,
                                Register NewReg) {
  if (MO.isReg() && MO.getReg() == OldReg)
    MO.setReg(NewReg);
}

bool X86CrtGuardPatternPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::CrtGuardPattern))
    return false;

  const X86Subtarget &ST = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = ST.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E;) {
      MachineInstr &MI = *I;
      ++I;

      if (MI.getOpcode() != X86::TEST8ri)
        continue;

      // Only handle TEST8ri with immediate 1 (the guard bit check).
      if (!MI.getOperand(1).isImm() || MI.getOperand(1).getImm() != 1)
        continue;

      Register TestReg = MI.getOperand(0).getReg();
      DebugLoc DL = MI.getDebugLoc();

      // If the guard is in AL (conflicts with our constant register), we need
      // to reassign the guard to CL and fix up all related instructions.
      if (TestReg == X86::AL) {
        // Walk backward to find the MOVZX32rm8 that loads into EAX.
        MachineInstr *LoadMI = nullptr;
        for (auto RI = MachineBasicBlock::reverse_iterator(MI.getIterator()),
                  RE = MBB.rend();
             RI != RE; ++RI) {
          if (RI->getOpcode() == X86::MOVZX32rm8 &&
              RI->getOperand(0).getReg() == X86::EAX) {
            LoadMI = &*RI;
            break;
          }
        }

        if (LoadMI) {
          // Replace MOVZX32rm8 $eax, [guard] with MOV8rm $cl, [guard].
          // This avoids zero-extension (matches original MSVC 6 output) and
          // frees AL for the constant.
          auto MIB = BuildMI(MBB, *LoadMI, LoadMI->getDebugLoc(),
                             TII->get(X86::MOV8rm), X86::CL);
          // Copy memory operands (operands 1..N).
          for (unsigned i = 1; i < LoadMI->getNumOperands(); ++i)
            MIB.add(LoadMI->getOperand(i));
          MIB.cloneMemRefs(*LoadMI);
          LoadMI->eraseFromParent();
        }

        // Insert: MOV8ri AL, 1
        BuildMI(MBB, MI, DL, TII->get(X86::MOV8ri), X86::AL).addImm(1);

        // Insert: TEST8rr CL, AL (TestRev will swap to AL, CL for correct
        // ModR/M encoding 84 C8).
        BuildMI(MBB, MI, DL, TII->get(X86::TEST8rr))
            .addReg(X86::CL)
            .addReg(X86::AL);

        MI.eraseFromParent();

        // Fix up OR8ri and MOV8mr in successor blocks: change AL -> CL.
        for (MachineBasicBlock *Succ : MBB.successors()) {
          for (auto SI = Succ->begin(), SE = Succ->end(); SI != SE;) {
            MachineInstr &SMI = *SI;
            ++SI;

            if (SMI.getOpcode() == X86::OR8ri &&
                SMI.getOperand(0).getReg() == X86::AL &&
                SMI.getOperand(2).isImm() && SMI.getOperand(2).getImm() == 1) {
              // Replace OR8ri AL, 1 with OR8rr CL, AL.
              BuildMI(*Succ, SMI, SMI.getDebugLoc(), TII->get(X86::OR8rr),
                      X86::CL)
                  .addReg(X86::CL)
                  .addReg(X86::AL);
              SMI.eraseFromParent();
              continue;
            }

            if (SMI.getOpcode() == X86::MOV8mr) {
              // Change store from AL to CL.
              for (unsigned i = 0; i < SMI.getNumOperands(); ++i)
                replaceRegInOperand(SMI.getOperand(i), X86::AL, X86::CL);
            }
          }
        }

        Changed = true;
        continue;
      }

      // TestReg is not AL: simple case, just materialize constant into AL.
      // Insert: MOV8ri AL, 1
      BuildMI(MBB, MI, DL, TII->get(X86::MOV8ri), X86::AL).addImm(1);

      // Insert: TEST8rr TestReg, AL (TestRev will swap to AL, TestReg for
      // correct ModR/M encoding).
      BuildMI(MBB, MI, DL, TII->get(X86::TEST8rr))
          .addReg(TestReg)
          .addReg(X86::AL);

      MI.eraseFromParent();

      // Also handle OR8ri in successor blocks for the simple case.
      for (MachineBasicBlock *Succ : MBB.successors()) {
        for (auto SI = Succ->begin(), SE = Succ->end(); SI != SE;) {
          MachineInstr &SMI = *SI;
          ++SI;

          if (SMI.getOpcode() == X86::OR8ri &&
              SMI.getOperand(2).isImm() && SMI.getOperand(2).getImm() == 1) {
            Register OrReg = SMI.getOperand(0).getReg();
            BuildMI(*Succ, SMI, SMI.getDebugLoc(), TII->get(X86::OR8rr), OrReg)
                .addReg(OrReg)
                .addReg(X86::AL);
            SMI.eraseFromParent();
          }
        }
      }

      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86CrtGuardPatternPass() {
  return new X86CrtGuardPatternPass();
}

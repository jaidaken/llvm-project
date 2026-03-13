//===---- X86PreferSeteEcx.cpp - Route sete through ECX -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: This pass converts "sete al; movzx eax, al" into
// "sete cl; movzx eax, cl" for functions with the PreferSeteEcx attribute.
// MSVC 6.0 sometimes routes the sete result through CL/ECX before moving
// it to EAX, producing different instruction encodings.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-sete-ecx"
#define X86_PREFER_SETE_ECX_NAME "X86 prefer sete ecx pass"

namespace {
class X86PreferSeteEcxPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferSeteEcxPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return X86_PREFER_SETE_ECX_NAME; }
};
} // end anonymous namespace

char X86PreferSeteEcxPass::ID = 0;

/// Check if the opcode is a SETcc to an 8-bit register.
static bool isSETccReg(unsigned Opc) {
  switch (Opc) {
  case X86::SETEr:  case X86::SETNEr:
  case X86::SETAr:  case X86::SETAEr:
  case X86::SETBr:  case X86::SETBEr:
  case X86::SETGr:  case X86::SETGEr:
  case X86::SETLr:  case X86::SETLEr:
  case X86::SETSr:  case X86::SETNSr:
  case X86::SETPr:  case X86::SETNPr:
    return true;
  default:
    return false;
  }
}

bool X86PreferSeteEcxPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::PreferSeteEcx))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Two-pass approach: collect patterns, then replace.
    struct SetePattern {
      MachineInstr *SetccMI;
      MachineInstr *MovzxMI;
    };
    SmallVector<SetePattern, 4> Patterns;

    {
      LivePhysRegs LiveRegs(*TRI);
      LiveRegs.addLiveOuts(MBB);

      // Scan backward to track ECX liveness.
      for (auto I = MBB.rbegin(), E = MBB.rend(); I != E; ++I) {
        MachineInstr &MI = *I;

        // Look for MOVZX32rr8 EAX, AL (or similar with AL as source).
        if (MI.getOpcode() == X86::MOVZX32rr8) {
          Register DstReg = MI.getOperand(0).getReg();
          Register SrcReg = MI.getOperand(1).getReg();

          // Only convert when MOVZX reads from AL and writes to EAX.
          if (DstReg == X86::EAX && SrcReg == X86::AL) {
            // Check if ECX is dead here (we'll use CL as intermediary).
            if (!LiveRegs.contains(X86::ECX)) {
              // Look backward for the SETcc that writes AL.
              auto PrevI = std::next(I);
              while (PrevI != E) {
                MachineInstr &Prev = *PrevI;
                if (isSETccReg(Prev.getOpcode()) &&
                    Prev.getOperand(0).getReg() == X86::AL) {
                  Patterns.push_back({&Prev, &MI});
                  break;
                }
                // If something else writes AL, stop looking.
                if (Prev.modifiesRegister(X86::AL, TRI))
                  break;
                // If something reads EFLAGS, stop (SETcc needs EFLAGS).
                if (Prev.readsRegister(X86::EFLAGS, TRI))
                  break;
                ++PrevI;
              }
            }
          }
        }

        LiveRegs.stepBackward(MI);
      }
    }

    // Pass 2: Replace patterns.
    for (auto &P : Patterns) {
      DebugLoc DL = P.SetccMI->getDebugLoc();
      unsigned SetccOpc = P.SetccMI->getOpcode();

      // Replace: sete al; movzx eax, al
      // With:    sete cl; movzx eax, cl

      // New SETcc CL.
      BuildMI(MBB, *P.SetccMI, DL, TII->get(SetccOpc), X86::CL);

      // New MOVZX32rr8 EAX, CL.
      BuildMI(MBB, *P.MovzxMI, DL, TII->get(X86::MOVZX32rr8), X86::EAX)
          .addReg(X86::CL);

      P.MovzxMI->eraseFromParent();
      P.SetccMI->eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferSeteEcxPass() {
  return new X86PreferSeteEcxPass();
}

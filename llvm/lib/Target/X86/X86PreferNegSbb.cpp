//===---- X86PreferNegSbb.cpp - Boolean via neg+sbb+inc/neg ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: This pass converts LLVM boolean test patterns into MSVC 6.0
// neg+sbb idioms for functions with PreferNegSbb.
//
// COND_E (== 0): xor+test+sete  -> neg+sbb+inc
//   neg eax; sbb eax,eax; inc eax
//   val==0: CF=0, sbb→0, inc→1. val!=0: CF=1, sbb→-1, inc→0.
//
// COND_NE (!= 0): xor+test+setne -> neg+sbb+neg
//   neg eax; sbb eax,eax; neg eax
//   val==0: CF=0, sbb→0, neg→0. val!=0: CF=1, sbb→-1, neg→1.
//
// COND_E (== Imm): xor+cmp+sete  -> dec/sub+neg+sbb+inc
//   dec eax; neg eax; sbb eax,eax; inc eax   (when Imm==1)
//   sub eax,N; neg eax; sbb eax,eax; inc eax (when Imm>1)
//
// COND_NE (!= Imm): xor+cmp+setne -> dec/sub+neg+sbb+neg
//   dec eax; neg eax; sbb eax,eax; neg eax   (when Imm==1)
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-neg-sbb"
#define X86_PREFER_NEG_SBB_NAME "X86 prefer neg+sbb+inc pass"

namespace {
class X86PreferNegSbbPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferNegSbbPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return X86_PREFER_NEG_SBB_NAME; }
};
} // end anonymous namespace

char X86PreferNegSbbPass::ID = 0;

bool X86PreferNegSbbPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::PreferNegSbb))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Pattern A (same register, COND_E):
    //   XOR32rr Dst, Dst; TEST32rr Src, Src; SETCCr COND_E Dst_low8
    //   -> NEG32r Src; SBB32rr Dst, Dst; INC32r Dst
    //
    // Pattern B (same register, COND_NE):
    //   XOR32rr Dst, Dst; TEST32rr Src, Src; SETCCr COND_NE Dst_low8
    //   -> NEG32r Src; SBB32rr Dst, Dst; NEG32r Dst
    //
    // Pattern C (different registers, COND_NE + trailing MOV):
    //   XOR32rr Tmp, Tmp; TEST32rr Src, Src; SETCCr COND_NE Tmp_low8;
    //   MOV32rr Dst, Tmp
    //   -> NEG32r Src; SBB32rr Dst, Dst; NEG32r Dst
    for (auto I = MBB.begin(), E = MBB.end(); I != E; /*incremented below*/) {
      MachineInstr &XorMI = *I;

      // Step 1: Find XOR32rr/XOR32rr_REV Dst, Dst (self-xor = zero).
      unsigned XorOpc = XorMI.getOpcode();
      if (XorOpc != X86::XOR32rr && XorOpc != X86::XOR32rr_REV) {
        ++I;
        continue;
      }

      Register XorDst = XorMI.getOperand(0).getReg();
      Register XorSrc1 = XorMI.getOperand(1).getReg();
      Register XorSrc2 = XorMI.getOperand(2).getReg();
      if (XorSrc1 != XorDst || XorSrc2 != XorDst) {
        ++I;
        continue;
      }

      // Step 2: Find TEST32rr Src, Src or CMP32ri8/CMP32ri Src, Imm.
      auto TestI = std::next(I);
      if (TestI == E) {
        ++I;
        continue;
      }

      Register SrcReg;
      int64_t CmpImm = 0; // 0 = TEST (compare against zero)

      if (TestI->getOpcode() == X86::TEST32rr) {
        Register TestSrc1 = TestI->getOperand(0).getReg();
        Register TestSrc2 = TestI->getOperand(1).getReg();
        if (TestSrc1 != TestSrc2) {
          ++I;
          continue;
        }
        SrcReg = TestSrc1;
      } else if (TestI->getOpcode() == X86::CMP32ri8 ||
                 TestI->getOpcode() == X86::CMP32ri) {
        SrcReg = TestI->getOperand(0).getReg();
        CmpImm = TestI->getOperand(1).getImm();
        if (CmpImm <= 0) {
          ++I;
          continue; // Only handle positive immediates
        }
      } else {
        ++I;
        continue;
      }

      // Step 3: Find SETCCr with COND_E or COND_NE immediately after TEST.
      auto SetccI = std::next(TestI);
      if (SetccI == E || SetccI->getOpcode() != X86::SETCCr) {
        ++I;
        continue;
      }

      X86::CondCode CC = X86::getCondFromSETCC(*SetccI);
      if (CC != X86::COND_E && CC != X86::COND_NE) {
        ++I;
        continue;
      }

      // Check for trailing MOV32rr (Pattern C: different register for result).
      // If SETNE writes to XorDst (same reg), no MOV needed.
      // If SETNE writes to XorDst but there's a MOV after, check if we should
      // use the MOV destination as the final result register.
      MachineInstr *MovToErase = nullptr;
      Register FinalDst = XorDst; // default: SBB writes to XorDst
      auto AfterSetcc = std::next(SetccI);
      if (AfterSetcc != E) {
        unsigned MovOpc = AfterSetcc->getOpcode();
        if (MovOpc == X86::MOV32rr || MovOpc == X86::MOV32rr_REV) {
          Register MovSrc = AfterSetcc->getOperand(1).getReg();
          if (MovSrc == XorDst) {
            // Pattern C: XOR zeros Tmp, SETNE writes Tmp, MOV copies Tmp->Dst
            FinalDst = AfterSetcc->getOperand(0).getReg();
            MovToErase = &*AfterSetcc;
          }
        }
      }

      DebugLoc DL = XorMI.getDebugLoc();

      // When comparing against a non-zero immediate, emit DEC/SUB first
      // to shift the comparison to zero.
      if (CmpImm > 0) {
        if (CmpImm == 1) {
          // DEC32r is 1 byte (48+r), SUB32ri8 is 3 bytes
          BuildMI(MBB, XorMI, DL, TII->get(X86::DEC32r), SrcReg)
              .addReg(SrcReg);
        } else {
          BuildMI(MBB, XorMI, DL, TII->get(X86::SUB32ri8), SrcReg)
              .addReg(SrcReg)
              .addImm(CmpImm);
        }
      }

      // Build: NEG32r Src (sets CF=1 if Src!=0, CF=0 if Src==0)
      BuildMI(MBB, XorMI, DL, TII->get(X86::NEG32r), SrcReg)
          .addReg(SrcReg);

      // SBB32rr_REV FinalDst, FinalDst: CF=1 -> -1, CF=0 -> 0
      // Use _REV encoding (0x1b) to match MSVC 6.0 output.
      BuildMI(MBB, XorMI, DL, TII->get(X86::SBB32rr_REV), FinalDst)
          .addReg(FinalDst, RegState::Undef)
          .addReg(FinalDst, RegState::Undef);

      if (CC == X86::COND_E) {
        // == 0: INC turns -1->0 (was nonzero), 0->1 (was zero)
        BuildMI(MBB, XorMI, DL, TII->get(X86::INC32r), FinalDst)
            .addReg(FinalDst);
      } else {
        // != 0: NEG turns -1->1 (was nonzero), 0->0 (was zero)
        BuildMI(MBB, XorMI, DL, TII->get(X86::NEG32r), FinalDst)
            .addReg(FinalDst);
      }

      // Erase original instructions.
      auto NextI = MovToErase ? std::next(MachineBasicBlock::iterator(MovToErase))
                              : std::next(SetccI);
      if (MovToErase)
        MovToErase->eraseFromParent();
      SetccI->eraseFromParent();
      TestI->eraseFromParent();
      XorMI.eraseFromParent();
      I = NextI;
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferNegSbbPass() {
  return new X86PreferNegSbbPass();
}

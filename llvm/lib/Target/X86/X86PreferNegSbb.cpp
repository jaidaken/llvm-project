//===---- X86PreferNegSbb.cpp - Boolean NOT via neg+sbb+inc ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: This pass converts the LLVM boolean NOT pattern
// (xor reg, reg; test src, src; sete reg) into the MSVC 6.0 pattern
// (neg reg; sbb reg, reg; inc reg) for functions with PreferNegSbb.
//
// MSVC 6.0 generates "neg eax; sbb eax, eax; inc eax" for "return !val;"
// which computes: if val==0 then CF=0, sbb→0, inc→1; if val!=0 then CF=1,
// sbb→-1, inc→0. This is equivalent to "val == 0" as a 32-bit result.
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
    // Look for: XOR32rr/XOR32rr_REV Dst, Dst (zero)
    //           TEST32rr Src, Src
    //           SETEr Dst_low8
    // Replace:  NEG32r Src
    //           SBB32rr Dst, Dst
    //           INC32r Dst (or ADD32ri8 Dst, 1)
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

      // Step 2: Find TEST32rr Src, Src immediately after.
      auto TestI = std::next(I);
      if (TestI == E || TestI->getOpcode() != X86::TEST32rr) {
        ++I;
        continue;
      }

      Register TestSrc1 = TestI->getOperand(0).getReg();
      Register TestSrc2 = TestI->getOperand(1).getReg();
      if (TestSrc1 != TestSrc2) {
        ++I;
        continue;
      }
      Register SrcReg = TestSrc1;

      // Step 3: Find SETCCr with COND_E immediately after TEST.
      auto SeteI = std::next(TestI);
      if (SeteI == E || SeteI->getOpcode() != X86::SETCCr ||
          X86::getCondFromSETCC(*SeteI) != X86::COND_E) {
        ++I;
        continue;
      }

      DebugLoc DL = XorMI.getDebugLoc();

      // Build: neg reg; sbb dst, dst; inc dst
      // NEG32r sets CF=1 if Src!=0, CF=0 if Src==0.
      BuildMI(MBB, XorMI, DL, TII->get(X86::NEG32r), SrcReg)
          .addReg(SrcReg);

      // SBB32rr Dst, Dst: if CF=1 then Dst=-1, if CF=0 then Dst=0.
      BuildMI(MBB, XorMI, DL, TII->get(X86::SBB32rr), XorDst)
          .addReg(XorDst, RegState::Undef)
          .addReg(XorDst, RegState::Undef);

      // INC32r Dst: -1+1=0 (was nonzero), 0+1=1 (was zero). Boolean NOT.
      BuildMI(MBB, XorMI, DL, TII->get(X86::INC32r), XorDst)
          .addReg(XorDst);

      // Erase original XOR, TEST, SETE.
      auto NextI = std::next(SeteI);
      SeteI->eraseFromParent();
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

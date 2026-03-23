//===--- X86Prefer8BitOps.cpp - 8-bit sub+neg for boolean compare ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: MSVC 6.0 sometimes uses 8-bit sub+neg to compute a boolean
// equality result instead of a 32-bit cmp+setcc.
//
// Pattern: XOR32rr Dst,Dst; CMP32ri8 EAX,N; SETCCr(COND_E) AL
//       -> SUB8ri AL,N; NEG8r AL; SBB32rr_REV Dst,Dst; INC32r Dst
//
// Pattern: XOR32rr Dst,Dst; CMP32ri8 EAX,N; SETCCr(COND_NE) AL
//       -> SUB8ri AL,N; NEG8r AL; SBB32rr_REV Dst,Dst; NEG32r Dst
//
// Also handles CMP8ri (8-bit compare) and CMP32ri (32-bit immediate).
//
// The SUB+NEG trick: sub al,N sets ZF when al==N (result is 0). Then NEG
// propagates: neg 0 = 0 (CF=0), neg nonzero = nonzero (CF=1). The carry
// flag feeds into SBB for the final 0/1 boolean.
//
// This differs from PreferNegSbb's prefer_neg_sbb_8bit mode because this
// pass handles cases where the source register is EAX and the comparison
// target is a non-zero immediate, converting the full XOR+CMP+SETcc chain
// into an 8-bit SUB+NEG followed by SBB+INC/NEG.
//
// Gated on the "prefer_8bit_ops" string attribute.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-8bit-ops"
#define X86_PREFER_8BIT_OPS_NAME "X86 prefer 8-bit sub+neg ops pass"

namespace {
class X86Prefer8BitOpsPass : public MachineFunctionPass {
public:
  static char ID;
  X86Prefer8BitOpsPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return X86_PREFER_8BIT_OPS_NAME; }
};
} // end anonymous namespace

char X86Prefer8BitOpsPass::ID = 0;

/// Return true if this is a CMP instruction we can convert to 8-bit SUB.
static bool isSupportedCmp(unsigned Opc) {
  return Opc == X86::CMP32ri8 || Opc == X86::CMP32ri || Opc == X86::CMP8ri;
}

bool X86Prefer8BitOpsPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("prefer_8bit_ops"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Look for: XOR32rr Dst,Dst; CMP EAX,Imm; SETCCr(E/NE) AL
    // Replace:  SUB8ri AL,Imm; NEG8r AL; SBB32rr_REV Dst,Dst; INC32r/NEG32r Dst
    for (auto I = MBB.begin(), E = MBB.end(); I != E; /*below*/) {
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

      // Step 2: Find CMP32ri8/CMP32ri/CMP8ri with a positive immediate.
      auto CmpI = std::next(I);
      if (CmpI == E || !isSupportedCmp(CmpI->getOpcode())) {
        ++I;
        continue;
      }

      Register CmpReg = CmpI->getOperand(0).getReg();
      int64_t CmpImm = CmpI->getOperand(1).getImm();

      // Only handle positive immediates and EAX/AL source.
      if (CmpImm <= 0) {
        ++I;
        continue;
      }

      bool Is8BitCmp = (CmpI->getOpcode() == X86::CMP8ri);
      if (Is8BitCmp && CmpReg != X86::AL) {
        ++I;
        continue;
      }
      if (!Is8BitCmp && CmpReg != X86::EAX) {
        ++I;
        continue;
      }

      // Step 3: Find SETCCr with COND_E or COND_NE.
      auto SetccI = std::next(CmpI);
      if (SetccI == E || SetccI->getOpcode() != X86::SETCCr) {
        ++I;
        continue;
      }

      X86::CondCode CC = X86::getCondFromSETCC(*SetccI);
      if (CC != X86::COND_E && CC != X86::COND_NE) {
        ++I;
        continue;
      }

      // The SETCCr must write to the low 8 bits of XorDst (AL for EAX, etc).
      Register SetccDst = SetccI->getOperand(0).getReg();
      const X86RegisterInfo *TRI =
          static_cast<const X86RegisterInfo *>(STI.getRegisterInfo());
      if (!TRI->isSubRegisterEq(XorDst, SetccDst)) {
        ++I;
        continue;
      }

      DebugLoc DL = XorMI.getDebugLoc();

      // Build: SUB8ri AL, Imm
      BuildMI(MBB, XorMI, DL, TII->get(X86::SUB8ri), X86::AL)
          .addReg(X86::AL)
          .addImm(CmpImm);

      // Build: NEG8r AL
      BuildMI(MBB, XorMI, DL, TII->get(X86::NEG8r), X86::AL)
          .addReg(X86::AL);

      // Build: SBB32rr_REV Dst, Dst (CF=1 -> -1, CF=0 -> 0)
      BuildMI(MBB, XorMI, DL, TII->get(X86::SBB32rr_REV), XorDst)
          .addReg(XorDst, RegState::Undef)
          .addReg(XorDst, RegState::Undef);

      if (CC == X86::COND_E) {
        // == N: INC turns -1->0 (was not equal), 0->1 (was equal)
        BuildMI(MBB, XorMI, DL, TII->get(X86::INC32r), XorDst)
            .addReg(XorDst);
      } else {
        // != N: NEG turns -1->1 (was not equal), 0->0 (was equal)
        BuildMI(MBB, XorMI, DL, TII->get(X86::NEG32r), XorDst)
            .addReg(XorDst);
      }

      // Erase original: XOR, CMP, SETCCr
      auto NextI = std::next(SetccI);
      SetccI->eraseFromParent();
      CmpI->eraseFromParent();
      XorMI.eraseFromParent();
      I = NextI;
      Changed = true;
    }
  }

  // Second pass: NOT32r -> NOT8r when only the low byte matters.
  // Pattern: NOT32r EAX; AND32ri8 EAX, mask (where mask <= 0xFF)
  // MSVC 6.0 uses: NOT8r AL; AND32ri8 EAX, mask
  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E; ) {
      MachineInstr &NotMI = *I;

      if (NotMI.getOpcode() != X86::NOT32r) {
        ++I;
        continue;
      }

      Register NotReg = NotMI.getOperand(0).getReg();
      if (NotReg != X86::EAX && NotReg != X86::ECX && NotReg != X86::EDX) {
        ++I;
        continue;
      }

      // Check next instruction is AND32ri8 on the same register.
      auto AndI = std::next(I);
      if (AndI == E) {
        ++I;
        continue;
      }
      if (AndI->getOpcode() != X86::AND32ri8 &&
          AndI->getOpcode() != X86::AND32ri) {
        ++I;
        continue;
      }
      if (AndI->getOperand(0).getReg() != NotReg) {
        ++I;
        continue;
      }

      int64_t Mask = AndI->getOperand(2).getImm();
      if (Mask < 0 || Mask > 0xFF) {
        ++I;
        continue;
      }

      // Get the 8-bit sub-register.
      Register SubReg8;
      if (NotReg == X86::EAX) SubReg8 = X86::AL;
      else if (NotReg == X86::ECX) SubReg8 = X86::CL;
      else SubReg8 = X86::DL;

      // Replace NOT32r with NOT8r on the sub-register.
      DebugLoc DL = NotMI.getDebugLoc();
      BuildMI(MBB, NotMI, DL, TII->get(X86::NOT8r), SubReg8)
          .addReg(SubReg8);

      auto NextI = std::next(I);
      NotMI.eraseFromParent();
      I = NextI;
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86Prefer8BitOpsPass() {
  return new X86Prefer8BitOpsPass();
}

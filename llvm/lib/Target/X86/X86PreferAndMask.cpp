//===------- X86PreferAndMask.cpp - Convert MOVZX32rr16 to AND mask -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: This pass converts MOVZX32rr16 into MOV32rr + AND32ri 0xFFFF
// when the function has the PreferAndMask attribute. This matches MSVC 6.0's
// code generation pattern of "mov ecx, edi; and ecx, 0x0000ffff" instead of
// LLVM's "movzx ecx, di" for (x & 0xffff).
//
// For self-register cases (movzx ecx, cx), only AND32ri is emitted.
// For cross-register cases (movzx ebx, di), MOV32rr + AND32ri is emitted.
//
// Must run BEFORE X86ReversedOps and BEFORE X86ExpandMovzx.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-and-mask"
#define X86_PREFER_AND_MASK_NAME "X86 prefer AND mask pass"

namespace {
class X86PreferAndMaskPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferAndMaskPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return X86_PREFER_AND_MASK_NAME; }
};
} // end anonymous namespace

char X86PreferAndMaskPass::ID = 0;

bool X86PreferAndMaskPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::PreferAndMask))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    SmallVector<MachineInstr *, 8> ToReplace;
    {
      LivePhysRegs LiveRegs(*TRI);
      LiveRegs.addLiveOuts(MBB);

      for (auto I = MBB.rbegin(), E = MBB.rend(); I != E; ++I) {
        MachineInstr &MI = *I;

        if (MI.getOpcode() == X86::MOVZX32rr16) {
          // AND clobbers EFLAGS, MOVZX does not. Only replace if EFLAGS dead.
          if (!LiveRegs.contains(X86::EFLAGS)) {
            ToReplace.push_back(&MI);
          }
        }

        LiveRegs.stepBackward(MI);
      }
    }

    for (MachineInstr *MI : ToReplace) {
      Register DstReg = MI->getOperand(0).getReg();
      Register SrcReg16 = MI->getOperand(1).getReg();
      DebugLoc DL = MI->getDebugLoc();

      Register SrcReg32 =
          TRI->getMatchingSuperReg(SrcReg16, X86::sub_16bit,
                                   &X86::GR32RegClass);
      if (!SrcReg32)
        continue;

      if (SrcReg32 != DstReg) {
        // Cross-register: movzx ebx, di -> mov ebx, edi; and ebx, 0xffff
        BuildMI(MBB, *MI, DL, TII->get(X86::MOV32rr), DstReg)
            .addReg(SrcReg32);
      }

      // AND32ri to mask to 16 bits.
      BuildMI(MBB, *MI, DL, TII->get(X86::AND32ri), DstReg)
          .addReg(DstReg)
          .addImm(0xFFFF);

      MI->eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferAndMaskPass() {
  return new X86PreferAndMaskPass();
}

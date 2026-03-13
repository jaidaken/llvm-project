//===------- X86ExpandMovzx.cpp - Expand MOVZX to XOR+MOV ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: This pass expands MOVZX32rm8 into XOR32rr_REV + MOV8rm when the
// function has the ExpandMovzx attribute. This matches MSVC 6.0's code
// generation pattern of "xor eax, eax; mov al, [mem]" instead of LLVM's
// "movzx eax, byte ptr [mem]".
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-expand-movzx"
#define X86_EXPAND_MOVZX_NAME "X86 MOVZX expansion pass"

namespace {
class X86ExpandMovzxPass : public MachineFunctionPass {
public:
  static char ID;
  X86ExpandMovzxPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return X86_EXPAND_MOVZX_NAME; }
};
} // end anonymous namespace

char X86ExpandMovzxPass::ID = 0;

static bool isMovzxToExpand(unsigned Opcode) {
  return Opcode == X86::MOVZX32rm8 || Opcode == X86::MOVZX32rm16 ||
         Opcode == X86::MOVZX32rr8 || Opcode == X86::MOVZX32rr16;
}

static unsigned getMovOpcodeForMovzx(unsigned MovzxOpcode) {
  switch (MovzxOpcode) {
  case X86::MOVZX32rm8:  return X86::MOV8rm;
  case X86::MOVZX32rm16: return X86::MOV16rm;
  case X86::MOVZX32rr8:  return X86::MOV8rr;
  case X86::MOVZX32rr16: return X86::MOV16rr;
  default: llvm_unreachable("Unexpected MOVZX opcode");
  }
}

static unsigned getSubRegForMovzx(unsigned MovzxOpcode) {
  switch (MovzxOpcode) {
  case X86::MOVZX32rm8:
  case X86::MOVZX32rr8:
    return X86::sub_8bit;
  case X86::MOVZX32rm16:
  case X86::MOVZX32rr16:
    return X86::sub_16bit;
  default: llvm_unreachable("Unexpected MOVZX opcode");
  }
}

static bool isMemoryMovzx(unsigned Opcode) {
  return Opcode == X86::MOVZX32rm8 || Opcode == X86::MOVZX32rm16;
}

bool X86ExpandMovzxPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::ExpandMovzx))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Two-pass approach: first collect MOVZX instructions, then expand.
    // This avoids iterator invalidation issues.

    // Pass 1: Compute EFLAGS liveness at each MOVZX instruction.
    // We iterate backwards to track liveness.
    SmallVector<MachineInstr *, 8> ToExpand;
    {
      LivePhysRegs LiveRegs(*TRI);
      LiveRegs.addLiveOuts(MBB);

      for (auto I = MBB.rbegin(), E = MBB.rend(); I != E; ++I) {
        MachineInstr &MI = *I;

        if (isMovzxToExpand(MI.getOpcode())) {
          // Check if EFLAGS is live here. XOR will clobber it.
          if (!LiveRegs.contains(X86::EFLAGS)) {
            Register DstReg = MI.getOperand(0).getReg();
            unsigned SubRegIdx = getSubRegForMovzx(MI.getOpcode());
            Register DstSubReg = TRI->getSubReg(DstReg, SubRegIdx);
            if (DstSubReg) {
              // For memory operands, check that the destination register
              // is not used as a base or index register. XOR clears the
              // dest before MOV, so "xor ecx,ecx; mov cl,[ecx+N]" is
              // wrong when MOVZX was "movzx ecx, [ecx+N]".
              bool OverlapsMemOp = false;
              if (isMemoryMovzx(MI.getOpcode())) {
                for (unsigned i = 1; i < MI.getNumOperands(); ++i) {
                  const MachineOperand &MO = MI.getOperand(i);
                  if (MO.isReg() && MO.getReg() != 0 &&
                      TRI->regsOverlap(DstReg, MO.getReg())) {
                    OverlapsMemOp = true;
                    break;
                  }
                }
              }
              if (!OverlapsMemOp)
                ToExpand.push_back(&MI);
            }
          }
        }

        LiveRegs.stepBackward(MI);
      }
    }

    // Pass 2: Expand collected MOVZX instructions.
    for (MachineInstr *MI : ToExpand) {
      Register DstReg = MI->getOperand(0).getReg();
      unsigned MovzxOpcode = MI->getOpcode();
      unsigned MovOpcode = getMovOpcodeForMovzx(MovzxOpcode);
      unsigned SubRegIdx = getSubRegForMovzx(MovzxOpcode);
      Register DstSubReg = TRI->getSubReg(DstReg, SubRegIdx);
      DebugLoc DL = MI->getDebugLoc();

      // Insert XOR32rr_REV to clear the full 32-bit register.
      // Uses opcode 0x33 to match MSVC output.
      BuildMI(MBB, *MI, DL, TII->get(X86::XOR32rr_REV), DstReg)
          .addReg(DstReg, RegState::Undef)
          .addReg(DstReg, RegState::Undef);

      // Insert MOV to load the byte/word into the sub-register.
      if (isMemoryMovzx(MovzxOpcode)) {
        auto MIB = BuildMI(MBB, *MI, DL, TII->get(MovOpcode), DstSubReg);
        // Copy all memory operands from original MOVZX (operands 1..N).
        for (unsigned i = 1; i < MI->getNumOperands(); ++i)
          MIB.add(MI->getOperand(i));
        // Also copy memory references for alias analysis.
        MIB.cloneMemRefs(*MI);
      } else {
        // Register-to-register: MOVZX32rr8 / MOVZX32rr16.
        Register SrcReg = MI->getOperand(1).getReg();
        BuildMI(MBB, *MI, DL, TII->get(MovOpcode), DstSubReg)
            .addReg(SrcReg);
      }

      MI->eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86ExpandMovzxPass() {
  return new X86ExpandMovzxPass();
}

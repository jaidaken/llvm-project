//===--- X86FixupMovzxOverlap.cpp - Fix MOVZX dest/base overlap -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: When the register allocator assigns the MOVZX destination to
// the same register as the memory base (e.g., movzx ecx, byte [ecx+N]),
// ExpandMovzx cannot expand it because xor ecx,ecx would clobber the
// address. This pass rewrites the MOVZX destination to EAX and propagates
// the change through all subsequent uses, then deletes the now-redundant
// XOR EAX,EAX and MOV EAX,ECX instructions.
//
// Must run BEFORE X86ExpandMovzx (which then sees no overlap and expands).
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-fixup-movzx-overlap"

namespace {
class X86FixupMovzxOverlapPass : public MachineFunctionPass {
public:
  static char ID;
  X86FixupMovzxOverlapPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 fixup MOVZX dest/base overlap";
  }
};
} // end anonymous namespace

char X86FixupMovzxOverlapPass::ID = 0;

/// Map a 32-bit register to its 8-bit sub-register.
static Register get8BitSubReg(Register Reg32) {
  switch (Reg32) {
  case X86::EAX: return X86::AL;
  case X86::ECX: return X86::CL;
  case X86::EDX: return X86::DL;
  case X86::EBX: return X86::BL;
  default: return X86::NoRegister;
  }
}

/// Map a 32-bit register to its 16-bit sub-register.
static Register get16BitSubReg(Register Reg32) {
  switch (Reg32) {
  case X86::EAX: return X86::AX;
  case X86::ECX: return X86::CX;
  case X86::EDX: return X86::DX;
  case X86::EBX: return X86::BX;
  default: return X86::NoRegister;
  }
}

/// Replace OldReg (and its sub-registers) with NewReg equivalents in MI.
static void replaceRegWithSubs(MachineInstr &MI, Register OldReg32,
                                Register NewReg32) {
  Register Old8 = get8BitSubReg(OldReg32);
  Register New8 = get8BitSubReg(NewReg32);
  Register Old16 = get16BitSubReg(OldReg32);
  Register New16 = get16BitSubReg(NewReg32);

  for (MachineOperand &MO : MI.operands()) {
    if (!MO.isReg())
      continue;
    Register R = MO.getReg();
    if (R == OldReg32)
      MO.setReg(NewReg32);
    else if (R == Old8)
      MO.setReg(New8);
    else if (R == Old16)
      MO.setReg(New16);
  }
}

bool X86FixupMovzxOverlapPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::ExpandMovzx))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E; ++I) {
      MachineInstr &MovzxMI = *I;

      // Only handle MOVZX32rm8 and MOVZX32rm16 (memory forms).
      if (MovzxMI.getOpcode() != X86::MOVZX32rm8 &&
          MovzxMI.getOpcode() != X86::MOVZX32rm16)
        continue;

      Register DstReg = MovzxMI.getOperand(0).getReg();

      // Check if dest overlaps with any memory operand register.
      bool HasOverlap = false;
      for (unsigned i = 1; i < MovzxMI.getNumOperands(); ++i) {
        const MachineOperand &MO = MovzxMI.getOperand(i);
        if (MO.isReg() && MO.getReg() != X86::NoRegister &&
            TRI->regsOverlap(DstReg, MO.getReg())) {
          HasOverlap = true;
          break;
        }
      }
      if (!HasOverlap)
        continue;

      // Can't fix if dest is already EAX (nothing to swap to).
      if (DstReg == X86::EAX)
        continue;

      // Verify EAX is not used in the memory operand.
      bool EAXInMemOp = false;
      for (unsigned i = 1; i < MovzxMI.getNumOperands(); ++i) {
        const MachineOperand &MO = MovzxMI.getOperand(i);
        if (MO.isReg() && TRI->regsOverlap(MO.getReg(), X86::EAX)) {
          EAXInMemOp = true;
          break;
        }
      }
      if (EAXInMemOp)
        continue;

      // Change MOVZX destination from DstReg to EAX.
      Register OldReg = DstReg;
      MovzxMI.getOperand(0).setReg(X86::EAX);

      // Walk forward through remaining instructions, rewriting OldReg->EAX.
      SmallVector<MachineInstr*, 4> ToErase;
      auto NextI = std::next(I);
      for (auto J = NextI; J != E; ++J) {
        MachineInstr &MI = *J;
        if (MI.isPseudo())
          continue;

        // Check for XOR EAX,EAX (dead zero from original sete path).
        // Delete it - ExpandMovzx will insert its own XOR.
        unsigned Opc = MI.getOpcode();
        if ((Opc == X86::XOR32rr || Opc == X86::XOR32rr_REV) &&
            MI.getOperand(0).getReg() == X86::EAX &&
            MI.getOperand(1).getReg() == X86::EAX &&
            MI.getOperand(2).getReg() == X86::EAX) {
          ToErase.push_back(&MI);
          continue;
        }

        // Rewrite OldReg (and sub-regs) to EAX (and sub-regs).
        replaceRegWithSubs(MI, OldReg, X86::EAX);

        // Check for self-move MOV32rr EAX, EAX (was MOV EAX, OldReg).
        if ((Opc == X86::MOV32rr || Opc == X86::MOV32rr_REV) &&
            MI.getOperand(0).getReg() == X86::EAX &&
            MI.getOperand(1).getReg() == X86::EAX) {
          ToErase.push_back(&MI);
        }
      }

      for (MachineInstr *MI : ToErase)
        MI->eraseFromParent();

      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86FixupMovzxOverlapPass() {
  return new X86FixupMovzxOverlapPass();
}

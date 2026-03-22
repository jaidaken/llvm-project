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

/// Check if a MOV32rm has dest overlapping with base register.
static bool hasMemOverlap(MachineInstr &MI, const TargetRegisterInfo *TRI) {
  Register DstReg = MI.getOperand(0).getReg();
  for (unsigned i = 1; i < MI.getNumOperands(); ++i) {
    const MachineOperand &MO = MI.getOperand(i);
    if (MO.isReg() && MO.getReg() != X86::NoRegister &&
        TRI->regsOverlap(DstReg, MO.getReg()))
      return true;
  }
  return false;
}

/// Check if a register is used in the memory operands of an instruction.
static bool regInMemOps(MachineInstr &MI, Register Reg,
                        const TargetRegisterInfo *TRI) {
  for (unsigned i = 1; i < MI.getNumOperands(); ++i) {
    const MachineOperand &MO = MI.getOperand(i);
    if (MO.isReg() && TRI->regsOverlap(MO.getReg(), Reg))
      return true;
  }
  return false;
}

bool X86FixupMovzxOverlapPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::ExpandMovzx) &&
      !MF.getFunction().hasFnAttribute(Attribute::NoCalleeSaves))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E; ++I) {
      MachineInstr &LoadMI = *I;
      unsigned Opc = LoadMI.getOpcode();

      // === Pattern 1: MOVZX32rm8/16 with dest==base overlap ===
      // Fix: change dest to EAX (for expand_movzx to work later).
      // Gate: ExpandMovzx attribute.
      if ((Opc == X86::MOVZX32rm8 || Opc == X86::MOVZX32rm16) &&
          MF.getFunction().hasFnAttribute(Attribute::ExpandMovzx)) {
        Register DstReg = LoadMI.getOperand(0).getReg();
        if (!hasMemOverlap(LoadMI, TRI) || DstReg == X86::EAX ||
            regInMemOps(LoadMI, X86::EAX, TRI))
          continue;

        // Change MOVZX destination from DstReg to EAX.
        Register OldReg = DstReg;
        LoadMI.getOperand(0).setReg(X86::EAX);

        // Walk forward, rewriting OldReg->EAX. Delete dead XOR/self-MOV.
        SmallVector<MachineInstr*, 4> ToErase;
        for (auto J = std::next(I); J != E; ++J) {
          if (J->isPseudo()) continue;
          unsigned JOpc = J->getOpcode();
          // Delete XOR EAX,EAX (ExpandMovzx will insert its own).
          if ((JOpc == X86::XOR32rr || JOpc == X86::XOR32rr_REV) &&
              J->getOperand(0).getReg() == X86::EAX &&
              J->getOperand(1).getReg() == X86::EAX &&
              J->getOperand(2).getReg() == X86::EAX) {
            ToErase.push_back(&*J);
            continue;
          }
          replaceRegWithSubs(*J, OldReg, X86::EAX);
          // Delete self-move MOV EAX, EAX.
          if ((JOpc == X86::MOV32rr || JOpc == X86::MOV32rr_REV) &&
              J->getOperand(0).getReg() == X86::EAX &&
              J->getOperand(1).getReg() == X86::EAX)
            ToErase.push_back(&*J);
        }
        for (MachineInstr *MI : ToErase) MI->eraseFromParent();
        Changed = true;
        continue;
      }

      // === Pattern 2: MOV32rm with dest==base overlap ===
      // Fix: change dest to EDX (EAX is used for zero/result).
      // Used for: mov ecx,[ecx+N]; xor eax; test ecx; setne al
      // Target:   mov edx,[ecx+N]; xor eax; test edx; setne al
      // Gate: NoCalleeSaves attribute (these are simple accessor functions).
      if (Opc == X86::MOV32rm &&
          MF.getFunction().hasFnAttribute(Attribute::NoCalleeSaves)) {
        Register DstReg = LoadMI.getOperand(0).getReg();
        if (!hasMemOverlap(LoadMI, TRI) || DstReg == X86::EDX ||
            regInMemOps(LoadMI, X86::EDX, TRI))
          continue;

        // Verify this is followed by XOR+TEST+SETCCr pattern
        // (don't change registers for arbitrary MOV32rm overlaps).
        auto NextIt = std::next(I);
        while (NextIt != E && NextIt->isPseudo()) ++NextIt;
        if (NextIt == E) continue;
        unsigned NextOpc = NextIt->getOpcode();
        bool IsXorEax = (NextOpc == X86::XOR32rr || NextOpc == X86::XOR32rr_REV) &&
                         NextIt->getOperand(0).getReg() == X86::EAX;
        if (!IsXorEax) continue;

        // Change MOV32rm destination from DstReg to EDX.
        Register OldReg = DstReg;
        LoadMI.getOperand(0).setReg(X86::EDX);

        // Walk forward, rewriting OldReg->EDX (but NOT EAX references).
        for (auto J = std::next(I); J != E; ++J) {
          if (J->isPseudo()) continue;
          replaceRegWithSubs(*J, OldReg, X86::EDX);
        }
        Changed = true;
        continue;
      }
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86FixupMovzxOverlapPass() {
  return new X86FixupMovzxOverlapPass();
}

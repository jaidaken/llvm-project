//===--- X86UnfoldCmpMem.cpp - Unfold CMP [mem],imm to MOV+CMP -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: MSVC 6.0 never generates CMP [mem], imm directly. Instead it
// loads the value into a register first, then compares:
//
//   mov dl, byte ptr [ecx + 0x8c]    ; load into DL
//   xor eax, eax                     ; zero EAX for sete result
//   cmp dl, 0x10                     ; compare register
//   sete al                          ; set result
//
// Clang generates the folded form:
//
//   xor eax, eax                     ; zero EAX
//   cmp byte ptr [ecx + 0x8c], 0x10  ; compare memory directly
//   sete al                          ; set result
//
// This pass detects CMP8mi/CMP32mi followed by SETCCr and unfolds to
// MOV + XOR + CMP + SETCCr, matching MSVC 6.0's pattern.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "x86-unfold-cmp-mem"

namespace {
class X86UnfoldCmpMemPass : public MachineFunctionPass {
public:
  static char ID;
  X86UnfoldCmpMemPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 unfold CMP [mem], imm to MOV+CMP";
  }
};
} // end anonymous namespace

char X86UnfoldCmpMemPass::ID = 0;

/// Get the register-immediate CMP opcode for a memory-immediate CMP.
static unsigned getCmpRiForCmpMi(unsigned Opc) {
  switch (Opc) {
  case X86::CMP8mi:   return X86::CMP8ri;
  case X86::CMP32mi:  return X86::CMP32ri;
  case X86::CMP32mi8: return X86::CMP32ri8;
  case X86::CMP16mi:  return X86::CMP16ri;
  case X86::CMP16mi8: return X86::CMP16ri8;
  default: return 0;
  }
}

/// Get the MOV-from-memory opcode for a CMP memory size.
static unsigned getMovRmForCmpMi(unsigned Opc) {
  switch (Opc) {
  case X86::CMP8mi:   return X86::MOV8rm;
  case X86::CMP32mi:  return X86::MOV32rm;
  case X86::CMP32mi8: return X86::MOV32rm;
  case X86::CMP16mi:  return X86::MOV16rm;
  case X86::CMP16mi8: return X86::MOV16rm;
  default: return 0;
  }
}

/// Get the default scratch register for the load.
/// MSVC 6.0 uses EDX/DL for byte loads in accessor patterns.
static Register getScratchReg(unsigned CmpOpc) {
  switch (CmpOpc) {
  case X86::CMP8mi:   return X86::DL;
  case X86::CMP32mi:
  case X86::CMP32mi8: return X86::EDX;
  case X86::CMP16mi:
  case X86::CMP16mi8: return X86::DX;
  default: return X86::EDX;
  }
}

/// Get the 32-bit super-register for XOR zeroing.
static Register getXorReg(Register ScratchReg) {
  if (ScratchReg == X86::DL || ScratchReg == X86::DX || ScratchReg == X86::EDX)
    return X86::EDX;
  if (ScratchReg == X86::AL || ScratchReg == X86::AX || ScratchReg == X86::EAX)
    return X86::EAX;
  if (ScratchReg == X86::CL || ScratchReg == X86::CX || ScratchReg == X86::ECX)
    return X86::ECX;
  return X86::EDX;
}

bool X86UnfoldCmpMemPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("unfold_cmp_mem"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E; ) {
      MachineInstr &CmpMI = *I;

      unsigned CmpOpc = CmpMI.getOpcode();
      unsigned CmpRiOpc = getCmpRiForCmpMi(CmpOpc);
      if (!CmpRiOpc) {
        ++I;
        continue;
      }

      // CMP8mi/CMP32mi operand layout: [base, scale, index, disp, segment, imm]
      // Memory operand is operands 0-4, immediate is operand 5.
      if (CmpMI.getNumOperands() < 6) {
        ++I;
        continue;
      }

      // Check if the next non-debug instruction is SETCCr into AL.
      auto Next = std::next(I);
      while (Next != E && Next->isDebugInstr())
        ++Next;

      bool HasSetcc = false;
      Register SetccReg;
      if (Next != E && Next->getOpcode() == X86::SETCCr) {
        HasSetcc = true;
        SetccReg = Next->getOperand(0).getReg();
      }

      // Also check: is there an XOR zeroing EAX before this CMP?
      // Pattern: XOR32rr EAX,EAX; CMP8mi [mem],imm; SETCCr AL
      MachineInstr *PrevXor = nullptr;
      {
        auto PrevIt = I;
        if (PrevIt != MBB.begin()) {
          --PrevIt;
          while (PrevIt != MBB.begin() && PrevIt->isDebugInstr())
            --PrevIt;
          MachineInstr &Prev = *PrevIt;
          if ((Prev.getOpcode() == X86::XOR32rr ||
               Prev.getOpcode() == X86::XOR32rr_REV) &&
              Prev.getOperand(0).getReg() == X86::EAX &&
              Prev.getOperand(1).getReg() == X86::EAX) {
            PrevXor = &Prev;
          }
        }
      }

      // We need the SETCCr to know this is the bool accessor pattern.
      // Without it, we don't know if MSVC would unfold this CMP.
      if (!HasSetcc) {
        ++I;
        continue;
      }

      DebugLoc DL = CmpMI.getDebugLoc();
      unsigned MovOpc = getMovRmForCmpMi(CmpOpc);
      Register ScratchReg = getScratchReg(CmpOpc);
      Register XorFullReg = getXorReg(ScratchReg);

      // Get the immediate value from the CMP.
      int64_t ImmVal = CmpMI.getOperand(5).getImm();

      // Build the MSVC 6.0 sequence:
      // 1. MOV scratch, [mem]  (load the value)
      // 2. XOR EAX, EAX        (zero for sete - moved here from before CMP)
      // 3. CMP scratch, imm    (register compare)
      // (SETCCr AL stays as is)

      // 1. MOV scratch, [mem]
      auto MIB = BuildMI(MBB, CmpMI, DL, TII->get(MovOpc), ScratchReg);
      for (unsigned i = 0; i < 5; ++i)
        MIB.add(CmpMI.getOperand(i));
      MIB.cloneMemRefs(CmpMI);

      // 2. XOR EAX, EAX (if there was one before the CMP, move it here)
      if (PrevXor) {
        // Rebuild the XOR at this position (between MOV and CMP)
        BuildMI(MBB, CmpMI, DL, TII->get(PrevXor->getOpcode()), X86::EAX)
            .addReg(X86::EAX, RegState::Undef)
            .addReg(X86::EAX, RegState::Undef);
        PrevXor->eraseFromParent();
      }

      // 3. CMP scratch, imm (or TEST scratch, scratch when imm == 0)
      if (ImmVal == 0 && CmpOpc == X86::CMP8mi) {
        // cmp dl, 0 -> test dl, dl (shorter encoding: 84 D2 vs 80 FA 00)
        BuildMI(MBB, CmpMI, DL, TII->get(X86::TEST8rr))
            .addReg(ScratchReg)
            .addReg(ScratchReg);
      } else if (ImmVal == 0 && (CmpOpc == X86::CMP32mi || CmpOpc == X86::CMP32mi8)) {
        BuildMI(MBB, CmpMI, DL, TII->get(X86::TEST32rr))
            .addReg(ScratchReg)
            .addReg(ScratchReg);
      } else {
        BuildMI(MBB, CmpMI, DL, TII->get(CmpRiOpc))
            .addReg(ScratchReg)
            .addImm(ImmVal);
      }

      // Remove original CMP [mem], imm.
      auto NextI = std::next(I);
      CmpMI.eraseFromParent();
      I = NextI;
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86UnfoldCmpMemPass() {
  return new X86UnfoldCmpMemPass();
}

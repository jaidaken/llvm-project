//===--- X86UnfoldCmpMem.cpp - Unfold CMP [mem],imm to MOV+CMP -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: MSVC 6.0 often loads a value into a register before comparing,
// rather than comparing memory directly. This pass unfolds CMP [mem],imm
// into MOV reg,[mem] + CMP/TEST reg patterns.
//
// Two modes (controlled by separate attributes):
//
// 1. unfold_cmp_mem: CMP+SETcc pattern (bool accessors)
//    mov dl,[mem]; xor eax,eax; cmp dl,imm; sete al
//
// 2. prefer_mov_test: CMP+Jcc pattern (branch conditions)
//    mov eax,[mem]; test eax,eax; jne label  (when imm==0)
//    mov eax,[mem]; cmp eax,imm; jne label   (when imm!=0)
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/LivePhysRegs.h"
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

/// Get the TEST opcode for a given CMP size.
static unsigned getTestRrForCmpMi(unsigned Opc) {
  switch (Opc) {
  case X86::CMP8mi:   return X86::TEST8rr;
  case X86::CMP32mi:
  case X86::CMP32mi8: return X86::TEST32rr;
  case X86::CMP16mi:
  case X86::CMP16mi8: return X86::TEST16rr;
  default: return X86::TEST32rr;
  }
}

/// For the SETcc variant, MSVC uses EDX/DL as scratch (EAX holds sete result).
static Register getScratchRegForSetcc(unsigned CmpOpc) {
  switch (CmpOpc) {
  case X86::CMP8mi:   return X86::DL;
  case X86::CMP32mi:
  case X86::CMP32mi8: return X86::EDX;
  case X86::CMP16mi:
  case X86::CMP16mi8: return X86::DX;
  default: return X86::EDX;
  }
}

/// For the Jcc variant, MSVC uses EAX as scratch (no sete needs it).
static Register getScratchRegForJcc(unsigned CmpOpc) {
  switch (CmpOpc) {
  case X86::CMP8mi:   return X86::AL;
  case X86::CMP32mi:
  case X86::CMP32mi8: return X86::EAX;
  case X86::CMP16mi:
  case X86::CMP16mi8: return X86::AX;
  default: return X86::EAX;
  }
}

/// Check if an instruction is a conditional branch.
static bool isJcc(const MachineInstr &MI) {
  return MI.getOpcode() == X86::JCC_1 || MI.getOpcode() == X86::JCC_4;
}

bool X86UnfoldCmpMemPass::runOnMachineFunction(MachineFunction &MF) {
  bool EnableSetcc = MF.getFunction().hasFnAttribute("unfold_cmp_mem");
  bool EnableJcc = MF.getFunction().hasFnAttribute("prefer_mov_test");

  if (!EnableSetcc && !EnableJcc)
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

      if (CmpMI.getNumOperands() < 6) {
        ++I;
        continue;
      }

      // Find the next non-debug instruction after the CMP.
      auto Next = std::next(I);
      while (Next != E && Next->isDebugInstr())
        ++Next;
      if (Next == E) {
        ++I;
        continue;
      }

      // Determine which variant we're handling.
      bool IsSETccVariant = EnableSetcc && Next->getOpcode() == X86::SETCCr;
      bool IsJccVariant = EnableJcc && isJcc(*Next);

      if (!IsSETccVariant && !IsJccVariant) {
        ++I;
        continue;
      }

      // Look for preceding XOR zeroing EAX (only relevant for SETcc variant).
      MachineInstr *PrevXor = nullptr;
      if (IsSETccVariant) {
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

      DebugLoc DL = CmpMI.getDebugLoc();
      unsigned MovOpc = getMovRmForCmpMi(CmpOpc);
      int64_t ImmVal = CmpMI.getOperand(5).getImm();

      // Choose scratch register based on variant.
      Register ScratchReg = IsSETccVariant
          ? getScratchRegForSetcc(CmpOpc)
          : getScratchRegForJcc(CmpOpc);

      // === Build the replacement sequence ===

      // 1. MOV scratch, [mem]
      auto MIB = BuildMI(MBB, CmpMI, DL, TII->get(MovOpc), ScratchReg);
      for (unsigned i = 0; i < 5; ++i)
        MIB.add(CmpMI.getOperand(i));
      MIB.cloneMemRefs(CmpMI);

      // 2. For SETcc variant: move XOR EAX,EAX between MOV and CMP.
      if (IsSETccVariant && PrevXor) {
        BuildMI(MBB, CmpMI, DL, TII->get(PrevXor->getOpcode()), X86::EAX)
            .addReg(X86::EAX, RegState::Undef)
            .addReg(X86::EAX, RegState::Undef);
        PrevXor->eraseFromParent();
      }

      // 3. CMP/TEST
      if (ImmVal == 0) {
        // cmp X, 0 -> test X, X (shorter encoding)
        unsigned TestOpc = getTestRrForCmpMi(CmpOpc);
        BuildMI(MBB, CmpMI, DL, TII->get(TestOpc))
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

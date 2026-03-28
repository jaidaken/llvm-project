//===--- X86PreferFirstLoadEax.cpp - Force first load into EAX ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: Post-regalloc pass that forces the first memory load in the
// entry block into EAX.
//
// MSVC 6.0 allocates EAX for the first memory load in many functions, while
// Clang's register allocator often picks EDX or another register. This matters
// because:
//   - test ah, 0x80 (EAX high byte) produces F6 C4 80 (3 bytes)
//   - test dh, 0x80 (EDX high byte) produces F6 C6 80 (different ModR/M)
//   - test eax, 0x8000 uses special 1-byte opcode A9 (5 bytes total)
//   - test edx, 0x8000 uses 2-byte opcode F7 C2 (6 bytes total)
//
// The pass finds the first MOV32rm in the entry block. If its destination is
// not EAX, it swaps that register with EAX throughout the entire function.
//
// Gated on the "prefer_first_load_eax" function attribute.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-first-load-eax"
#define PASS_NAME "X86 prefer first load into EAX"

namespace {
class X86PreferFirstLoadEaxPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferFirstLoadEaxPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return PASS_NAME; }
};
} // end anonymous namespace

char X86PreferFirstLoadEaxPass::ID = 0;

/// Swap register Reg between the OldReg32 family and the EAX family.
/// Handles 32-bit, 16-bit, and 8-bit sub-register variants.
/// Returns Reg unchanged if it belongs to neither family.
static unsigned swapReg(unsigned Reg, unsigned OldReg32, unsigned NewReg32) {
  if (Reg == NewReg32) return OldReg32;
  if (Reg == OldReg32) return NewReg32;

  // Sub-register mappings for EAX
  unsigned EAX_16 = X86::AX, EAX_Lo = X86::AL, EAX_Hi = X86::AH;

  // Sub-register mappings for the other register
  unsigned Other_16 = 0, Other_Lo = 0, Other_Hi = 0;
  switch (OldReg32) {
  case X86::EDX: Other_16 = X86::DX; Other_Lo = X86::DL; Other_Hi = X86::DH; break;
  case X86::ECX: Other_16 = X86::CX; Other_Lo = X86::CL; Other_Hi = X86::CH; break;
  case X86::EBX: Other_16 = X86::BX; Other_Lo = X86::BL; Other_Hi = X86::BH; break;
  case X86::ESI: Other_16 = X86::SI; Other_Lo = X86::SIL; Other_Hi = 0; break;
  case X86::EDI: Other_16 = X86::DI; Other_Lo = X86::DIL; Other_Hi = 0; break;
  case X86::EBP: Other_16 = X86::BP; Other_Lo = X86::BPL; Other_Hi = 0; break;
  default: return Reg;
  }

  // EAX sub-regs -> Other sub-regs
  if (Reg == EAX_16) return Other_16;
  if (Reg == EAX_Lo) return Other_Lo;
  if (Reg == EAX_Hi && Other_Hi) return Other_Hi;

  // Other sub-regs -> EAX sub-regs
  if (Reg == Other_16) return EAX_16;
  if (Reg == Other_Lo) return EAX_Lo;
  if (Other_Hi && Reg == Other_Hi) return EAX_Hi;

  return Reg;
}

bool X86PreferFirstLoadEaxPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("prefer_first_load_eax"))
    return false;

  MachineBasicBlock &EntryMBB = MF.front();

  // Find the first MOV32rm in the entry block (skip pseudos, pushes, etc.)
  MachineInstr *FirstLoad = nullptr;
  for (MachineInstr &MI : EntryMBB) {
    if (MI.isPseudo())
      continue;
    if (MI.getOpcode() == X86::MOV32rm) {
      FirstLoad = &MI;
      break;
    }
  }

  if (!FirstLoad)
    return false;

  Register OldReg = FirstLoad->getOperand(0).getReg();
  if (OldReg == X86::EAX)
    return false; // Already in EAX, nothing to do.

  // Swap OldReg <-> EAX throughout the entire function.
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      for (MachineOperand &MO : MI.operands()) {
        if (!MO.isReg())
          continue;
        unsigned NewReg = swapReg(MO.getReg(), OldReg, X86::EAX);
        if (NewReg != MO.getReg()) {
          MO.setReg(NewReg);
          Changed = true;
        }
      }
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferFirstLoadEaxPass() {
  return new X86PreferFirstLoadEaxPass();
}

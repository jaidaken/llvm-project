//===--- X86SwapCmpRegisters.cpp - Fix bitfield accessor register alloc ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: MSVC 6.0 uses EDX for field loads and ECX for sete results
// in bitfield accessor functions. Clang assigns ECX for the field and EAX
// for sete. This pass detects the pattern and rewrites registers:
//
//   mov eax, [ecx+N]     ; ptr chain (no change)
//   mov ecx, [eax+M]     ; field -> ECX  =>  mov edx, [eax+M]
//   xor eax, eax         ; zero EAX      =>  xor ecx, ecx
//   cmp ecx, IMM         ; compare ECX   =>  cmp edx, IMM
//   sete al              ; result in AL  =>  sete cl
//   ret                  ;               =>  mov eax, ecx; ret
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-swap-cmp-registers"

namespace {
class X86SwapCmpRegistersPass : public MachineFunctionPass {
public:
  static char ID;
  X86SwapCmpRegistersPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 swap CMP registers for MSVC 6.0 bitfield accessors";
  }
};
} // end anonymous namespace

char X86SwapCmpRegistersPass::ID = 0;

/// Replace all occurrences of OldReg with NewReg in MI's operands.
static void replaceReg(MachineInstr &MI, Register OldReg, Register NewReg) {
  for (MachineOperand &MO : MI.operands()) {
    if (MO.isReg() && MO.getReg() == OldReg)
      MO.setReg(NewReg);
  }
}

bool X86SwapCmpRegistersPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::Msvc6RegAlloc))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Collect all non-pseudo instructions.
    SmallVector<MachineInstr*, 16> Instrs;
    for (MachineInstr &MI : MBB) {
      if (!MI.isPseudo() || MI.isReturn())
        Instrs.push_back(&MI);
    }

    // Pattern: look for SETCCr writing AL, preceded by CMP/TEST using ECX,
    // preceded by XOR EAX,EAX, preceded by MOV32rm into ECX.
    for (unsigned i = 0; i < Instrs.size(); ++i) {
      MachineInstr *SetccMI = Instrs[i];
      if (SetccMI->getOpcode() != X86::SETCCr)
        continue;
      if (SetccMI->getOperand(0).getReg() != X86::AL)
        continue;

      // Walk backward to find CMP/TEST using ECX.
      if (i < 1) continue;
      MachineInstr *CmpMI = Instrs[i - 1];
      bool IsCmp = (CmpMI->getOpcode() == X86::CMP32ri ||
                    CmpMI->getOpcode() == X86::CMP32ri8) &&
                   CmpMI->getOperand(0).getReg() == X86::ECX;
      bool IsTest = CmpMI->getOpcode() == X86::TEST32rr &&
                    CmpMI->getOperand(0).getReg() == X86::ECX &&
                    CmpMI->getOperand(1).getReg() == X86::ECX;
      if (!IsCmp && !IsTest)
        continue;

      // Walk backward to find XOR EAX, EAX.
      if (i < 2) continue;
      MachineInstr *XorMI = Instrs[i - 2];
      unsigned XorOpc = XorMI->getOpcode();
      if (XorOpc != X86::XOR32rr && XorOpc != X86::XOR32rr_REV)
        continue;
      if (XorMI->getOperand(0).getReg() != X86::EAX ||
          XorMI->getOperand(1).getReg() != X86::EAX ||
          XorMI->getOperand(2).getReg() != X86::EAX)
        continue;

      // Walk backward to find MOV32rm into ECX.
      if (i < 3) continue;
      MachineInstr *FieldMI = Instrs[i - 3];
      if (FieldMI->getOpcode() != X86::MOV32rm ||
          FieldMI->getOperand(0).getReg() != X86::ECX)
        continue;

      // Walk forward to find what follows SETCCr.
      MachineInstr *AfterSetcc = (i + 1 < Instrs.size()) ? Instrs[i + 1] : nullptr;

      // Pattern matched! Apply transformation.
      DebugLoc DL = SetccMI->getDebugLoc();

      // 1. Change field load: ECX -> EDX
      FieldMI->getOperand(0).setReg(X86::EDX);

      // 2. Change XOR: EAX,EAX -> ECX,ECX
      XorMI->getOperand(0).setReg(X86::ECX);
      XorMI->getOperand(1).setReg(X86::ECX);
      XorMI->getOperand(2).setReg(X86::ECX);

      // 3. Change CMP/TEST: ECX -> EDX
      replaceReg(*CmpMI, X86::ECX, X86::EDX);

      // 4. Change SETCCr: AL -> CL
      SetccMI->getOperand(0).setReg(X86::CL);

      // 5. Handle post-sete: if there's a MOVZX or nothing before RET
      // Use MOV32rr_REV (opcode 8B) to match MSVC 6.0 encoding.
      // The ReversedOps pass only reverses MOV when MOV32rr_REV attribute
      // is set, so we emit the reversed form directly.
      if (AfterSetcc && AfterSetcc->isReturn()) {
        BuildMI(MBB, *AfterSetcc, DL, TII->get(X86::MOV32rr_REV), X86::EAX)
            .addReg(X86::ECX);
      } else if (AfterSetcc &&
                 (AfterSetcc->getOpcode() == X86::MOVZX32rr8 ||
                  AfterSetcc->getOpcode() == X86::MOVZX32rr8_NOREX)) {
        BuildMI(MBB, *AfterSetcc, DL, TII->get(X86::MOV32rr_REV), X86::EAX)
            .addReg(X86::ECX);
        AfterSetcc->eraseFromParent();
      }

      Changed = true;
      break; // Only one pattern per basic block
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86SwapCmpRegistersPass() {
  return new X86SwapCmpRegistersPass();
}

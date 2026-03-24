//===--- X86Msvc6RegSwap.cpp - Fix bitfield accessor register allocation ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: Post-regalloc pass for ~24 Shape B bitfield accessor functions.
// MSVC 6.0 uses EDX for the field load and ECX for the sete result, but
// Clang uses ECX for the field load (overwriting dead `this`) and EAX for
// sete. This pass detects the pointer-chain + comparison pattern and rewrites
// registers to match MSVC 6.0 output.
//
// Gated on the msvc6_regswap function attribute.
//
// MSVC 6.0:  mov eax,[ecx+N]; mov edx,[eax+M]; xor ecx,ecx; test edx,edx;
//            sete cl; mov eax,ecx; ret
// Clang:     mov eax,[ecx+N]; mov ecx,[eax+M]; xor eax,eax; test ecx,ecx;
//            sete al; <movzx eax,al>; ret
//
// Transformation applied per basic block:
//   mov ecx,[eax+M]  -> mov edx,[eax+M]   (field load)
//   xor eax,eax      -> xor ecx,ecx       (zero register)
//   test ecx,ecx     -> test edx,edx      (comparison)
//   cmp ecx,imm      -> cmp edx,imm       (comparison variant)
//   sete al          -> sete cl            (result)
//   movzx eax,al     -> mov eax,ecx       (widen, or insert before ret)
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-msvc6-regswap"
#define X86_MSVC6_REGSWAP_NAME "X86 MSVC6 register swap for bitfield accessors"

namespace {
class X86Msvc6RegSwapPass : public MachineFunctionPass {
public:
  static char ID;
  X86Msvc6RegSwapPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return X86_MSVC6_REGSWAP_NAME; }
};
} // end anonymous namespace

char X86Msvc6RegSwapPass::ID = 0;

/// Replace all occurrences of OldReg with NewReg in MI's operands.
static void replaceReg(MachineInstr &MI, Register OldReg, Register NewReg) {
  for (MachineOperand &MO : MI.operands()) {
    if (MO.isReg() && MO.getReg() == OldReg)
      MO.setReg(NewReg);
  }
}

bool X86Msvc6RegSwapPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::Msvc6RegSwap))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Collect all non-pseudo instructions (plus returns).
    SmallVector<MachineInstr *, 16> Instrs;
    for (MachineInstr &MI : MBB) {
      if (!MI.isPseudo() || MI.isReturn())
        Instrs.push_back(&MI);
    }

    // Pattern: look for SETCCr writing AL, preceded by TEST/CMP using ECX,
    // preceded by XOR EAX,EAX, preceded by MOV32rm into ECX.
    for (unsigned i = 0; i < Instrs.size(); ++i) {
      MachineInstr *SetccMI = Instrs[i];
      if (SetccMI->getOpcode() != X86::SETCCr)
        continue;
      if (SetccMI->getOperand(0).getReg() != X86::AL)
        continue;

      // Walk backward to find TEST/CMP using ECX.
      if (i < 1)
        continue;
      MachineInstr *CmpMI = Instrs[i - 1];
      bool IsTest = CmpMI->getOpcode() == X86::TEST32rr &&
                    CmpMI->getOperand(0).getReg() == X86::ECX &&
                    CmpMI->getOperand(1).getReg() == X86::ECX;
      bool IsCmp = (CmpMI->getOpcode() == X86::CMP32ri ||
                    CmpMI->getOpcode() == X86::CMP32ri8) &&
                   CmpMI->getOperand(0).getReg() == X86::ECX;
      if (!IsTest && !IsCmp)
        continue;

      // Walk backward to find XOR EAX, EAX.
      if (i < 2)
        continue;
      MachineInstr *XorMI = Instrs[i - 2];
      unsigned XorOpc = XorMI->getOpcode();
      if (XorOpc != X86::XOR32rr && XorOpc != X86::XOR32rr_REV)
        continue;
      if (XorMI->getOperand(0).getReg() != X86::EAX ||
          XorMI->getOperand(1).getReg() != X86::EAX ||
          XorMI->getOperand(2).getReg() != X86::EAX)
        continue;

      // Walk backward to find MOV32rm into ECX (the field load).
      if (i < 3)
        continue;
      MachineInstr *FieldMI = Instrs[i - 3];
      if (FieldMI->getOpcode() != X86::MOV32rm ||
          FieldMI->getOperand(0).getReg() != X86::ECX)
        continue;

      // Walk backward to find the info load (MOV32rm into ECX from [ECX+N]).
      // This is the pointer chain root: mov ecx, [ecx+0x28]
      MachineInstr *InfoMI = nullptr;
      if (i >= 4) {
        MachineInstr *Cand = Instrs[i - 4];
        if (Cand->getOpcode() == X86::MOV32rm &&
            Cand->getOperand(0).getReg() == X86::ECX &&
            Cand->getOperand(1).isReg() &&
            Cand->getOperand(1).getReg() == X86::ECX)
          InfoMI = Cand;
      }

      // Look at what follows the SETCCr.
      MachineInstr *AfterSetcc =
          (i + 1 < Instrs.size()) ? Instrs[i + 1] : nullptr;

      // Pattern matched. Apply register swap.
      DebugLoc DL = SetccMI->getDebugLoc();

      // 0. Info load: ECX -> EAX (if present)
      // Changes: mov ecx,[ecx+N] -> mov eax,[ecx+N]
      if (InfoMI) {
        InfoMI->getOperand(0).setReg(X86::EAX);
        // Also update the field load's base from ECX to EAX
        // (since info ptr is now in EAX, not ECX)
        if (FieldMI->getOperand(1).isReg() &&
            FieldMI->getOperand(1).getReg() == X86::ECX)
          FieldMI->getOperand(1).setReg(X86::EAX);
      }

      // 1. Field load: ECX -> EDX
      FieldMI->getOperand(0).setReg(X86::EDX);

      // 2. XOR zero: EAX,EAX -> ECX,ECX
      XorMI->getOperand(0).setReg(X86::ECX);
      XorMI->getOperand(1).setReg(X86::ECX);
      XorMI->getOperand(2).setReg(X86::ECX);

      // 3. TEST/CMP: ECX -> EDX
      replaceReg(*CmpMI, X86::ECX, X86::EDX);

      // 4. SETCCr: AL -> CL
      SetccMI->getOperand(0).setReg(X86::CL);

      // 5. Handle the post-sete instruction.
      // Use MOV32rr_REV (opcode 0x8B) to match MSVC 6.0 encoding.
      if (AfterSetcc && AfterSetcc->isReturn()) {
        // Insert MOV EAX, ECX before RET.
        BuildMI(MBB, *AfterSetcc, DL, TII->get(X86::MOV32rr_REV), X86::EAX)
            .addReg(X86::ECX);
      } else if (AfterSetcc &&
                 (AfterSetcc->getOpcode() == X86::MOVZX32rr8 ||
                  AfterSetcc->getOpcode() == X86::MOVZX32rr8_NOREX)) {
        // Replace MOVZX with MOV EAX, ECX.
        BuildMI(MBB, *AfterSetcc, DL, TII->get(X86::MOV32rr_REV), X86::EAX)
            .addReg(X86::ECX);
        AfterSetcc->eraseFromParent();
      }

      Changed = true;
      break; // One pattern per basic block.
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86Msvc6RegSwapPass() {
  return new X86Msvc6RegSwapPass();
}

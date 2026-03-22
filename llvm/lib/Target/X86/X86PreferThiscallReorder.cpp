//===--- X86PreferThiscallReorder.cpp - Reorder this-ptr loads ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: When a no_callee_saves function loads a field from ECX (this)
// and then overwrites ECX with a stack param for a thiscall, the compiler
// inserts a redundant "mov eax, ecx" to save this first. MSVC 6.0 instead
// reads from ECX before overwriting it. This pass detects the pattern:
//
//   mov eax, ecx            ; save this
//   mov ecx, [esp+N]        ; load param for thiscall
//   mov eax, [eax+disp]     ; load field from saved this
//
// And rewrites to:
//
//   mov eax, [ecx+disp]     ; load field from this directly
//   mov ecx, [esp+N]        ; load param for thiscall
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-thiscall-reorder"

namespace {
class X86PreferThiscallReorderPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferThiscallReorderPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 prefer thiscall field load reorder";
  }
};
} // end anonymous namespace

char X86PreferThiscallReorderPass::ID = 0;

bool X86PreferThiscallReorderPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("no_tail_call") &&
      !MF.getFunction().hasFnAttribute(Attribute::NoCalleeSaves))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    auto I = MBB.begin();
    while (I != MBB.end()) {
      auto Next1 = std::next(I);
      if (Next1 == MBB.end()) break;
      auto Next2 = std::next(Next1);
      if (Next2 == MBB.end()) break;

      MachineInstr &MI0 = *I;
      MachineInstr &MI1 = *Next1;
      MachineInstr &MI2 = *Next2;

      // Pattern: MOV32rr EAX, ECX  (or MOV32rr_REV)
      //          MOV32rm ECX, [ESP+N]
      //          MOV32rm EAX, [EAX+disp]
      bool isMov0 = (MI0.getOpcode() == X86::MOV32rr ||
                      MI0.getOpcode() == X86::MOV32rr_REV) &&
                     MI0.getOperand(0).getReg() == X86::EAX &&
                     MI0.getOperand(1).getReg() == X86::ECX;

      if (!isMov0) { ++I; continue; }

      bool isLoad1 = MI1.getOpcode() == X86::MOV32rm &&
                      MI1.getOperand(0).getReg() == X86::ECX &&
                      MI1.getNumOperands() >= 6 &&
                      MI1.getOperand(1).isReg() &&
                      MI1.getOperand(1).getReg() == X86::ESP;

      if (!isLoad1) { ++I; continue; }

      bool isLoad2 = MI2.getOpcode() == X86::MOV32rm &&
                      MI2.getOperand(0).getReg() == X86::EAX &&
                      MI2.getNumOperands() >= 6 &&
                      MI2.getOperand(1).isReg() &&
                      MI2.getOperand(1).getReg() == X86::EAX;

      if (!isLoad2) { ++I; continue; }

      // Rewrite: change MI2's base from EAX to ECX, then delete MI0,
      // and swap MI1/MI2 order.
      DebugLoc DL = MI0.getDebugLoc();

      // Get MI2's displacement
      int64_t Disp2 = MI2.getOperand(4).isImm() ? MI2.getOperand(4).getImm() : 0;

      // Build: MOV32rm EAX, [ECX+disp]
      // Copy MI1's memory operand format but change base to ECX and disp
      MachineInstr *NewLoad = BuildMI(MBB, MI0, DL, TII->get(X86::MOV32rm), X86::EAX)
          .addReg(X86::ECX)      // base
          .addImm(MI2.getOperand(2).getImm()) // scale
          .addReg(MI2.getOperand(3).getReg()) // index
          .addImm(Disp2)         // disp
          .addReg(MI2.getOperand(5).getReg()); // segment

      // MI1 (mov ecx, [esp+N]) stays as is but moves after the new load
      // Remove MI0 (mov eax, ecx) and MI2 (mov eax, [eax+disp])
      MI0.eraseFromParent();
      MI2.eraseFromParent();
      // MI1 is now after NewLoad (correct order)

      Changed = true;
      I = std::next(MachineBasicBlock::iterator(*NewLoad));
      continue;

      // Pattern B: MOV32rr EAX, ECX; MOV32rm ECX, [ESP+N]; ADD32ri/ADD32ri8 EAX, imm; PUSH32r EAX
      // Rewrite to: ADD32ri/ADD32ri8 ECX, imm; PUSH32r ECX; MOV32rm ECX, [ESP+N+4]
      // This handles address computation (this + offset) for thiscall argument setup.
    PatternB:
      ;
    }

    // Second pass: Pattern B (needs 4 instructions)
    I = MBB.begin();
    while (I != MBB.end()) {
      auto N1 = std::next(I);
      if (N1 == MBB.end()) break;
      auto N2 = std::next(N1);
      if (N2 == MBB.end()) break;
      auto N3 = std::next(N2);
      if (N3 == MBB.end()) break;

      MachineInstr &MI0 = *I;   // mov eax, ecx
      MachineInstr &MI1 = *N1;  // mov ecx, [esp+N]
      MachineInstr &MI2 = *N2;  // add eax, imm
      MachineInstr &MI3 = *N3;  // push eax

      // Check: MOV32rr EAX, ECX
      bool isMov0 = (MI0.getOpcode() == X86::MOV32rr ||
                      MI0.getOpcode() == X86::MOV32rr_REV) &&
                     MI0.getOperand(0).getReg() == X86::EAX &&
                     MI0.getOperand(1).getReg() == X86::ECX;
      if (!isMov0) { ++I; continue; }

      // Check: MOV32rm ECX, [ESP+N]
      bool isLoad1 = MI1.getOpcode() == X86::MOV32rm &&
                      MI1.getOperand(0).getReg() == X86::ECX &&
                      MI1.getOperand(1).isReg() &&
                      MI1.getOperand(1).getReg() == X86::ESP;
      if (!isLoad1) { ++I; continue; }

      // Check: ADD32ri or ADD32ri8 EAX, imm
      bool isAdd2 = (MI2.getOpcode() == X86::ADD32ri ||
                      MI2.getOpcode() == X86::ADD32ri8) &&
                     MI2.getOperand(0).getReg() == X86::EAX;
      if (!isAdd2) { ++I; continue; }
      int64_t AddImm = MI2.getOperand(2).getImm();

      // Check: PUSH32r EAX
      bool isPush3 = MI3.getOpcode() == X86::PUSH32r &&
                      MI3.getOperand(0).getReg() == X86::EAX;
      if (!isPush3) { ++I; continue; }

      // Matched! Rewrite:
      // add ecx, imm; push ecx; mov ecx, [esp+N+4]
      DebugLoc DL = MI0.getDebugLoc();
      int64_t EspDisp = MI1.getOperand(4).getImm();

      // Build: ADD ECX, imm (same opcode, change reg to ECX)
      BuildMI(MBB, MI0, DL, TII->get(MI2.getOpcode()), X86::ECX)
          .addReg(X86::ECX)
          .addImm(AddImm);

      // Build: PUSH32r ECX
      BuildMI(MBB, MI0, DL, TII->get(X86::PUSH32r))
          .addReg(X86::ECX);

      // Build: MOV32rm ECX, [ESP + N + 4] (push shifted ESP)
      BuildMI(MBB, MI0, DL, TII->get(X86::MOV32rm), X86::ECX)
          .addReg(X86::ESP)
          .addImm(MI1.getOperand(2).getImm())
          .addReg(MI1.getOperand(3).getReg())
          .addImm(EspDisp + 4)
          .addReg(MI1.getOperand(5).getReg());

      // Remove original 4 instructions
      MI0.eraseFromParent();
      MI1.eraseFromParent();
      MI2.eraseFromParent();
      MI3.eraseFromParent();

      Changed = true;
      continue;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferThiscallReorderPass() {
  return new X86PreferThiscallReorderPass();
}

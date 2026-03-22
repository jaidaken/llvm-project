//===--- X86PreferVtableEdx.cpp - Fix vtable call register conflict --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: After PreferMovPush unfolds PUSH32rmm to MOV32rm EAX + PUSH EAX,
// the register allocator may assign both the vtable pointer load and the
// parameter load to EAX, creating a conflict:
//
//   mov eax, [ecx]       ; vtable load
//   mov eax, [esp+4]     ; param load — clobbers vtable!
//   push eax             ; push param
//   call [eax+0x914]     ; WRONG: eax has param, not vtable
//
// MSVC 6.0 uses EDX for parameter loads in this pattern:
//
//   mov edx, [esp+4]     ; param load into EDX
//   mov eax, [ecx]       ; vtable load into EAX
//   push edx             ; push param from EDX
//   call [eax+0x914]     ; CORRECT: eax has vtable
//
// This pass detects the conflict and rewrites param loads to EDX.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-vtable-edx"

namespace {
class X86PreferVtableEdxPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferVtableEdxPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 prefer vtable EDX param pass";
  }
};
} // end anonymous namespace

char X86PreferVtableEdxPass::ID = 0;

bool X86PreferVtableEdxPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::NoCalleeSaves))
    return false;


  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E; ++I) {
      MachineInstr &CallMI = *I;

      // Step 1: Find CALL32m with a register base (indirect vtable call).
      // Also check FARCALL32m in case the compiler uses that variant.
      unsigned CallOpc = CallMI.getOpcode();
      if (CallOpc != X86::CALL32m && CallOpc != X86::FARCALL32m)
        continue;

      // CALL32m operands: base, scale, index, disp, segment
      if (!CallMI.getOperand(0).isReg())
        continue;
      Register CallBase = CallMI.getOperand(0).getReg();
      if (CallBase == X86::NoRegister || CallBase == X86::ESP)
        continue;

      // Step 2: Walk backward to find PUSH32r using CallBase.
      // Skip CFI and debug pseudo-instructions.
      auto PushIt = I;
      do {
        if (PushIt == MBB.begin()) break;
        --PushIt;
      } while (PushIt->isPseudo() && PushIt != MBB.begin());
      if (PushIt->isPseudo()) continue;
      MachineInstr &PushMI = *PushIt;
      if (PushMI.getOpcode() != X86::PUSH32r ||
          PushMI.getOperand(0).getReg() != CallBase)
        continue;

      // Step 3: Walk backward to find MOV32rm writing CallBase from ESP
      // (the parameter load).
      if (PushIt == MBB.begin())
        continue;
      auto ParamIt = std::prev(PushIt);
      MachineInstr &ParamMI = *ParamIt;
      if (ParamMI.getOpcode() != X86::MOV32rm ||
          ParamMI.getOperand(0).getReg() != CallBase)
        continue;
      // Check source is ESP-based.
      if (!ParamMI.getOperand(1).isReg() ||
          ParamMI.getOperand(1).getReg() != X86::ESP)
        continue;

      // Step 4: Walk backward to find MOV32rm writing CallBase from non-ESP
      // (the vtable load).
      if (ParamIt == MBB.begin())
        continue;
      auto VtableIt = std::prev(ParamIt);
      MachineInstr &VtableMI = *VtableIt;
      if (VtableMI.getOpcode() != X86::MOV32rm ||
          VtableMI.getOperand(0).getReg() != CallBase)
        continue;
      // Vtable load must NOT be from ESP.
      if (VtableMI.getOperand(1).isReg() &&
          VtableMI.getOperand(1).getReg() == X86::ESP)
        continue;

      // Confirmed: both loads write to CallBase (the conflict).
      // Fix: change param load dest to EDX, change push to EDX,
      // and reorder so param load comes before vtable load.


      DebugLoc DL = ParamMI.getDebugLoc();

      // Build new param load: MOV32rm EDX, [ESP+offset]
      int64_t ParamDisp = ParamMI.getOperand(4).getImm();
      MachineInstr *NewParam = BuildMI(MBB, VtableMI, DL,
          TII->get(X86::MOV32rm), X86::EDX)
          .addReg(X86::ESP)
          .addImm(ParamMI.getOperand(2).getImm())  // scale
          .addReg(ParamMI.getOperand(3).getReg())   // index
          .addImm(ParamDisp)                         // disp
          .addReg(ParamMI.getOperand(5).getReg());   // segment

      // VtableMI stays in place (now after NewParam).
      // Change push to use EDX.
      PushMI.getOperand(0).setReg(X86::EDX);

      // Remove old param load.
      ParamMI.eraseFromParent();

      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferVtableEdxPass() {
  return new X86PreferVtableEdxPass();
}

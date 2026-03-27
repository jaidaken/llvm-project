//===--- X86PreferPreloadStackParam.cpp - Pre-load stack params before CSR --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp / deusex-decomp: MSVC 6.0 pre-loads incoming stack parameters
// into scratch registers BEFORE callee-save register pushes:
//
//   mov eax, [esp+0x04]    ; pre-load stack param into scratch reg
//   push esi                ; callee-save push (changes ESP!)
//   push eax                ; push the pre-loaded value
//
// Clang folds this into a single memory operand after the callee-save pushes:
//
//   push esi                ; callee-save push
//   push [esp+0x08]         ; push directly from adjusted stack offset
//
// This pass, gated on the "prefer_preload_stack_param" string attribute,
// finds PUSH32rmm instructions with ESP-relative addressing that appear
// after callee-save pushes in the entry block. It unfolds them into
// MOV32rm EAX, [ESP+adjusted_offset] + PUSH32r EAX, placing the MOV
// before the callee-save pushes and adjusting the ESP offset accordingly.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-preload-stack-param"
#define X86_PREFER_PRELOAD_STACK_PARAM_NAME \
  "X86 prefer pre-load stack params before callee-save pushes"

namespace {
class X86PreferPreloadStackParamPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferPreloadStackParamPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return X86_PREFER_PRELOAD_STACK_PARAM_NAME;
  }

private:
  /// Check if an instruction is a callee-save register push (PUSH32r of a
  /// callee-saved register like ESI, EDI, EBX).
  static bool isCalleeSavePush(const MachineInstr &MI);

  /// Check if a PUSH32rmm reads from an ESP-relative address.
  static bool isPushFromEsp(const MachineInstr &MI);
};
} // end anonymous namespace

char X86PreferPreloadStackParamPass::ID = 0;

bool X86PreferPreloadStackParamPass::isCalleeSavePush(const MachineInstr &MI) {
  if (MI.getOpcode() != X86::PUSH32r)
    return false;
  Register Reg = MI.getOperand(0).getReg();
  // Callee-saved registers in the i386 cdecl/thiscall/stdcall ABIs.
  return Reg == X86::ESI || Reg == X86::EDI || Reg == X86::EBX;
}

bool X86PreferPreloadStackParamPass::isPushFromEsp(const MachineInstr &MI) {
  if (MI.getOpcode() != X86::PUSH32rmm)
    return false;
  // PUSH32rmm operands: base(0), scale(1), index(2), disp(3), segment(4)
  const MachineOperand &Base = MI.getOperand(0);
  const MachineOperand &Scale = MI.getOperand(1);
  const MachineOperand &Index = MI.getOperand(2);
  const MachineOperand &Disp = MI.getOperand(3);
  return Base.isReg() && Base.getReg() == X86::ESP &&
         Scale.isImm() && Scale.getImm() == 1 &&
         Index.isReg() && Index.getReg() == X86::NoRegister &&
         Disp.isImm();
}

bool X86PreferPreloadStackParamPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("prefer_preload_stack_param"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  // Only process the entry basic block - callee-save pushes are in the
  // prologue.
  MachineBasicBlock &EntryMBB = MF.front();

  // Find the range of callee-save pushes in the prologue.
  // The prologue typically looks like:
  //   push ebp           (frame pointer setup, if present)
  //   mov ebp, esp       (frame pointer setup, if present)
  //   push esi           (callee-save)
  //   push edi           (callee-save)
  //   sub esp, N          (local frame allocation)
  //   ...
  //
  // We want to find callee-save pushes (ESI, EDI, EBX) that appear before
  // the first non-prologue instruction.

  // First, find the first callee-save push and the last callee-save push.
  MachineBasicBlock::iterator FirstCSRPush = EntryMBB.end();
  MachineBasicBlock::iterator LastCSRPush = EntryMBB.end();
  unsigned CSRPushCount = 0;

  for (auto I = EntryMBB.begin(), E = EntryMBB.end(); I != E; ++I) {
    // Skip CFI instructions.
    if (I->getOpcode() == TargetOpcode::CFI_INSTRUCTION)
      continue;
    // Skip EBP setup: push ebp / mov ebp, esp
    if (I->getOpcode() == X86::PUSH32r &&
        I->getOperand(0).getReg() == X86::EBP)
      continue;
    if (I->getOpcode() == X86::MOV32rr)
      continue;

    if (isCalleeSavePush(*I)) {
      if (FirstCSRPush == EntryMBB.end())
        FirstCSRPush = I;
      LastCSRPush = I;
      CSRPushCount++;
      continue;
    }

    // Once we hit a non-CSR instruction past the prologue header, stop
    // looking for CSR pushes. But first check if it's a SUB ESP (frame
    // allocation) or other prologue instruction.
    if (I->getOpcode() == X86::SUB32ri || I->getOpcode() == X86::SUB32ri8)
      break;
    // If we already found CSR pushes, the prologue CSR section is done.
    if (FirstCSRPush != EntryMBB.end())
      break;
  }

  if (CSRPushCount == 0)
    return false;

  // Scan the entry block for PUSH32rmm with ESP-relative addressing that
  // appears after the callee-save pushes.
  for (auto I = std::next(LastCSRPush), E = EntryMBB.end(); I != E;) {
    MachineInstr &MI = *I++;

    if (!isPushFromEsp(MI))
      continue;

    int64_t OrigDisp = MI.getOperand(3).getImm();

    // The ESP offset in the PUSH32rmm is relative to ESP after the
    // callee-save pushes. The original stack param offset (before callee-save
    // pushes) would be: OrigDisp - 4 * CSRPushCount.
    // Since we're moving the MOV to before the callee-save pushes, we use
    // the pre-CSR offset.
    int64_t PreCSRDisp = OrigDisp - 4 * CSRPushCount;

    // Safety: if the adjusted displacement would be negative, this isn't a
    // stack parameter - skip it.
    if (PreCSRDisp < 0)
      continue;

    DebugLoc DL = MI.getDebugLoc();

    // Build: MOV32rm EAX, [ESP + PreCSRDisp]
    // Place it BEFORE the first callee-save push.
    BuildMI(EntryMBB, FirstCSRPush, DL, TII->get(X86::MOV32rm), X86::EAX)
        .addReg(X86::ESP)           // base
        .addImm(1)                  // scale
        .addReg(X86::NoRegister)    // index
        .addImm(PreCSRDisp)         // displacement
        .addReg(X86::NoRegister);   // segment

    // Build: PUSH32r EAX at the original location (replacing PUSH32rmm).
    BuildMI(EntryMBB, MI, DL, TII->get(X86::PUSH32r))
        .addReg(X86::EAX, RegState::Kill);

    // Remove the original PUSH32rmm.
    MI.eraseFromParent();
    Changed = true;

    // Only handle one such transformation per function - MSVC 6.0 typically
    // pre-loads one stack param this way. If there are multiple, the
    // first CSR push iterator may have shifted, so stop to be safe.
    break;
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferPreloadStackParamPass() {
  return new X86PreferPreloadStackParamPass();
}

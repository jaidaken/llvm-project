//===-- X86Msvc6FastcallRegFix.cpp - Swap EAX<->ECX for fastcall ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: Post-regalloc pass that swaps EAX<->ECX throughout a function
// when LLVM's register allocator assigned them opposite to MSVC 6.0's pattern.
//
// MSVC 6.0 fastcall functions copy the this-pointer from ECX to EAX at entry,
// then use EAX as the base register for member stores. LLVM's greedy allocator
// sometimes assigns these the other way around. This pass detects the mismatch
// by counting store base registers in the entry block and swaps EAX<->ECX
// (plus sub-registers) in all instructions when the registers are reversed.
//
// Gated behind the "msvc6_fastcall_regfix" string function attribute.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
using namespace llvm;

#define DEBUG_TYPE "x86-msvc6-fastcall-regfix"
#define X86_MSVC6_FASTCALL_REGFIX_NAME \
  "X86 MSVC 6.0 fastcall EAX<->ECX register fix"

namespace {
class X86Msvc6FastcallRegFixPass : public MachineFunctionPass {
public:
  static char ID;
  X86Msvc6FastcallRegFixPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return X86_MSVC6_FASTCALL_REGFIX_NAME;
  }
};
} // end anonymous namespace

char X86Msvc6FastcallRegFixPass::ID = 0;

/// Map a register to its swapped counterpart (EAX<->ECX and sub-registers).
static unsigned swapRegAC(unsigned Reg) {
  switch (Reg) {
  case X86::EAX: return X86::ECX;
  case X86::ECX: return X86::EAX;
  case X86::AX:  return X86::CX;
  case X86::CX:  return X86::AX;
  case X86::AL:  return X86::CL;
  case X86::CL:  return X86::AL;
  case X86::AH:  return X86::CH;
  case X86::CH:  return X86::AH;
  default: return Reg;
  }
}

/// Map a register to its swapped counterpart (ECX<->EDX and sub-registers).
static unsigned swapRegCD(unsigned Reg) {
  switch (Reg) {
  case X86::ECX: return X86::EDX;
  case X86::EDX: return X86::ECX;
  case X86::CX:  return X86::DX;
  case X86::DX:  return X86::CX;
  case X86::CL:  return X86::DL;
  case X86::DL:  return X86::CL;
  case X86::CH:  return X86::DH;
  case X86::DH:  return X86::CH;
  default: return Reg;
  }
}

bool X86Msvc6FastcallRegFixPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("msvc6_fastcall_regfix"))
    return false;

  MachineBasicBlock &EntryMBB = MF.front();

  // Find the entry MOV: first non-pseudo, non-debug instruction that is
  // MOV32rr or MOV32rr_REV with dst=EAX, src=ECX (the this-ptr copy).
  MachineInstr *EntryMov = nullptr;
  for (MachineInstr &MI : EntryMBB) {
    if (MI.isPseudo() || MI.isDebugInstr())
      continue;
    unsigned Opc = MI.getOpcode();
    if ((Opc == X86::MOV32rr || Opc == X86::MOV32rr_REV) &&
        MI.getOperand(0).getReg() == X86::EAX &&
        MI.getOperand(1).getReg() == X86::ECX) {
      EntryMov = &MI;
    }
    break; // Only check the first real instruction.
  }

  if (!EntryMov)
    return false;

  // Count store instructions that use EAX vs ECX as base register
  // in the entry basic block.
  unsigned StoresViaEAX = 0;
  unsigned StoresViaECX = 0;

  for (MachineInstr &MI : EntryMBB) {
    unsigned Opc = MI.getOpcode();
    if (Opc != X86::MOV32mr && Opc != X86::MOV32mi &&
        Opc != X86::MOV16mr && Opc != X86::MOV8mr)
      continue;
    unsigned BaseReg = MI.getOperand(0).getReg();
    if (BaseReg == X86::EAX)
      StoresViaEAX++;
    else if (BaseReg == X86::ECX)
      StoresViaECX++;
  }

  // Phase 1: EAX<->ECX swap.
  // Only swap when ALL stores use the wrong register (ECX instead of EAX).
  // If both registers are used as bases, the situation is ambiguous - skip.
  bool NeedACSwap = (StoresViaECX > 0 && StoresViaEAX == 0);

  bool Changed = false;
  if (NeedACSwap) {
    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : MBB) {
        if (&MI == EntryMov)
          continue;
        for (MachineOperand &MO : MI.operands()) {
          if (!MO.isReg())
            continue;
          unsigned NewReg = swapRegAC(MO.getReg());
          if (NewReg != MO.getReg()) {
            MO.setReg(NewReg);
            Changed = true;
          }
        }
      }
    }
  }

  // Phase 2: ECX<->EDX swap for shuttle register.
  // After the entry MOV (eax=ecx), MSVC loads the stack param into ECX and
  // uses EDX as a shuttle register for copying data. LLVM may use ECX as the
  // shuttle instead of EDX. Detect this by checking if EDX is used as a
  // load destination or store source in the entry block. If not, but ECX is
  // used that way (beyond the param load), swap ECX<->EDX for those uses.
  //
  // Pattern: mov ecx,[esp+N] (param load) then mov edx,[ecx+M] (shuttle load)
  // If LLVM generates: mov ecx,[esp+N] then mov ecx,[ecx+M] we need to swap.
  //
  // Heuristic: count non-param loads into EDX vs ECX. If ECX is used as both
  // the param pointer and the shuttle (load dest after the param load), swap.
  unsigned LoadsIntoEDX = 0;
  unsigned LoadsIntoECX = 0;
  bool SeenParamLoad = false;

  for (MachineInstr &MI : EntryMBB) {
    if (&MI == EntryMov)
      continue;
    unsigned Opc = MI.getOpcode();

    // Detect param load: MOV32rm into ECX from ESP-relative address
    if (!SeenParamLoad && Opc == X86::MOV32rm &&
        MI.getOperand(0).getReg() == X86::ECX &&
        MI.getOperand(1).getReg() == X86::ESP) {
      SeenParamLoad = true;
      continue;
    }

    // After param load, count loads into ECX vs EDX
    if (SeenParamLoad) {
      if (Opc == X86::MOV32rm || Opc == X86::MOV16rm || Opc == X86::MOV8rm ||
          Opc == X86::MOVZX32rm8 || Opc == X86::MOVZX32rm16) {
        unsigned DstReg = MI.getOperand(0).getReg();
        if (DstReg == X86::EDX || DstReg == X86::DL || DstReg == X86::DX)
          LoadsIntoEDX++;
        else if (DstReg == X86::ECX || DstReg == X86::CL || DstReg == X86::CX)
          LoadsIntoECX++;
      }
    }
  }

  // If MSVC would use EDX as shuttle but LLVM uses ECX, we need to
  // reroute the shuttle through EDX. The pattern is:
  //   mov ecx, [esp+N]    ; param load (keep as-is)
  //   mov ecx, [ecx+M]    ; shuttle load - LLVM reuses ecx, MSVC uses edx
  //   mov [eax+K], ecx    ; store - should use edx
  //
  // Fix: change destination of shuttle loads from ECX to EDX, and change
  // source of stores from ECX to EDX. Don't touch the base register of
  // the shuttle load (it should stay as ECX = param ptr).
  if (SeenParamLoad && LoadsIntoECX > 0 && LoadsIntoEDX == 0) {
    SeenParamLoad = false;
    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : MBB) {
        if (&MI == EntryMov)
          continue;
        // Skip the param load (MOV32rm ECX, [ESP+N])
        if (!SeenParamLoad && MI.getOpcode() == X86::MOV32rm &&
            MI.getOperand(0).getReg() == X86::ECX &&
            MI.getOperand(1).getReg() == X86::ESP) {
          SeenParamLoad = true;
          continue;
        }
        if (!SeenParamLoad)
          continue;

        unsigned Opc = MI.getOpcode();
        // For loads (MOV32rm, MOV8rm, etc.): only change the dest register
        // (operand 0) from ECX/CL to EDX/DL. Leave the base register alone.
        if (Opc == X86::MOV32rm || Opc == X86::MOV16rm || Opc == X86::MOV8rm ||
            Opc == X86::MOVZX32rm8 || Opc == X86::MOVZX32rm16) {
          MachineOperand &DstOp = MI.getOperand(0);
          if (DstOp.isReg()) {
            unsigned NewReg = swapRegCD(DstOp.getReg());
            if (NewReg != DstOp.getReg()) {
              DstOp.setReg(NewReg);
              Changed = true;
            }
          }
          continue;
        }
        // For stores (MOV32mr, MOV8mr, etc.): only change the source register
        // (last operand) from ECX/CL to EDX/DL. Leave the base register alone.
        if (Opc == X86::MOV32mr || Opc == X86::MOV16mr || Opc == X86::MOV8mr) {
          MachineOperand &SrcOp = MI.getOperand(MI.getNumExplicitOperands() - 1);
          if (SrcOp.isReg()) {
            unsigned NewReg = swapRegCD(SrcOp.getReg());
            if (NewReg != SrcOp.getReg()) {
              SrcOp.setReg(NewReg);
              Changed = true;
            }
          }
          continue;
        }
        // For other instructions, do a full swap of ECX<->EDX in all operands.
        for (MachineOperand &MO : MI.operands()) {
          if (!MO.isReg())
            continue;
          unsigned NewReg = swapRegCD(MO.getReg());
          if (NewReg != MO.getReg()) {
            MO.setReg(NewReg);
            Changed = true;
          }
        }
      }
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86Msvc6FastcallRegFixPass() {
  return new X86Msvc6FastcallRegFixPass();
}

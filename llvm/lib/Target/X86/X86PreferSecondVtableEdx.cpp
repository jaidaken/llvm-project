//===--- X86PreferSecondVtableEdx.cpp - Vtable reload into EDX ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: After a thiscall vtable call, the compiler reloads the vtable
// pointer into EAX (mov eax, [esi]).  MSVC 6.0 uses EDX for this second
// vtable reload (mov edx, [esi]).
//
// This pass finds the 2nd MOV32rm reg, [ESI] (the vtable reload that follows
// a CALL) and, if the destination is EAX, rewrites it and all uses to EDX
// until EDX is redefined by another instruction.
//
// Gated on the "prefer_second_vtable_edx" function attribute.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-second-vtable-edx"

namespace {

/// Map a 32-bit register to its sub-registers for EAX<->EDX swap.
static unsigned mapEaxToEdx(unsigned Reg) {
  switch (Reg) {
  case X86::EAX: return X86::EDX;
  case X86::AX:  return X86::DX;
  case X86::AL:  return X86::DL;
  case X86::AH:  return X86::DH;
  default:       return 0;
  }
}

/// Returns true if Reg is EAX or one of its sub-registers.
static bool isEaxSubreg(unsigned Reg) {
  return Reg == X86::EAX || Reg == X86::AX ||
         Reg == X86::AL  || Reg == X86::AH;
}

/// Returns true if Reg is EDX or one of its sub-registers.
static bool isEdxSubreg(unsigned Reg) {
  return Reg == X86::EDX || Reg == X86::DX ||
         Reg == X86::DL  || Reg == X86::DH;
}

/// Returns true if MI is MOV32rm with base ESI and destination Reg.
static bool isEsiLoad(const MachineInstr &MI, unsigned DestReg) {
  if (MI.getOpcode() != X86::MOV32rm)
    return false;
  if (MI.getOperand(0).getReg() != DestReg)
    return false;
  return MI.getOperand(1).isReg() &&
         MI.getOperand(1).getReg() == X86::ESI;
}

/// Returns true if MI is a CALL instruction (direct or indirect).
static bool isCallInstr(const MachineInstr &MI) {
  return MI.isCall() && !MI.isReturn();
}

class X86PreferSecondVtableEdxPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferSecondVtableEdxPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 prefer second vtable reload into EDX";
  }
};
} // end anonymous namespace

char X86PreferSecondVtableEdxPass::ID = 0;

bool X86PreferSecondVtableEdxPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("prefer_second_vtable_edx"))
    return false;

  bool Changed = false;

  // Scan ALL blocks in layout order, counting vtable loads (MOV32rm reg, [ESI])
  // globally.  After prevent_setcc_merge splits the function, the two vtable
  // loads may end up in different basic blocks.
  unsigned EsiLoadCount = 0;
  bool SeenCall = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E; ++I) {
      MachineInstr &MI = *I;
      if (MI.isPseudo() || MI.isDebugInstr())
        continue;

      if (isCallInstr(MI)) {
        SeenCall = true;
        continue;
      }

      // Count MOV32rm ?, [ESI] instructions (any destination).
      if (MI.getOpcode() == X86::MOV32rm &&
          MI.getOperand(1).isReg() &&
          MI.getOperand(1).getReg() == X86::ESI) {
        ++EsiLoadCount;

        // We want the 2nd ESI load, and it must come after a CALL.
        if (EsiLoadCount == 2 && SeenCall &&
            MI.getOperand(0).getReg() == X86::EAX) {
          // Rewrite this load's destination from EAX to EDX.
          MI.getOperand(0).setReg(X86::EDX);

          // Rewrite all uses of EAX -> EDX until EDX is redefined.
          // This may span into subsequent blocks.
          auto RewriteIt = std::next(I);
          auto *RewriteMBB = &MBB;
          bool Done = false;

          while (!Done) {
            auto RewriteEnd = RewriteMBB->end();
            while (RewriteIt != RewriteEnd) {
              MachineInstr &NMI = *RewriteIt;
              if (NMI.isPseudo() || NMI.isDebugInstr()) {
                ++RewriteIt;
                continue;
              }

              // Check if this instruction redefines EDX (before we rewrite).
              // If it does, stop rewriting after processing this instruction.
              bool RedefinesEdx = false;
              for (const MachineOperand &MO : NMI.operands()) {
                if (MO.isReg() && MO.isDef() && isEdxSubreg(MO.getReg()))
                  RedefinesEdx = true;
              }

              // Rewrite EAX uses to EDX in this instruction.
              for (MachineOperand &MO : NMI.operands()) {
                if (!MO.isReg())
                  continue;
                unsigned NewReg = mapEaxToEdx(MO.getReg());
                if (NewReg)
                  MO.setReg(NewReg);
              }

              ++RewriteIt;
              if (RedefinesEdx) {
                Done = true;
                break;
              }
            }
            if (!Done) {
              // Move to the next block in layout order.
              auto NextMBB = std::next(MachineFunction::iterator(RewriteMBB));
              if (NextMBB == MF.end()) {
                Done = true;
              } else {
                RewriteMBB = &*NextMBB;
                RewriteIt = RewriteMBB->begin();
              }
            }
          }

          Changed = true;
          // Found and processed the 2nd vtable load; we are done.
          return Changed;
        }
      }
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferSecondVtableEdxPass() {
  return new X86PreferSecondVtableEdxPass();
}

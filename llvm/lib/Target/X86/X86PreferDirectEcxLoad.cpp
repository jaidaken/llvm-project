//===--- X86PreferDirectEcxLoad.cpp - Load sub-object directly into ECX ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: MSVC 6.0 loads a sub-object pointer directly into ECX when
// followed by a thiscall. Clang uses EAX as an intermediate and adds an
// extra MOV. This pass detects the pattern:
//
//   mov eax, [mem]        ; load sub-object into EAX
//   test eax, eax         ; NULL check
//   je .skip
//   mov ecx, eax          ; EXTRA: copy to ECX for thiscall
//   ...                   ; vtable load + call
//
// And rewrites to:
//
//   mov ecx, [mem]        ; load sub-object directly into ECX
//   test ecx, ecx         ; NULL check
//   je .skip
//   ...                   ; vtable load + call (MOV ECX, EAX removed)
//
// All uses of EAX between the load and the MOV ECX, EAX are rewritten to ECX.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-direct-ecx-load"

namespace {
class X86PreferDirectEcxLoadPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferDirectEcxLoadPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 prefer direct ECX load for thiscall";
  }
};
} // end anonymous namespace

char X86PreferDirectEcxLoadPass::ID = 0;

/// Return true if the instruction defines or uses ECX (would conflict with
/// rewriting EAX->ECX in the gap between the load and the MOV ECX, EAX).
static bool touchesECX(const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg())
      continue;
    Register Reg = MO.getReg();
    if (Reg == X86::ECX || Reg == X86::CX || Reg == X86::CL || Reg == X86::CH)
      return true;
  }
  return false;
}

/// Return true if the instruction defines EAX (or a sub-register).
static bool definesEAX(const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.isDef())
      continue;
    Register Reg = MO.getReg();
    if (Reg == X86::EAX || Reg == X86::AX || Reg == X86::AL || Reg == X86::AH)
      return true;
  }
  return false;
}

/// Rewrite all operand references from EAX to ECX in the given instruction.
static void rewriteEAXtoECX(MachineInstr &MI) {
  for (MachineOperand &MO : MI.operands()) {
    if (!MO.isReg())
      continue;
    if (MO.getReg() == X86::EAX)
      MO.setReg(X86::ECX);
    else if (MO.getReg() == X86::AX)
      MO.setReg(X86::CX);
    else if (MO.getReg() == X86::AL)
      MO.setReg(X86::CL);
    else if (MO.getReg() == X86::AH)
      MO.setReg(X86::CH);
  }
}

/// Rewrite all operand references from OldReg to NewReg in the given
/// instruction (32-bit registers only).
static void rewriteReg(MachineInstr &MI, Register OldReg, Register NewReg) {
  for (MachineOperand &MO : MI.operands()) {
    if (!MO.isReg())
      continue;
    if (MO.getReg() == OldReg)
      MO.setReg(NewReg);
  }
}

/// After removing MOV ECX, EAX, scan forward from \p Start to find and rewrite
/// the vtable load pattern. The compiler originally chose a register other than
/// EAX (e.g. EDX) for the vtable load to avoid clobbering EAX. Since EAX is
/// now free, rewrite the vtable register to EAX to match MSVC 6.0 output.
///
/// Pattern: MOV32rm REG, [ECX+N] followed by CALL32m [REG+N]
/// Rewrite: MOV32rm EAX, [ECX+N] followed by CALL32m [EAX+N]
static void rewriteVtableReg(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator Start) {
  for (auto K = Start; K != MBB.end(); ++K) {
    MachineInstr &MI = *K;

    // Stop at terminators or anything that defines EAX (would conflict).
    if (MI.isTerminator())
      return;
    if (definesEAX(MI))
      return;

    // Look for MOV32rm REG, [ECX ...] where REG is not EAX and not ECX.
    if (MI.getOpcode() != X86::MOV32rm)
      continue;

    Register VtableReg = MI.getOperand(0).getReg();
    if (VtableReg == X86::EAX || VtableReg == X86::ECX)
      continue;

    // Check that the base register of the load is ECX (operand 1).
    if (MI.getOperand(1).getReg() != X86::ECX)
      continue;

    // Found the vtable load. Rewrite destination to EAX.
    MI.getOperand(0).setReg(X86::EAX);

    // Scan forward from the vtable load to find the CALL that uses VtableReg
    // as a base register, and rewrite it to EAX.
    for (auto L = std::next(K); L != MBB.end(); ++L) {
      MachineInstr &CallMI = *L;

      // Look for CALL32m (indirect call through memory).
      if (CallMI.getOpcode() == X86::CALL32m) {
        // The base register is operand 0 in CALL32m.
        if (CallMI.getOperand(0).isReg() &&
            CallMI.getOperand(0).getReg() == VtableReg) {
          rewriteReg(CallMI, VtableReg, X86::EAX);
        }
        break;
      }

      // Also handle CALL32r (indirect call through register).
      if (CallMI.getOpcode() == X86::CALL32r) {
        if (CallMI.getOperand(0).isReg() &&
            CallMI.getOperand(0).getReg() == VtableReg) {
          rewriteReg(CallMI, VtableReg, X86::EAX);
        }
        break;
      }

      // If VtableReg is redefined before the CALL, bail.
      for (const MachineOperand &MO : CallMI.operands()) {
        if (MO.isReg() && MO.isDef() && MO.getReg() == VtableReg)
          return;
      }

      // If EAX is used/defined before the CALL, bail (would conflict).
      if (definesEAX(CallMI))
        return;
      for (const MachineOperand &MO : CallMI.operands()) {
        if (MO.isReg() && MO.isUse() && MO.getReg() == X86::EAX)
          return;
      }

      if (CallMI.isTerminator())
        break;
    }

    return; // Only handle one vtable load per pattern.
  }
}

bool X86PreferDirectEcxLoadPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("prefer_direct_ecx_load"))
    return false;

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(); I != MBB.end(); ++I) {
      MachineInstr &LoadMI = *I;

      // Step 1: Find MOV32rm EAX, [mem]
      if (LoadMI.getOpcode() != X86::MOV32rm)
        continue;
      if (LoadMI.getOperand(0).getReg() != X86::EAX)
        continue;

      // Step 2: Scan forward to find MOV32rr ECX, EAX (the copy we want to
      // eliminate). Along the way, collect all instructions whose EAX references
      // must be rewritten to ECX. Bail if anything touches ECX or if EAX is
      // redefined by something other than the copy target.
      SmallVector<MachineInstr *, 8> RewriteCandidates;
      MachineInstr *CopyMI = nullptr;
      bool Bail = false;

      auto J = std::next(I);
      for (; J != MBB.end(); ++J) {
        MachineInstr &MI = *J;

        // Found the copy: MOV32rr ECX, EAX (or MOV32rr_REV)
        if ((MI.getOpcode() == X86::MOV32rr ||
             MI.getOpcode() == X86::MOV32rr_REV) &&
            MI.getOperand(0).getReg() == X86::ECX &&
            MI.getOperand(1).getReg() == X86::EAX) {
          CopyMI = &MI;
          break;
        }

        // If this instruction defines EAX (other than implicit EFLAGS etc.),
        // the chain is broken - we can't rewrite past a redefinition.
        if (definesEAX(MI)) {
          Bail = true;
          break;
        }

        // If this instruction touches ECX at all, we can't safely rewrite
        // EAX->ECX without clobbering the existing ECX usage.
        if (touchesECX(MI)) {
          Bail = true;
          break;
        }

        // If this instruction is a branch/terminator that leaves the block,
        // stop scanning (the copy must be in the same block path).
        if (MI.isTerminator()) {
          // Conditional branches are OK - the copy might be after them in
          // a fall-through successor. But we only handle intra-block patterns.
          // However, we allow conditional jumps (Jcc) as part of the pattern
          // since the NULL-check je is between the test and the copy.
          if (MI.isUnconditionalBranch()) {
            Bail = true;
            break;
          }
          // Conditional branch: continue scanning (fall-through continues)
          continue;
        }

        // This instruction uses EAX and will need rewriting.
        RewriteCandidates.push_back(&MI);
      }

      if (Bail || !CopyMI)
        continue;

      // Step 3: Apply the transformation.
      // Change the load destination from EAX to ECX.
      LoadMI.getOperand(0).setReg(X86::ECX);

      // Rewrite all intermediate instructions from EAX to ECX.
      for (MachineInstr *MI : RewriteCandidates)
        rewriteEAXtoECX(*MI);

      // Remove the now-redundant MOV32rr ECX, EAX. Save the position of the
      // instruction after the copy so we can scan from it for the vtable load.
      auto AfterCopy = std::next(MachineBasicBlock::iterator(CopyMI));
      CopyMI->eraseFromParent();

      // Rewrite the vtable load register from whatever the compiler chose
      // (to avoid clobbering EAX) to EAX, since EAX is now free.
      if (AfterCopy != MBB.end())
        rewriteVtableReg(MBB, AfterCopy);

      Changed = true;
      // Don't advance I - the iterator is still valid and points to the
      // rewritten load. Next iteration will move past it.
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferDirectEcxLoadPass() {
  return new X86PreferDirectEcxLoadPass();
}

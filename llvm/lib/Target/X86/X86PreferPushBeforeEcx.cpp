//===--- X86PreferPushBeforeEcx.cpp - Push args before ECX setup ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: Post-regalloc pass that reorders thiscall argument pushes before
// the ECX (this pointer) setup to match MSVC 6.0 codegen.
//
// MSVC 6.0 pushes arguments BEFORE setting up ECX for thiscall:
//   mov eax, [esi]         ; load vtable
//   push 0x6b              ; push arg FIRST
//   mov ecx, esi           ; then setup this
//   call [eax + 0x8e8]     ; vtable call
//
// Clang sets up ECX BEFORE pushing:
//   mov eax, [esi]         ; load vtable
//   mov ecx, esi           ; setup this FIRST
//   push 0x6b              ; then push arg
//   call [eax + 0x8e8]     ; vtable call
//
// This pass detects:
//   MOV32rr ECX, REG   (where REG is ESI, EDI, or EBX)
//   PUSH ...           (one or more)
//   CALL ...
//
// And moves the MOV ECX to just before the CALL (after all PUSHes).
//
// Known limitation: when MOV ECX, ESI is followed by a branch that leads
// to two different basic blocks each containing their own PUSH+CALL
// sequence, this pass cannot handle the reorder. The pattern looks like:
//
//   mov ecx, esi       ; set up this
//   test/cmp ...       ; conditional
//   je .path2
// .path1:
//   push ...; call ... ; call site 1
// .path2:
//   push ...; call ... ; call site 2
//
// In this case, ECX is set before the branch and consumed in separate
// blocks. The pass only matches MOV+PUSH+CALL within a single basic block,
// so it skips these cross-block cases. MSVC 6.0 would push before setting
// ECX in each path independently. Handling this would require duplicating
// the MOV into each successor block, which is not implemented.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-push-before-ecx"

namespace {
class X86PreferPushBeforeEcxPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferPushBeforeEcxPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 prefer push before ECX setup";
  }
};
} // end anonymous namespace

char X86PreferPushBeforeEcxPass::ID = 0;

/// Check if MI is a MOV32rr/MOV32rr_REV that sets ECX from ESI, EDI, or EBX.
static bool isMovEcxFromCalleeSaved(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  if (Opc != X86::MOV32rr && Opc != X86::MOV32rr_REV)
    return false;
  if (MI.getNumOperands() < 2)
    return false;
  if (!MI.getOperand(0).isReg() || MI.getOperand(0).getReg() != X86::ECX)
    return false;
  if (!MI.getOperand(1).isReg())
    return false;
  Register SrcReg = MI.getOperand(1).getReg();
  return SrcReg == X86::ESI || SrcReg == X86::EDI || SrcReg == X86::EBX;
}

/// Check if MI is a PUSH instruction (any variant used for argument passing).
static bool isArgPush(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case X86::PUSH32i8:
  case X86::PUSH32i:
  case X86::PUSH32r:
  case X86::PUSH32rmm:
    return true;
  default:
    return false;
  }
}

/// Check if MI is any CALL instruction.
static bool isAnyCall(const MachineInstr &MI) {
  return MI.isCall();
}

/// Check if a PUSH instruction references ESP in any of its operands.
/// If so, we would need to adjust the displacement after moving the MOV
/// past the push (since MOV ECX, REG doesn't touch ESP, this is unlikely
/// but we check for safety).
static bool pushReferencesEsp(const MachineInstr &MI) {
  // Only check explicit operands. Every PUSH has implicit-def $esp
  // which would cause a false positive.
  for (const MachineOperand &MO : MI.explicit_operands()) {
    if (MO.isReg() && MO.getReg() == X86::ESP)
      return true;
  }
  return false;
}

/// Check if the MOV instruction references ESP (e.g., mov ecx, [esp+N]).
/// The pattern we match only has MOV32rr ECX, REG so this should never
/// be true, but we check for safety.
static bool movReferencesEsp(const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isReg() && MO.getReg() == X86::ESP)
      return true;
  }
  return false;
}

bool X86PreferPushBeforeEcxPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("prefer_push_before_ecx"))
    return false;

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E; /* updated in body */) {
      // Look for: MOV32rr ECX, REG (ESI/EDI/EBX)
      if (!isMovEcxFromCalleeSaved(*I)) {
        ++I;
        continue;
      }

      auto MovIt = I;

      // If the MOV references ESP, skip (shouldn't happen for reg-reg MOV).
      if (movReferencesEsp(*MovIt)) {
        ++I;
        continue;
      }

      // Skip past instructions between MOV ECX and the first PUSH.
      // The compiler often places vtable loads (MOV EAX, [ESI]) or
      // ADJCALLSTACKDOWN pseudos between the MOV ECX and the PUSHes.
      // Skip pseudos unconditionally (they produce no code).
      // Skip real instructions as long as they don't write ECX or ESP.
      auto Next = std::next(MovIt);
      while (Next != E && !isArgPush(*Next) && !isAnyCall(*Next)) {
        // Always skip pseudo instructions (ADJCALLSTACKDOWN, CFI, etc.)
        // They declare ESP/EFLAGS defs but produce no actual code.
        if (Next->isPseudo()) {
          ++Next;
          continue;
        }
        // For real instructions, bail if they write ECX or ESP.
        bool WritesEcxOrEsp = false;
        for (const MachineOperand &MO : Next->operands()) {
          if (MO.isReg() && MO.isDef() &&
              (MO.getReg() == X86::ECX || MO.getReg() == X86::ESP)) {
            WritesEcxOrEsp = true;
            break;
          }
        }
        if (WritesEcxOrEsp)
          break;
        ++Next;
      }

      if (Next == E || !isArgPush(*Next)) {
        ++I;
        continue;
      }

      // Collect all consecutive PUSHes (and skip pseudos between them).
      auto PushEnd = Next;
      while (PushEnd != E && (isArgPush(*PushEnd) || PushEnd->isPseudo()))
        ++PushEnd;

      // PushEnd should now point to a CALL.
      if (PushEnd == E || !isAnyCall(*PushEnd)) {
        ++I;
        continue;
      }

      // Safety: if any PUSH references ESP, we would need to adjust
      // ESP-relative offsets within those PUSHes after moving the MOV
      // past them. Since MOV ECX, REG does not modify ESP, the PUSHes
      // themselves don't change, but we bail on this edge case for now.
      // (In practice, pushes of immediates and registers don't use ESP.)
      bool HasEspPush = false;
      for (auto P = Next; P != PushEnd; ++P) {
        if (pushReferencesEsp(*P)) {
          HasEspPush = true;
          break;
        }
      }
      if (HasEspPush) {
        ++I;
        continue;
      }

      // Move the MOV ECX, REG to just before the CALL (after all PUSHes).
      // Before: MOV ECX, ESI; PUSH ...; PUSH ...; CALL
      // After:  PUSH ...; PUSH ...; MOV ECX, ESI; CALL
      MBB.splice(PushEnd, &MBB, MovIt);
      Changed = true;

      // Continue scanning from the first PUSH (now at the position where
      // MOV was, which is Next). The iterator I was invalidated by splice,
      // so reset to Next.
      I = Next;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferPushBeforeEcxPass() {
  return new X86PreferPushBeforeEcxPass();
}

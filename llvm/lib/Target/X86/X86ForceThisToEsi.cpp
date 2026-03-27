//===--- X86ForceThisToEsi.cpp - Force this ptr from ECX to ESI -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: Post-regalloc pass that forces `this` (ECX) into ESI at function
// entry and rewrites all subsequent ECX uses to ESI.
//
// MSVC 6.0 saves `this` (ECX) to ESI at function entry:
//   push esi          ; (from forced_callee_saves)
//   mov esi, ecx      ; this pass inserts this
//
// Then all subsequent code uses ESI instead of ECX for `this`. Clang only
// saves to ESI when ECX gets clobbered by a function call. For functions
// that don't call anything before using `this`, Clang uses ECX directly,
// producing different bytes.
//
// This pass:
//   1. Gates on the force_this_esi attribute.
//   2. Inserts MOV32rr ESI, ECX after any prologue pushes.
//   3. Rewrites USE operands of ECX to ESI, but only while ECX still holds
//      the original `this` value. After an instruction that redefines ECX
//      from a non-ESI/EDI source (e.g., `MOV32rm ECX, [mem]` - a sub-object
//      load), rewriting stops until the next `MOV32rr ECX, ESI/EDI` restores
//      ECX from the this-pointer register.
//   4. Removes nop MOVs (e.g., `MOV ESI, ESI`) that result from rewriting
//      the compiler's own ECX-to-ESI save.
//   5. Reorders thiscall call setup: moves `MOV ECX, ESI` to just before
//      the CALL, after any PUSHes (MSVC 6.0 pushes args first, then sets
//      up ECX).
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
using namespace llvm;

#define DEBUG_TYPE "x86-force-this-to-esi"
#define X86_FORCE_THIS_TO_ESI_NAME "X86 force this pointer to ESI pass"

namespace {
class X86ForceThisToEsiPass : public MachineFunctionPass {
public:
  static char ID;
  X86ForceThisToEsiPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return X86_FORCE_THIS_TO_ESI_NAME; }
};
} // end anonymous namespace

char X86ForceThisToEsiPass::ID = 0;

/// Check if an instruction is a prologue PUSH (callee-saved register save).
static bool isProloguePush(const MachineInstr &MI) {
  return MI.getOpcode() == X86::PUSH32r;
}

/// Rewrite a register USE operand: ECX->TargetReg32 (and sub-regs).
/// Only rewrites uses (not defs) so that outgoing thiscall call setup
/// instructions like `MOV32rr ECX, ESI` keep ECX as the destination.
/// Returns true if the operand was changed.
static bool rewriteEcxToTarget(MachineOperand &MO,
                                Register Target32, Register Target16,
                                Register Target8) {
  if (!MO.isReg() || MO.isDef())
    return false;
  Register Reg = MO.getReg();
  if (Reg == X86::ECX) { MO.setReg(Target32); return true; }
  if (Reg == X86::CX)  { MO.setReg(Target16); return true; }
  if (Reg == X86::CL)  { MO.setReg(Target8);  return true; }
  return false;
}

/// Check if a MOV instruction is a nop (source == dest).
static bool isNopMov(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  if (Opc != X86::MOV32rr && Opc != X86::MOV32rr_REV &&
      Opc != X86::MOV16rr && Opc != X86::MOV16rr_REV &&
      Opc != X86::MOV8rr && Opc != X86::MOV8rr_REV)
    return false;
  return MI.getNumOperands() >= 2 &&
         MI.getOperand(0).isReg() && MI.getOperand(1).isReg() &&
         MI.getOperand(0).getReg() == MI.getOperand(1).getReg();
}

/// Check if MI is a MOV that sets ECX from TargetReg (thiscall this-ptr setup).
static bool isMovEcxTarget(const MachineInstr &MI, Register TargetReg) {
  unsigned Opc = MI.getOpcode();
  if (Opc != X86::MOV32rr && Opc != X86::MOV32rr_REV)
    return false;
  return MI.getNumOperands() >= 2 &&
         MI.getOperand(0).isReg() && MI.getOperand(0).getReg() == X86::ECX &&
         MI.getOperand(1).isReg() && MI.getOperand(1).getReg() == TargetReg;
}

/// Check if MI is a PUSH instruction (any variant).
static bool isAnyPush(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case X86::PUSH32r:
  case X86::PUSH32i8:
  case X86::PUSH32i:
  case X86::PUSH32rmm:
    return true;
  default:
    return false;
  }
}

/// Check if MI is a CALL instruction (any variant).
static bool isAnyCall(const MachineInstr &MI) {
  return MI.isCall();
}

bool X86ForceThisToEsiPass::runOnMachineFunction(MachineFunction &MF) {
  const Function &Fn = MF.getFunction();
  bool UseEsi = Fn.hasFnAttribute(Attribute::ForceThisEsi);
  bool UseEdi = Fn.hasFnAttribute("force_this_edi");
  if (!UseEsi && !UseEdi)
    return false;

  // Pick target register set.
  Register Target32 = UseEsi ? X86::ESI : X86::EDI;
  Register Target16 = UseEsi ? X86::SI  : X86::DI;
  Register Target8  = UseEsi ? X86::SIL : X86::DIL;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();

  MachineBasicBlock &EntryMBB = MF.front();

  // Find the insertion point: after all prologue PUSHes.
  MachineBasicBlock::iterator InsertPt = EntryMBB.begin();
  while (InsertPt != EntryMBB.end() && isProloguePush(*InsertPt))
    ++InsertPt;

  // Insert: MOV32rr Target32, ECX.
  DebugLoc DL;
  MachineInstr *InsertedMov =
      BuildMI(EntryMBB, InsertPt, DL, TII->get(X86::MOV32rr), Target32)
          .addReg(X86::ECX);

  // Rewrite USE operands of ECX/CX/CL to Target, but only while ECX still
  // holds the original `this` value. Track ECX redefinitions: if ECX is
  // defined from a non-ESI/EDI source (e.g., `MOV32rm ECX, [mem]` for a
  // sub-object field load), stop rewriting ECX uses. Resume rewriting after
  // a `MOV32rr ECX, ESI/EDI` that restores ECX from the this-pointer reg.
  //
  // This prevents corrupting the this-pointer when ECX is reused for a
  // different object, e.g.:
  //   mov ecx, [esi+0xbc]  ; load sub-object (ECX != this)
  //   call [ecx]           ; call method on sub-object
  // Without this check, the pass would rewrite both ECX uses to ESI,
  // producing `mov esi, [esi+0xbc]; call [esi]` which corrupts `this`.
  bool Changed = true; // We already inserted the MOV.

  // Two-pass approach: first pass computes EcxIsThis state at exit of each
  // block. Second pass uses predecessor exit states to determine entry state.
  DenseMap<MachineBasicBlock *, bool> BlockExitState;

  // First pass: compute exit state for each block.
  for (MachineBasicBlock &MBB : MF) {
    bool EcxIsThis = (&MBB == &MF.front()); // true only for entry block
    if (&MBB != &MF.front()) {
      // For non-entry blocks, assume true only if ALL predecessors exit true.
      // Default to true, will be corrected in second pass.
      EcxIsThis = true;
    }

    for (MachineInstr &MI : MBB) {
      if (&MI == InsertedMov)
        continue;
      bool DefsEcx = false;
      for (const MachineOperand &MO : MI.operands()) {
        if (MO.isReg() && MO.isDef() &&
            (MO.getReg() == X86::ECX || MO.getReg() == X86::CX ||
             MO.getReg() == X86::CL || MO.getReg() == X86::CH)) {
          DefsEcx = true;
          break;
        }
      }
      if (DefsEcx) {
        unsigned Opc = MI.getOpcode();
        if ((Opc == X86::MOV32rr || Opc == X86::MOV32rr_REV) &&
            MI.getNumOperands() >= 2 &&
            MI.getOperand(0).isReg() && MI.getOperand(0).getReg() == X86::ECX &&
            MI.getOperand(1).isReg() &&
            (MI.getOperand(1).getReg() == Target32))
          EcxIsThis = true;
        else
          EcxIsThis = false;
      }
    }
    BlockExitState[&MBB] = EcxIsThis;
  }

  // Second pass: rewrite with correct per-block entry state.
  for (MachineBasicBlock &MBB : MF) {
    bool EcxIsThis;
    if (&MBB == &MF.front()) {
      EcxIsThis = true;
    } else {
      // Entry state = AND of all predecessor exit states.
      // If any predecessor exits with EcxIsThis=false, we must be conservative.
      EcxIsThis = true;
      for (MachineBasicBlock *Pred : MBB.predecessors()) {
        auto It = BlockExitState.find(Pred);
        if (It != BlockExitState.end() && !It->second) {
          EcxIsThis = false;
          break;
        }
      }
    }

    for (MachineInstr &MI : MBB) {
      // Skip the MOV we just inserted.
      if (&MI == InsertedMov)
        continue;

      // IMPORTANT: Rewrite USEs BEFORE checking DEFs. For instructions like
      // `mov ecx, [ecx+0xbc]`, the USE of ECX (base register) must be
      // rewritten to ESI BEFORE the DEF of ECX breaks the alias. Otherwise
      // we get `mov ecx, [ecx+0xbc]` instead of `mov ecx, [esi+0xbc]`.
      if (EcxIsThis) {
        for (MachineOperand &MO : MI.operands())
          Changed |= rewriteEcxToTarget(MO, Target32, Target16, Target8);
      }

      // Now check if this instruction DEFs ECX, updating the alias state
      // for subsequent instructions.
      bool DefsEcx = false;
      for (const MachineOperand &MO : MI.operands()) {
        if (MO.isReg() && MO.isDef() &&
            (MO.getReg() == X86::ECX || MO.getReg() == X86::CX ||
             MO.getReg() == X86::CL || MO.getReg() == X86::CH)) {
          DefsEcx = true;
          break;
        }
      }

      if (DefsEcx) {
        // Check if this is `MOV32rr ECX, ESI/EDI` (restoring this-ptr).
        unsigned Opc = MI.getOpcode();
        if ((Opc == X86::MOV32rr || Opc == X86::MOV32rr_REV) &&
            MI.getNumOperands() >= 2 &&
            MI.getOperand(0).isReg() && MI.getOperand(0).getReg() == X86::ECX &&
            MI.getOperand(1).isReg() &&
            (MI.getOperand(1).getReg() == X86::ESI ||
             MI.getOperand(1).getReg() == X86::EDI)) {
          EcxIsThis = true;
        } else {
          EcxIsThis = false;
        }
      }
    }
  }

  // Remove nop MOVs created by rewriting (e.g., compiler's own
  // `MOV ESI, ECX` becomes `MOV ESI, ESI` after the ECX use is rewritten).
  SmallVector<MachineInstr *, 4> NopMovs;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (&MI == InsertedMov)
        continue;
      if (isNopMov(MI))
        NopMovs.push_back(&MI);
    }
  }
  for (MachineInstr *MI : NopMovs) {
    MI->eraseFromParent();
    Changed = true;
  }

  // Reorder thiscall call setup: MSVC 6.0 pushes arguments first, then
  // sets up ECX just before the CALL. Clang puts MOV ECX before the PUSHes.
  //
  // Pattern: MOV ECX, ESI; PUSH...; PUSH...; CALL
  // Target:  PUSH...; PUSH...; MOV ECX, ESI; CALL
  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E; ++I) {
      if (!isMovEcxTarget(*I, Target32))
        continue;

      // Check if followed by one or more PUSHes and then a CALL.
      auto MovIt = I;
      auto Next = std::next(MovIt);
      if (Next == E || !isAnyPush(*Next))
        continue;

      // Count PUSHes.
      auto PushStart = Next;
      auto PushEnd = PushStart;
      while (PushEnd != E && isAnyPush(*PushEnd))
        ++PushEnd;

      // PushEnd should now point to a CALL.
      if (PushEnd == E || !isAnyCall(*PushEnd))
        continue;

      // Move the MOV ECX, ESI to just before the CALL (after all PUSHes).
      MBB.splice(PushEnd, &MBB, MovIt);
      Changed = true;
      // Reset I to continue scanning.
      I = PushStart;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86ForceThisToEsiPass() {
  return new X86ForceThisToEsiPass();
}

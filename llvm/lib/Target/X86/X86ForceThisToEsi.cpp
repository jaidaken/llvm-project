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
//   3. Rewrites all subsequent USE operands of ECX to ESI.
//      DEF operands are NOT rewritten so that outgoing thiscall call setup
//      like `MOV ECX, ESI` keeps ECX as the destination.
//   4. Removes nop MOVs (e.g., `MOV ESI, ESI`) that result from rewriting
//      the compiler's own ECX-to-ESI save.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
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

/// Rewrite a register USE operand: ECX->ESI, CX->SI, CL->SIL.
/// Only rewrites uses (not defs) so that outgoing thiscall call setup
/// instructions like `MOV32rr ECX, ESI` keep ECX as the destination.
/// Returns true if the operand was changed.
static bool rewriteEcxToEsi(MachineOperand &MO) {
  if (!MO.isReg() || MO.isDef())
    return false;
  Register Reg = MO.getReg();
  if (Reg == X86::ECX) { MO.setReg(X86::ESI); return true; }
  if (Reg == X86::CX)  { MO.setReg(X86::SI);  return true; }
  if (Reg == X86::CL)  { MO.setReg(X86::SIL); return true; }
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

bool X86ForceThisToEsiPass::runOnMachineFunction(MachineFunction &MF) {
  const Function &Fn = MF.getFunction();
  if (!Fn.hasFnAttribute(Attribute::ForceThisEsi))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();

  MachineBasicBlock &EntryMBB = MF.front();

  // Find the insertion point: after all prologue PUSHes.
  // forced_callee_saves generates PUSH32r instructions at the start of the
  // entry block. We insert MOV ESI, ECX right after them.
  MachineBasicBlock::iterator InsertPt = EntryMBB.begin();
  while (InsertPt != EntryMBB.end() && isProloguePush(*InsertPt))
    ++InsertPt;

  // Insert: MOV32rr ESI, ECX. The encoding (normal vs REV) will be handled
  // by the ReversedOps pass later if MOV32rr_REV attribute is set.
  DebugLoc DL;
  MachineInstr *InsertedMov =
      BuildMI(EntryMBB, InsertPt, DL, TII->get(X86::MOV32rr), X86::ESI)
          .addReg(X86::ECX);

  // Rewrite all USE operands of ECX/CX/CL to ESI/SI/SIL in the entire
  // function, EXCEPT for the MOV we just inserted.
  bool Changed = true; // We already inserted the MOV.

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      // Skip the MOV we just inserted.
      if (&MI == InsertedMov)
        continue;

      for (MachineOperand &MO : MI.operands())
        Changed |= rewriteEcxToEsi(MO);
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

  return Changed;
}

FunctionPass *llvm::createX86ForceThisToEsiPass() {
  return new X86ForceThisToEsiPass();
}

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
//   2. Inserts MOV32rr ESI, ECX (or MOV32rr_REV if msvc6_regalloc is set)
//      after any prologue pushes from forced_callee_saves.
//   3. Rewrites all subsequent uses of ECX in the function body to ESI.
//   4. Before thiscall CALL instructions that need ECX, the rewrite
//      naturally produces `mov ecx, esi` (the caller sets up ECX from ESI).
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

/// Rewrite a register operand: ECX->ESI, CX->SI, CL->SIL.
/// Returns true if the operand was changed.
static bool rewriteEcxToEsi(MachineOperand &MO) {
  if (!MO.isReg())
    return false;
  Register Reg = MO.getReg();
  if (Reg == X86::ECX) { MO.setReg(X86::ESI); return true; }
  if (Reg == X86::CX)  { MO.setReg(X86::SI);  return true; }
  if (Reg == X86::CL)  { MO.setReg(X86::SIL); return true; }
  return false;
}

bool X86ForceThisToEsiPass::runOnMachineFunction(MachineFunction &MF) {
  const Function &Fn = MF.getFunction();
  if (!Fn.hasFnAttribute(Attribute::ForceThisEsi))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool UseREV = Fn.hasFnAttribute(Attribute::Msvc6RegAlloc);

  MachineBasicBlock &EntryMBB = MF.front();

  // Find the insertion point: after all prologue PUSHes.
  // forced_callee_saves generates PUSH32r instructions at the start of the
  // entry block. We insert MOV ESI, ECX right after them.
  MachineBasicBlock::iterator InsertPt = EntryMBB.begin();
  while (InsertPt != EntryMBB.end() && isProloguePush(*InsertPt))
    ++InsertPt;

  // Insert: MOV32rr ESI, ECX (or MOV32rr_REV for msvc6_regalloc encoding).
  unsigned MovOpc = UseREV ? X86::MOV32rr_REV : X86::MOV32rr;
  DebugLoc DL;
  BuildMI(EntryMBB, InsertPt, DL, TII->get(MovOpc), X86::ESI)
      .addReg(X86::ECX);

  // Rewrite all uses of ECX/CX/CL to ESI/SI/SIL in the entire function,
  // EXCEPT for the MOV we just inserted and EXCEPT for instructions that
  // set up ECX for outgoing thiscall calls (those will naturally become
  // `mov ecx, esi` after rewriting their source).
  //
  // We iterate every instruction in every block. The MOV we inserted reads
  // ECX and defines ESI - we skip it to avoid self-corruption.
  bool Changed = true; // We already inserted the MOV.

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      // Skip our inserted MOV (it's the only MOV32rr/MOV32rr_REV that
      // defines ESI and reads ECX in the entry block before any other code).
      if (&MBB == &EntryMBB &&
          (MI.getOpcode() == X86::MOV32rr ||
           MI.getOpcode() == X86::MOV32rr_REV) &&
          MI.getNumOperands() >= 2 &&
          MI.getOperand(0).isReg() && MI.getOperand(0).getReg() == X86::ESI &&
          MI.getOperand(1).isReg() && MI.getOperand(1).getReg() == X86::ECX)
        continue;

      for (MachineOperand &MO : MI.operands())
        Changed |= rewriteEcxToEsi(MO);
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86ForceThisToEsiPass() {
  return new X86ForceThisToEsiPass();
}

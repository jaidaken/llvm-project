//===--- X86InsertRedundantCmp.cpp - Insert CMP after DEC [mem] -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: MSVC 6.0 emits a redundant CMP [mem], 0 after DEC [mem] even
// though DEC already sets ZF.  Modern compilers omit the CMP, which prevents
// byte-exact matching of ~10-15 functions.
//
// Example (ScaredStiff):
//   dec word ptr [ecx + 0x58]
//   cmp word ptr [ecx + 0x58], 0x00   ; redundant - DEC already set ZF
//   jne skip
//
// This pass inserts CMP16mi [mem], 0 or CMP32mi8 [mem], 0 between a memory
// DEC and its EFLAGS consumer (Jcc, SETcc, CMOVcc) to reproduce the MSVC 6.0
// pattern.  The CMP re-reads the decremented value from memory and sets ZF.
//
// Gated on the function attribute "insert_redundant_cmp".
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "x86-insert-redundant-cmp"

namespace {
class X86InsertRedundantCmpPass : public MachineFunctionPass {
public:
  static char ID;
  X86InsertRedundantCmpPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 insert redundant CMP after DEC [mem]";
  }
};
} // end anonymous namespace

char X86InsertRedundantCmpPass::ID = 0;

/// Return the CMP memory-immediate opcode that matches a memory DEC size.
/// DEC16m -> CMP16mi (compare word [mem], imm16)
/// DEC32m -> CMP32mi8 (compare dword [mem], sign-extended imm8)
static unsigned getCmpMiForDecM(unsigned DecOpc) {
  switch (DecOpc) {
  case X86::DEC16m: return X86::CMP16mi;
  case X86::DEC32m: return X86::CMP32mi8;
  default: return 0;
  }
}

/// Check if an instruction reads EFLAGS (Jcc, SETcc, CMOVcc, etc.).
static bool usesEFLAGS(const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isReg() && MO.isUse() && MO.getReg() == X86::EFLAGS)
      return true;
  }
  return false;
}

/// Number of memory operands in X86 memory-form instructions
/// (base, scale, index, disp, segment).
static constexpr unsigned X86MemOperands = 5;

bool X86InsertRedundantCmpPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("insert_redundant_cmp"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E; ++I) {
      MachineInstr &DecMI = *I;

      unsigned DecOpc = DecMI.getOpcode();
      unsigned CmpOpc = getCmpMiForDecM(DecOpc);
      if (!CmpOpc)
        continue;

      // DEC [mem] has exactly 5 operands (base, scale, index, disp, segment).
      if (DecMI.getNumOperands() < X86MemOperands)
        continue;

      // Find the next non-debug instruction after the DEC.
      auto Next = std::next(I);
      while (Next != E && Next->isDebugInstr())
        ++Next;
      if (Next == E)
        continue;

      // Only insert the CMP if the next instruction consumes EFLAGS.
      if (!usesEFLAGS(*Next))
        continue;

      DebugLoc DL = DecMI.getDebugLoc();

      // Build: CMP [same_mem], 0
      // CMP memory-immediate format: 5 memory operands + 1 immediate.
      auto MIB = BuildMI(MBB, *Next, DL, TII->get(CmpOpc));
      for (unsigned i = 0; i < X86MemOperands; ++i)
        MIB.add(DecMI.getOperand(i));
      MIB.addImm(0);
      MIB.cloneMemRefs(DecMI);

      LLVM_DEBUG(dbgs() << "  Inserted redundant CMP after: " << DecMI);
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86InsertRedundantCmpPass() {
  return new X86InsertRedundantCmpPass();
}

//===--- X86CallTail.cpp - Convert tail jump to call + ret ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: Some MSVC 6.0 functions forward stack arguments to a vtable
// method via call (not jmp):
//
//   mov eax, [ecx]          ; load vtable
//   call [eax + 0xb04]      ; call vtable method
//   ret                     ; return to our caller
//
// With musttail, the compiler generates a tail jump (jmp [eax + 0xb04]).
// This pass converts TAILJMPm back to CALL32m + RET/RETI32 for functions
// with the call_tail attribute, restoring the original call + ret pattern.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86MachineFunctionInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-call-tail"
#define X86_CALL_TAIL_NAME "X86 convert tail jump to call + ret"

namespace {
class X86CallTailPass : public MachineFunctionPass {
public:
  static char ID;
  X86CallTailPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return X86_CALL_TAIL_NAME; }
};
} // end anonymous namespace

char X86CallTailPass::ID = 0;

bool X86CallTailPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("call_tail"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  const X86MachineFunctionInfo *FuncInfo = MF.getInfo<X86MachineFunctionInfo>();
  bool Changed = false;

  // Determine the ret cleanup bytes. Use ret_cleanup_override if present,
  // otherwise use the function's bytesToPopOnReturn (set by calling convention).
  unsigned RetCleanup = FuncInfo->getBytesToPopOnReturn();
  if (MF.getFunction().hasFnAttribute("ret_cleanup_override")) {
    unsigned Override = 0;
    MF.getFunction()
        .getFnAttribute("ret_cleanup_override")
        .getValueAsString()
        .getAsInteger(10, Override);
    RetCleanup = Override;
  }

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E;) {
      MachineInstr &MI = *I++;
      unsigned Opc = MI.getOpcode();

      // Match TAILJMPm (32-bit tail jump through memory).
      // TAILJMPm64 is included for completeness but this project targets 32-bit.
      if (Opc != X86::TAILJMPm && Opc != X86::TAILJMPm64)
        continue;

      DebugLoc DL = MI.getDebugLoc();

      // Build CALL32m (or CALL64m for 64-bit) with the same memory operands.
      unsigned CallOpc = (Opc == X86::TAILJMPm) ? X86::CALL32m : X86::CALL64m;
      MachineInstrBuilder CallMI =
          BuildMI(MBB, MI, DL, TII->get(CallOpc));
      for (unsigned i = 0; i < X86::AddrNumOperands; ++i)
        CallMI.add(MI.getOperand(i));

      // Build the appropriate RET instruction.
      if (RetCleanup > 0) {
        unsigned RetOpc = (Opc == X86::TAILJMPm) ? X86::RETI32 : X86::RETI64;
        BuildMI(MBB, MI, DL, TII->get(RetOpc)).addImm(RetCleanup);
      } else {
        unsigned RetOpc = (Opc == X86::TAILJMPm) ? X86::RET32 : X86::RET64;
        BuildMI(MBB, MI, DL, TII->get(RetOpc));
      }

      // Remove the original TAILJMPm.
      MI.eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86CallTailPass() {
  return new X86CallTailPass();
}

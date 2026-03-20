//===------- X86SwapBufRegister.cpp - Swap EBX<->ESI for buf pointer ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: Post-regalloc pass that swaps two physical registers throughout
// a function. MSVC 6.0 puts pointer parameters in ESI, but LLVM's greedy
// allocator assigns them to EBX due to callee-saved register interference.
// This pass swaps EBX<->ESI in all instructions (including prologue/epilogue)
// after register allocation is complete, achieving the correct assignment.
//
// Gated behind msvc6_regalloc attribute. Only fires when the primary memory
// base register is EBX (detected by counting memory-base uses).
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "MCTargetDesc/X86BaseInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
using namespace llvm;

#define DEBUG_TYPE "x86-swap-buf-register"
#define X86_SWAP_BUF_REGISTER_NAME "X86 swap buf register pass"

namespace {
class X86SwapBufRegisterPass : public MachineFunctionPass {
public:
  static char ID;
  X86SwapBufRegisterPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return X86_SWAP_BUF_REGISTER_NAME; }
};
} // end anonymous namespace

char X86SwapBufRegisterPass::ID = 0;

/// Swap two physical registers in a single operand.
static bool swapRegInOperand(MachineOperand &MO, MCPhysReg RegA,
                             MCPhysReg RegB, MCPhysReg SubA8,
                             MCPhysReg SubB8, MCPhysReg SubA16,
                             MCPhysReg SubB16) {
  if (!MO.isReg())
    return false;
  Register Reg = MO.getReg();
  if (Reg == RegA) { MO.setReg(RegB); return true; }
  if (Reg == RegB) { MO.setReg(RegA); return true; }
  if (Reg == SubA8) { MO.setReg(SubB8); return true; }
  if (Reg == SubB8) { MO.setReg(SubA8); return true; }
  if (Reg == SubA16) { MO.setReg(SubB16); return true; }
  if (Reg == SubB16) { MO.setReg(SubA16); return true; }
  return false;
}

bool X86SwapBufRegisterPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::Msvc6RegAlloc))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  (void)TII;

  // Count memory-base uses per physical register to find the primary base.
  unsigned ebxBaseUses = 0;
  unsigned esiBaseUses = 0;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      int MemOpIdx = X86II::getMemoryOperandNo(MI.getDesc().TSFlags);
      if (MemOpIdx < 0)
        continue;
      MemOpIdx += X86II::getOperandBias(MI.getDesc());
      unsigned BaseIdx = MemOpIdx + X86::AddrBaseReg;
      if (BaseIdx >= MI.getNumOperands())
        continue;
      const MachineOperand &BaseMO = MI.getOperand(BaseIdx);
      if (!BaseMO.isReg())
        continue;
      if (BaseMO.getReg() == X86::EBX)
        ebxBaseUses++;
      else if (BaseMO.getReg() == X86::ESI)
        esiBaseUses++;
    }
  }

  // Only swap if EBX is the primary memory base and ESI is not.
  // This means the allocator put the pointer in EBX when it should be ESI.
  if (ebxBaseUses <= esiBaseUses || ebxBaseUses < 4)
    return false;

  // Swap EBX <-> ESI throughout the entire function.
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      for (MachineOperand &MO : MI.operands()) {
        Changed |= swapRegInOperand(MO, X86::EBX, X86::ESI,
                                    X86::BL, X86::SIL,
                                    X86::BX, X86::SI);
      }
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86SwapBufRegisterPass() {
  return new X86SwapBufRegisterPass();
}

//===--- X86PreferSourceRegisterReuse.cpp - Reuse dying base register -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: Post-regalloc pass that reuses a dying base register as the
// destination of MOV32rm, matching MSVC 6.0 register allocation.
//
// When the last load from a pointer is the only remaining use of that pointer,
// MSVC 6.0 reuses the pointer's register for the loaded value:
//   mov edx, [edx+0x08]    ; pointer destroyed, register reused
//
// Clang allocates a fresh register:
//   mov eax, [edx+0x08]    ; pointer preserved in edx
//
// This pass finds MOV32rm where destReg != baseReg and baseReg is dead after
// the instruction. It changes destReg to baseReg and performs a global swap
// of the old destReg with baseReg for all subsequent uses in the function.
//
// Gate: function attribute "prefer_source_register_reuse".
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "MCTargetDesc/X86BaseInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-source-register-reuse"
#define PASS_NAME "X86 prefer source register reuse"

namespace {
class X86PreferSourceRegisterReusePass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferSourceRegisterReusePass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return PASS_NAME; }
};
} // end anonymous namespace

char X86PreferSourceRegisterReusePass::ID = 0;

/// Map a register from the OldReg32 family to the NewReg32 family, handling
/// 32-bit, 16-bit, and 8-bit sub-register variants. Returns Reg unchanged
/// if it does not belong to either family.
static unsigned mapReg(unsigned Reg, unsigned OldReg32, unsigned NewReg32) {
  if (Reg == OldReg32) return NewReg32;
  if (Reg == NewReg32) return OldReg32;

  // Build sub-register mappings for each family.
  auto getSubRegs = [](unsigned R32, unsigned &R16, unsigned &RLo,
                       unsigned &RHi) {
    switch (R32) {
    case X86::EAX: R16 = X86::AX; RLo = X86::AL; RHi = X86::AH; break;
    case X86::ECX: R16 = X86::CX; RLo = X86::CL; RHi = X86::CH; break;
    case X86::EDX: R16 = X86::DX; RLo = X86::DL; RHi = X86::DH; break;
    case X86::EBX: R16 = X86::BX; RLo = X86::BL; RHi = X86::BH; break;
    case X86::ESI: R16 = X86::SI; RLo = X86::SIL; RHi = 0; break;
    case X86::EDI: R16 = X86::DI; RLo = X86::DIL; RHi = 0; break;
    case X86::EBP: R16 = X86::BP; RLo = X86::BPL; RHi = 0; break;
    case X86::ESP: R16 = X86::SP; RLo = X86::SPL; RHi = 0; break;
    default:       R16 = 0;       RLo = 0;         RHi = 0; break;
    }
  };

  unsigned Old16, OldLo, OldHi;
  unsigned New16, NewLo, NewHi;
  getSubRegs(OldReg32, Old16, OldLo, OldHi);
  getSubRegs(NewReg32, New16, NewLo, NewHi);

  if (Reg == Old16 && New16) return New16;
  if (Reg == New16 && Old16) return Old16;
  if (Reg == OldLo && NewLo) return NewLo;
  if (Reg == NewLo && OldLo) return OldLo;
  if (OldHi && Reg == OldHi && NewHi) return NewHi;
  if (NewHi && Reg == NewHi && OldHi) return OldHi;

  return Reg;
}

/// Check whether Reg (or any of its sub/super registers) is used after MI
/// within the same basic block. Returns true if the register is dead.
static bool isRegDeadAfter(const MachineInstr &MI, unsigned Reg,
                           const TargetRegisterInfo *TRI) {
  const MachineBasicBlock &MBB = *MI.getParent();
  auto It = MI.getIterator();
  ++It;
  for (auto E = MBB.end(); It != E; ++It) {
    for (const MachineOperand &MO : It->operands()) {
      if (!MO.isReg())
        continue;
      if (TRI->regsOverlap(MO.getReg(), Reg)) {
        return false; // Register is used later.
      }
    }
  }
  // Also check if the register is live-out of this block.
  for (const MachineBasicBlock *Succ : MBB.successors()) {
    if (Succ->isLiveIn(Reg))
      return false;
  }
  return true;
}

bool X86PreferSourceRegisterReusePass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("prefer_source_register_reuse"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  bool Changed = false;

  // Process each basic block independently.
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.getOpcode() != X86::MOV32rm)
        continue;

      // Get memory operand location.
      int MemOpIdx = X86II::getMemoryOperandNo(MI.getDesc().TSFlags);
      if (MemOpIdx < 0)
        continue;
      MemOpIdx += X86II::getOperandBias(MI.getDesc());
      unsigned BaseIdx = MemOpIdx + X86::AddrBaseReg;
      if (BaseIdx >= MI.getNumOperands())
        continue;

      const MachineOperand &BaseMO = MI.getOperand(BaseIdx);
      if (!BaseMO.isReg() || BaseMO.getReg() == 0)
        continue;

      Register DestReg = MI.getOperand(0).getReg();
      Register BaseReg = BaseMO.getReg();

      // Only act when dest != base.
      if (DestReg == BaseReg)
        continue;

      // Only handle GPR32 registers.
      if (!X86::GR32RegClass.contains(DestReg) ||
          !X86::GR32RegClass.contains(BaseReg))
        continue;

      // Do not touch ESP - it is the stack pointer.
      if (DestReg == X86::ESP || BaseReg == X86::ESP)
        continue;

      // Check if BaseReg is dead after this instruction.
      if (!isRegDeadAfter(MI, BaseReg, TRI))
        continue;

      // Also verify the index register is not the same as base (rare but
      // possible). If the index register overlaps with BaseReg, skip.
      unsigned IndexIdx = MemOpIdx + X86::AddrIndexReg;
      if (IndexIdx < MI.getNumOperands()) {
        const MachineOperand &IndexMO = MI.getOperand(IndexIdx);
        if (IndexMO.isReg() && IndexMO.getReg() != 0 &&
            TRI->regsOverlap(IndexMO.getReg(), BaseReg))
          continue;
      }

      // Perform the swap: change destReg to baseReg in this instruction and
      // swap old destReg <-> baseReg in all subsequent instructions in the
      // entire function (from after this instruction to function end).
      // First, update the defining instruction itself.
      MI.getOperand(0).setReg(BaseReg);

      // Swap destReg <-> baseReg in all instructions after MI in the function.
      bool StartedSwapping = false;
      for (MachineBasicBlock &SwapMBB : MF) {
        for (MachineInstr &SwapMI : SwapMBB) {
          if (!StartedSwapping) {
            if (&SwapMI == &MI) {
              StartedSwapping = true;
            }
            continue;
          }
          for (MachineOperand &MO : SwapMI.operands()) {
            if (!MO.isReg())
              continue;
            unsigned NewReg = mapReg(MO.getReg(), DestReg, BaseReg);
            if (NewReg != MO.getReg())
              MO.setReg(NewReg);
          }
        }
      }

      Changed = true;
      // After swapping, the register names are different, so break and
      // restart the block scan to avoid confusion. Since this is a rare
      // transformation (typically 0-1 per function), this is fine.
      break;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferSourceRegisterReusePass() {
  return new X86PreferSourceRegisterReusePass();
}

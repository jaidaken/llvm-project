//===--- X86SuppressMovzx.cpp - Remove XOR before bare subreg loads -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: After X86ExpandMovzx converts MOVZX to XOR+MOV, MSVC 6.0
// sometimes emits just the bare MOV (e.g., `mov dl, [mem]`) without the
// preceding XOR, when the upper bits are never read.
//
// This pass removes the XOR in `xor reg,reg; mov subreg,[mem]` sequences
// when the full 32-bit register is dead after the subreg use. This produces
// the bare partial-register load that MSVC 6.0 generates.
//
// Examples:
//   Before: xor edx,edx; mov dl,[ecx+0xf0]; test dl,dl; setne al
//   After:  mov dl,[ecx+0xf0]; test dl,dl; setne al
//
//   Before: xor edx,edx; mov dx,[ecx+0x10]; test dx,dx; je .LBB0_1
//   After:  mov dx,[ecx+0x10]; test dx,dx; je .LBB0_1
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "x86-suppress-movzx"

namespace {
class X86SuppressMovzxPass : public MachineFunctionPass {
public:
  static char ID;
  X86SuppressMovzxPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 suppress MOVZX zero-extend";
  }
};
} // end anonymous namespace

char X86SuppressMovzxPass::ID = 0;

/// Check if MI is a self-XOR zeroing idiom (xor reg,reg or xor_rev reg,reg).
static bool isXorZero(const MachineInstr &MI, Register &Reg) {
  if (MI.getOpcode() != X86::XOR32rr && MI.getOpcode() != X86::XOR32rr_REV)
    return false;
  if (MI.getOperand(0).getReg() != MI.getOperand(1).getReg())
    return false;
  Reg = MI.getOperand(0).getReg();
  return true;
}

/// Check if MI is a byte or word load into a subreg of FullReg.
/// Returns the X86 subreg index (sub_8bit or sub_16bit) via SubRegIdx,
/// or 0 if the instruction doesn't match.
static unsigned isSubregLoad(const MachineInstr &MI, Register FullReg,
                             const TargetRegisterInfo *TRI) {
  unsigned SubIdx = 0;
  if (MI.getOpcode() == X86::MOV8rm)
    SubIdx = X86::sub_8bit;
  else if (MI.getOpcode() == X86::MOV16rm)
    SubIdx = X86::sub_16bit;
  else
    return 0;

  Register Dst = MI.getOperand(0).getReg();
  if (TRI->regsOverlap(Dst, FullReg) &&
      Dst == TRI->getSubReg(FullReg, SubIdx))
    return SubIdx;
  return 0;
}

/// Walk forward from the subreg load to check if the full 32-bit register
/// is ever read before being overwritten. If only the loaded subreg (8-bit
/// or 16-bit) is used, the XOR is unnecessary.
///
/// SubReg is the loaded subreg register (e.g., DL for 8-bit, DX for 16-bit).
/// SubRegIdx is the subreg index (sub_8bit or sub_16bit) so we know which
/// smaller subregs are safe to read.
static bool isFullRegDeadAfterSubregUse(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator AfterLoad,
                                         Register FullReg, Register SubReg,
                                         unsigned SubRegIdx,
                                         const TargetRegisterInfo *TRI) {
  // Walk forward through the block.
  for (auto I = AfterLoad, E = MBB.end(); I != E; ++I) {
    MachineInstr &MI = *I;

    // Skip debug/pseudo instructions.
    if (MI.isDebugInstr())
      continue;

    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || MO.getReg() == 0)
        continue;

      Register Reg = MO.getReg();

      // If this instruction defines (overwrites) the full register,
      // the upper bits are dead - the XOR was unnecessary.
      if (MO.isDef() && TRI->regsOverlap(Reg, FullReg))
        return true;

      // If this instruction reads a register overlapping FullReg,
      // determine if it only touches bits covered by the loaded subreg.
      if (MO.isUse() && TRI->regsOverlap(Reg, FullReg)) {
        // Reading exactly the loaded subreg is always safe.
        if (Reg == SubReg)
          continue;

        // For 16-bit loads, the 8-bit low subreg (e.g., DL) is also safe
        // since it's fully contained within the loaded 16-bit subreg.
        if (SubRegIdx == X86::sub_16bit) {
          Register Sub8 = TRI->getSubReg(FullReg, X86::sub_8bit);
          if (Reg == Sub8)
            continue;
        }

        // Any other overlapping read (the full reg, a wider subreg, or
        // a partially-overlapping subreg like DH) means the upper bits
        // that the XOR would have zeroed are live. Keep the XOR.
        return false;
      }
    }

    // At a terminator or call, conservatively assume the register may
    // be live-out.
    if (MI.isTerminator() || MI.isCall())
      return false;
  }

  // Reached end of block without the full reg being overwritten.
  // The register might be live-out. Conservative: keep the XOR.
  return false;
}

bool X86SuppressMovzxPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("suppress_movzx_zero"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E; ) {
      MachineInstr &MI = *I;

      // Look for XOR32rr/XOR32rr_REV self-zeroing.
      Register XorReg;
      if (!isXorZero(MI, XorReg)) {
        ++I;
        continue;
      }

      // Check if the next non-debug instruction is a byte or word load
      // into the subreg of the zeroed register.
      auto Next = std::next(I);
      while (Next != E && Next->isDebugInstr())
        ++Next;

      unsigned SubRegIdx = 0;
      if (Next != E)
        SubRegIdx = isSubregLoad(*Next, XorReg, TRI);

      if (!SubRegIdx) {
        ++I;
        continue;
      }

      Register SubReg = TRI->getSubReg(XorReg, SubRegIdx);
      auto AfterLoad = std::next(Next);

      // Check that the full register is dead after the subreg use.
      if (!isFullRegDeadAfterSubregUse(MBB, AfterLoad, XorReg, SubReg,
                                        SubRegIdx, TRI)) {
        ++I;
        continue;
      }

      LLVM_DEBUG(dbgs() << "SuppressMovzx: removing XOR before subreg load in "
                        << MF.getName() << "\n");

      // Remove the XOR. Keep the subreg load.
      auto NextI = std::next(I);
      MI.eraseFromParent();
      I = NextI;
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86SuppressMovzxPass() {
  return new X86SuppressMovzxPass();
}

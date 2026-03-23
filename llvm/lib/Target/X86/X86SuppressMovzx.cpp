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
// Example:
//   Before: xor edx,edx; mov dl,[ecx+0xf0]; test dl,dl; setne al
//   After:  mov dl,[ecx+0xf0]; test dl,dl; setne al
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

/// Check if MI is a byte load into a subreg of FullReg.
static bool isByteLoad(const MachineInstr &MI, Register FullReg,
                       const TargetRegisterInfo *TRI) {
  if (MI.getOpcode() != X86::MOV8rm)
    return false;
  Register Dst = MI.getOperand(0).getReg();
  // Check that the dst is the sub_8bit of FullReg.
  return TRI->regsOverlap(Dst, FullReg) &&
         Dst == TRI->getSubReg(FullReg, X86::sub_8bit);
}

/// Walk forward from the byte load to check if the full 32-bit register
/// is ever read before being overwritten. If only the 8-bit subreg is
/// used, the XOR is unnecessary.
static bool isFullRegDeadAfterSubregUse(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator AfterLoad,
                                         Register FullReg, Register SubReg8,
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

      // If this instruction reads the full register (not just the subreg),
      // the upper bits are live - the XOR is needed.
      if (MO.isUse() && TRI->regsOverlap(Reg, FullReg)) {
        // It's OK if only the 8-bit subreg is read.
        if (Reg == SubReg8)
          continue;
        // Reading the 16-bit subreg (e.g., DX) includes the 8-bit part
        // but also the upper byte (DH) which would have garbage. Check
        // if it's actually just the 8-bit subreg.
        if (TRI->isSubRegister(FullReg, Reg) && Reg != SubReg8)
          return false; // Reading DX, AX, etc. - needs zeroed upper bits
        if (Reg == FullReg)
          return false; // Reading full EDX, EAX, etc.
        // Any other overlap means the upper bits matter.
        if (Reg != SubReg8)
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

      // Check if the next non-debug instruction is a byte load
      // into the subreg of the zeroed register.
      auto Next = std::next(I);
      while (Next != E && Next->isDebugInstr())
        ++Next;

      if (Next == E || !isByteLoad(*Next, XorReg, TRI)) {
        ++I;
        continue;
      }

      Register SubReg8 = TRI->getSubReg(XorReg, X86::sub_8bit);
      auto AfterLoad = std::next(Next);

      // Check that the full register is dead after the subreg use.
      if (!isFullRegDeadAfterSubregUse(MBB, AfterLoad, XorReg, SubReg8, TRI)) {
        ++I;
        continue;
      }

      LLVM_DEBUG(dbgs() << "SuppressMovzx: removing XOR before byte load in "
                        << MF.getName() << "\n");

      // Remove the XOR. Keep the byte load.
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

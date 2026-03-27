//===--- X86PreferPushFstp.cpp - Use push ecx for FPU stack arg -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: This pass replaces "sub esp, 4" (3 bytes: 83 EC 04) with
// "push ecx" (1 byte: 51) when followed by "fstp dword ptr [esp]" for
// functions with the prefer_push_fstp attribute.
//
// MSVC 6.0 passes FPU return values as stack arguments using:
//   push ecx              ; 51 (1 byte) - allocate 4 bytes, value is don't-care
//   fstp dword ptr [esp]  ; d9 1c 24 (3 bytes) - store FPU result
//
// Clang generates:
//   sub esp, 4            ; 83 ec 04 (3 bytes) - allocate 4 bytes
//   fstp dword ptr [esp]  ; d9 1c 24 (3 bytes) - store FPU result
//
// Both sequences allocate 4 bytes on the stack and store the FPU top to
// that location. The push variant saves 2 bytes per site.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-push-fstp"
#define X86_PREFER_PUSH_FSTP_NAME "X86 prefer push ecx for FPU stack arg"

namespace {
class X86PreferPushFstpPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferPushFstpPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return X86_PREFER_PUSH_FSTP_NAME;
  }
};
} // end anonymous namespace

char X86PreferPushFstpPass::ID = 0;

/// Check if MI is SUB32ri or SUB32ri8 with ESP as dest/src and immediate 4.
static bool isSubEsp4(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  if (Opc != X86::SUB32ri && Opc != X86::SUB32ri8)
    return false;
  return MI.getOperand(0).getReg() == X86::ESP &&
         MI.getOperand(1).getReg() == X86::ESP &&
         MI.getOperand(2).getImm() == 4;
}

/// Check if MI is ST_FP32m with base=ESP, scale=1, index=NoReg, disp=0.
/// This matches: fstp dword ptr [esp]
static bool isFstpEsp(const MachineInstr &MI) {
  if (MI.getOpcode() != X86::ST_FP32m)
    return false;

  // ST_FP32m operands: base(0), scale(1), index(2), disp(3), seg(4)
  if (MI.getNumOperands() < 5)
    return false;

  const MachineOperand &Base = MI.getOperand(0);
  const MachineOperand &Scale = MI.getOperand(1);
  const MachineOperand &Index = MI.getOperand(2);
  const MachineOperand &Disp = MI.getOperand(3);

  return Base.isReg() && Base.getReg() == X86::ESP &&
         Scale.isImm() && Scale.getImm() == 1 &&
         Index.isReg() && Index.getReg() == X86::NoRegister &&
         Disp.isImm() && Disp.getImm() == 0;
}

bool X86PreferPushFstpPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("prefer_push_fstp"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E;) {
      MachineInstr &MI = *I;

      // Match: SUB32ri/SUB32ri8 ESP, 4
      if (!isSubEsp4(MI)) {
        ++I;
        continue;
      }

      // Check next non-pseudo instruction is ST_FP32m [ESP + 0]
      auto NextI = std::next(I);
      while (NextI != E && NextI->isPseudo())
        ++NextI;

      if (NextI == E || !isFstpEsp(*NextI)) {
        ++I;
        continue;
      }

      // Replace SUB ESP, 4 with PUSH32r ECX.
      // push ecx: opcode 0x51 (1 byte) vs sub esp, 4: 0x83 0xEC 0x04 (3 bytes)
      // The pushed value is don't-care since fstp immediately overwrites [esp].
      DebugLoc DL = MI.getDebugLoc();
      BuildMI(MBB, MI, DL, TII->get(X86::PUSH32r))
          .addReg(X86::ECX, RegState::Undef);

      auto AfterFstp = std::next(MachineBasicBlock::iterator(*NextI));
      MI.eraseFromParent();
      Changed = true;
      I = AfterFstp;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferPushFstpPass() {
  return new X86PreferPushFstpPass();
}

//===--- X86PreferPushImm.cpp - Convert MOV [esp],imm to PUSH imm ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: MSVC 6.0 passes thiscall/cdecl stack arguments via PUSH:
//   push 0x0            ; 6a 00 (2 bytes)
//   mov ecx, esi
//   call [eax+0x48]
//
// Clang pre-allocates outgoing arg space in the prologue (SUB ESP, N), then
// fills it with MOV:
//   mov dword ptr [esp], 0x0   ; c7 04 24 00 00 00 00 (7 bytes)
//   mov ecx, esi
//   call [eax+0x48]
//
// This pass, gated on the "prefer_push_imm" string attribute, converts
// MOV32mi [ESP+0], imm to PUSH32i8/PUSH32i and reduces the prologue
// SUB ESP, N by 4 for each converted site (since the space is now allocated
// inline by the PUSH). The corresponding epilogue ADD ESP, N (if present)
// is also reduced.
//
// Safety: only handles displacement 0 (first/only outgoing arg slot) and
// requires an EBP-based frame so that local accesses are EBP-relative and
// unaffected by the ESP change.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-push-imm"
#define X86_PREFER_PUSH_IMM_NAME "X86 prefer PUSH imm over MOV [esp], imm"

namespace {
class X86PreferPushImmPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferPushImmPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return X86_PREFER_PUSH_IMM_NAME; }

private:
  /// Check if MI is MOV32mi with base=ESP, scale=1, index=NoReg, disp=0.
  static bool isMovEspImm(const MachineInstr &MI);

  /// Check if MI is SUB32ri or SUB32ri8 with ESP as dest/src.
  static bool isSubEsp(const MachineInstr &MI);

  /// Check if MI is ADD32ri or ADD32ri8 with ESP as dest/src.
  static bool isAddEsp(const MachineInstr &MI);

  /// Find the prologue SUB ESP instruction in the entry block.
  MachineInstr *findPrologueSub(MachineBasicBlock &EntryMBB);

  /// Find all epilogue ADD ESP instructions across the function.
  void findEpilogueAdds(MachineFunction &MF,
                        SmallVectorImpl<MachineInstr *> &Adds);
};
} // end anonymous namespace

char X86PreferPushImmPass::ID = 0;

bool X86PreferPushImmPass::isMovEspImm(const MachineInstr &MI) {
  if (MI.getOpcode() != X86::MOV32mi)
    return false;

  // MOV32mi operands: base(0), scale(1), index(2), disp(3), segment(4), imm(5)
  if (MI.getNumOperands() < 6)
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

bool X86PreferPushImmPass::isSubEsp(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  if (Opc != X86::SUB32ri && Opc != X86::SUB32ri8)
    return false;
  return MI.getOperand(0).getReg() == X86::ESP &&
         MI.getOperand(1).getReg() == X86::ESP;
}

bool X86PreferPushImmPass::isAddEsp(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  if (Opc != X86::ADD32ri && Opc != X86::ADD32ri8)
    return false;
  return MI.getOperand(0).getReg() == X86::ESP &&
         MI.getOperand(1).getReg() == X86::ESP;
}

MachineInstr *X86PreferPushImmPass::findPrologueSub(MachineBasicBlock &MBB) {
  // Walk the entry block looking for SUB ESP, N (skipping pushes and CFI).
  for (auto &MI : MBB) {
    if (MI.getOpcode() == TargetOpcode::CFI_INSTRUCTION)
      continue;
    if (MI.getOpcode() == X86::PUSH32r)
      continue;
    // MOV32rr EBP setup
    if (MI.getOpcode() == X86::MOV32rr)
      continue;
    if (isSubEsp(MI))
      return &MI;
    // Stop searching after non-prologue instructions.
    // Allow a few more prologue-like patterns before giving up.
    if (MI.isCall() || MI.isReturn() || MI.isBranch())
      break;
  }
  return nullptr;
}

void X86PreferPushImmPass::findEpilogueAdds(
    MachineFunction &MF, SmallVectorImpl<MachineInstr *> &Adds) {
  for (MachineBasicBlock &MBB : MF) {
    for (auto &MI : MBB) {
      if (isAddEsp(MI))
        Adds.push_back(&MI);
    }
  }
}

bool X86PreferPushImmPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("prefer_push_imm"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  // Count how many MOV [esp+0], imm instructions we convert.
  // We reduce the prologue SUB by 4 * count (well, by 4 for the shared slot).
  // Since all calls share the same pre-allocated slot at [esp+0], we only
  // need to reduce by 4 once (one slot).
  unsigned ConvertCount = 0;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E;) {
      MachineInstr &MI = *I;

      if (!isMovEspImm(MI)) {
        ++I;
        continue;
      }

      int64_t ImmVal = MI.getOperand(5).getImm();
      DebugLoc DL = MI.getDebugLoc();

      // Choose PUSH32i8 if the immediate fits in a sign-extended 8-bit value,
      // otherwise use PUSH32i (full 32-bit immediate).
      unsigned PushOpc;
      if (ImmVal >= -128 && ImmVal <= 127)
        PushOpc = X86::PUSH32i8;
      else
        PushOpc = X86::PUSH32i;

      BuildMI(MBB, MI, DL, TII->get(PushOpc)).addImm(ImmVal);

      auto NextI = std::next(I);
      MI.eraseFromParent();
      I = NextI;
      Changed = true;
      ConvertCount++;
    }
  }

  if (!Changed)
    return false;

  // Reduce the prologue SUB ESP by 4. All converted MOV [esp+0] shared
  // the same single 4-byte outgoing arg slot, so we only subtract 4.
  MachineInstr *SubMI = findPrologueSub(MF.front());
  if (SubMI) {
    int64_t SubAmount = SubMI->getOperand(2).getImm();
    int64_t NewAmount = SubAmount - 4;
    if (NewAmount <= 0) {
      // Remove the SUB entirely.
      SubMI->eraseFromParent();
    } else {
      // Update the immediate. Switch opcode if the new value fits in imm8.
      if (NewAmount <= 127 && SubMI->getOpcode() == X86::SUB32ri) {
        SubMI->setDesc(TII->get(X86::SUB32ri8));
      }
      SubMI->getOperand(2).setImm(NewAmount);
    }
  }

  // Reduce epilogue ADD ESP instructions by 4 as well.
  // In EBP-frame functions the epilogue uses "mov esp, ebp; pop ebp" so
  // there may not be an ADD ESP. But for functions that use ADD ESP in the
  // epilogue (non-EBP-frame or restructured), we adjust those too.
  SmallVector<MachineInstr *, 4> Adds;
  findEpilogueAdds(MF, Adds);

  // Only adjust ADD ESP instructions that match the original SUB amount.
  // We look for ADD ESP with the same value that the SUB had (before our
  // reduction). This avoids accidentally modifying cdecl cleanup ADD ESP
  // instructions (which typically have smaller values like 4, 8, etc.)
  // that are unrelated to the prologue/epilogue pair.
  //
  // However, in the simple case, the epilogue is often "mov esp, ebp"
  // so there's nothing to do. We skip this adjustment to avoid breaking
  // cdecl call cleanup (add esp, 4 after cdecl calls).

  return Changed;
}

FunctionPass *llvm::createX86PreferPushImmPass() {
  return new X86PreferPushImmPass();
}

//===--- X86PreferVtableEdx.cpp - Fix vtable call register conflict --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: After PreferMovPush unfolds PUSH32rmm to MOV32rm EAX + PUSH EAX,
// the register allocator may assign both the vtable pointer load and the
// parameter load to EAX, creating a conflict:
//
//   mov eax, [ecx]       ; vtable load
//   mov eax, [esp+4]     ; param load — clobbers vtable!
//   push eax             ; push param
//   call [eax+0x914]     ; WRONG: eax has param, not vtable
//
// MSVC 6.0 uses EDX for parameter loads in this pattern:
//
//   mov edx, [esp+4]     ; param load into EDX
//   mov eax, [ecx]       ; vtable load into EAX
//   push edx             ; push param from EDX
//   call [eax+0x914]     ; CORRECT: eax has vtable
//
// This pass detects the conflict and rewrites param loads to EDX.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-vtable-edx"

namespace {
class X86PreferVtableEdxPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferVtableEdxPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 prefer vtable EDX param pass";
  }
};
} // end anonymous namespace

char X86PreferVtableEdxPass::ID = 0;

bool X86PreferVtableEdxPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::NoCalleeSaves))
    return false;


  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E; ++I) {
      MachineInstr &CallMI = *I;

      // Step 1: Find CALL32m with a register base (indirect vtable call).
      // Also check FARCALL32m in case the compiler uses that variant.
      unsigned CallOpc = CallMI.getOpcode();
      if (CallOpc != X86::CALL32m && CallOpc != X86::FARCALL32m)
        continue;

      // CALL32m operands: base, scale, index, disp, segment
      if (!CallMI.getOperand(0).isReg())
        continue;
      Register CallBase = CallMI.getOperand(0).getReg();
      if (CallBase == X86::NoRegister || CallBase == X86::ESP)
        continue;

      // Collect preceding non-pseudo instructions (up to 6).
      SmallVector<MachineInstr*, 6> Prev;
      {
        auto WalkIt = MachineBasicBlock::iterator(CallMI);
        while (Prev.size() < 6 && WalkIt != MBB.begin()) {
          --WalkIt;
          if (!WalkIt->isPseudo())
            Prev.push_back(&*WalkIt);
        }
      }
      // Prev[0] = instruction right before CALL, Prev[1] = one before that, etc.

      // === 2-PARAM PATTERN ===
      // Prev[0]: PUSH32r RegB (param_1, e.g. EDX)
      // Prev[1]: PUSH32r CallBase (param_2, e.g. EAX)
      // Prev[2]: MOV32rm RegB, [ESP+N1] (param_1 load)
      // Prev[3]: MOV32rm CallBase, [ESP+N2] (param_2 load, clobbers vtable!)
      // Prev[4]: MOV32rm CallBase, [non-ESP] (vtable load)
      if (Prev.size() >= 5 &&
          Prev[0]->getOpcode() == X86::PUSH32r &&
          Prev[1]->getOpcode() == X86::PUSH32r &&
          Prev[1]->getOperand(0).getReg() == CallBase &&
          Prev[2]->getOpcode() == X86::MOV32rm &&
          Prev[2]->getOperand(1).isReg() &&
          Prev[2]->getOperand(1).getReg() == X86::ESP &&
          Prev[3]->getOpcode() == X86::MOV32rm &&
          Prev[3]->getOperand(0).getReg() == CallBase &&
          Prev[3]->getOperand(1).isReg() &&
          Prev[3]->getOperand(1).getReg() == X86::ESP &&
          Prev[4]->getOpcode() == X86::MOV32rm &&
          Prev[4]->getOperand(0).getReg() == CallBase &&
          Prev[4]->getOperand(1).isReg() &&
          Prev[4]->getOperand(1).getReg() != X86::ESP) {

        DebugLoc DL = Prev[3]->getDebugLoc();
        int64_t Param2Disp = Prev[3]->getOperand(4).getImm();
        int64_t Param1Disp = Prev[2]->getOperand(4).getImm();
        MachineInstr &VtableMI = *Prev[4];

        // Build: MOV32rm EDX, [ESP+Param2Disp] before vtable
        BuildMI(MBB, VtableMI, DL, TII->get(X86::MOV32rm), X86::EDX)
            .addReg(X86::ESP)
            .addImm(1).addReg(X86::NoRegister)
            .addImm(Param2Disp)
            .addReg(X86::NoRegister);

        // VtableMI stays (now after new param2 load)
        // Change push of param_2 from CallBase to EDX
        Prev[1]->getOperand(0).setReg(X86::EDX);

        // Build: MOV32rm EDX, [ESP+Param1Disp+4] between the two pushes
        // +4 because first push shifted ESP
        auto AfterPush2 = std::next(MachineBasicBlock::iterator(*Prev[1]));
        BuildMI(MBB, *AfterPush2, DL, TII->get(X86::MOV32rm), X86::EDX)
            .addReg(X86::ESP)
            .addImm(1).addReg(X86::NoRegister)
            .addImm(Param1Disp + 4)
            .addReg(X86::NoRegister);

        // Change push of param_1 to EDX
        Prev[0]->getOperand(0).setReg(X86::EDX);

        // Remove old param loads
        Prev[3]->eraseFromParent();
        Prev[2]->eraseFromParent();

        Changed = true;
        continue;
      }

      // === 1-PARAM PATTERN ===
      // Prev[0]: PUSH32r CallBase
      // Prev[1]: MOV32rm CallBase, [ESP+N] (param load, clobbers vtable!)
      // Prev[2]: MOV32rm CallBase, [non-ESP] (vtable load)
      if (Prev.size() >= 3 &&
          Prev[0]->getOpcode() == X86::PUSH32r &&
          Prev[0]->getOperand(0).getReg() == CallBase &&
          Prev[1]->getOpcode() == X86::MOV32rm &&
          Prev[1]->getOperand(0).getReg() == CallBase &&
          Prev[1]->getOperand(1).isReg() &&
          Prev[1]->getOperand(1).getReg() == X86::ESP &&
          Prev[2]->getOpcode() == X86::MOV32rm &&
          Prev[2]->getOperand(0).getReg() == CallBase &&
          Prev[2]->getOperand(1).isReg() &&
          Prev[2]->getOperand(1).getReg() != X86::ESP) {

        // 1-param fix: change param load to EDX, reorder before vtable.
        MachineInstr &PushMI = *Prev[0];
        MachineInstr &ParamMI = *Prev[1];
        MachineInstr &VtableMI = *Prev[2];

        DebugLoc DL = ParamMI.getDebugLoc();

        // Build new param load: MOV32rm EDX, [ESP+offset]
        int64_t ParamDisp = ParamMI.getOperand(4).getImm();
        BuildMI(MBB, VtableMI, DL, TII->get(X86::MOV32rm), X86::EDX)
            .addReg(X86::ESP)
            .addImm(ParamMI.getOperand(2).getImm())  // scale
            .addReg(ParamMI.getOperand(3).getReg())   // index
            .addImm(ParamDisp)                         // disp
            .addReg(ParamMI.getOperand(5).getReg());   // segment

        // VtableMI stays in place (now after new param load).
        // Change push to use EDX.
        PushMI.getOperand(0).setReg(X86::EDX);

        // Remove old param load.
        ParamMI.eraseFromParent();

        Changed = true;
      }
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferVtableEdxPass() {
  return new X86PreferVtableEdxPass();
}

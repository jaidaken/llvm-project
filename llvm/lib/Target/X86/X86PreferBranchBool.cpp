//===--- X86PreferBranchBool.cpp - Convert setcc bool to branch pattern ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: MSVC 6.0 generates branch-based boolean returns:
//   test/cmp ...; je .L0; mov eax, 1; ret; .L0: xor eax, eax
//
// The compiler generates setcc-based boolean returns:
//   test/cmp ...; xor ecx, ecx; setne cl; mov eax, ecx; ret
//
// This pass converts the setcc pattern to the branch pattern by:
// 1. Detecting: XOR32rr reg,reg; SETCCr reg_lo, CC; MOV32rr EAX, reg; RET
// 2. Replacing with: JCC_1 .Lfalse, inverted(CC); MOV32ri EAX, 1; RET;
//    .Lfalse: XOR32rr EAX, EAX; (continues to next instruction/ret)
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-branch-bool"

namespace {
class X86PreferBranchBoolPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferBranchBoolPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 prefer branch-based boolean return";
  }
};
} // end anonymous namespace

char X86PreferBranchBoolPass::ID = 0;

/// Invert a condition code.
static X86::CondCode invertCC(X86::CondCode CC) {
  switch (CC) {
  case X86::COND_E:  return X86::COND_NE;
  case X86::COND_NE: return X86::COND_E;
  case X86::COND_A:  return X86::COND_BE;
  case X86::COND_AE: return X86::COND_B;
  case X86::COND_B:  return X86::COND_AE;
  case X86::COND_BE: return X86::COND_A;
  case X86::COND_G:  return X86::COND_LE;
  case X86::COND_GE: return X86::COND_L;
  case X86::COND_L:  return X86::COND_GE;
  case X86::COND_LE: return X86::COND_G;
  default: return X86::COND_INVALID;
  }
}

/// Get the 8-bit subreg for a 32-bit reg (e.g. ECX -> CL)
static MCPhysReg get8bitSubReg(MCPhysReg Reg) {
  switch (Reg) {
  case X86::EAX: return X86::AL;
  case X86::ECX: return X86::CL;
  case X86::EDX: return X86::DL;
  case X86::EBX: return X86::BL;
  default: return 0;
  }
}

bool X86PreferBranchBoolPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::Msvc6RegAlloc))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E; ) {
      // Pattern: XOR32rr reg, reg; SETCCr reg_lo, CC; MOV32rr EAX, reg; RET/RETI
      MachineInstr &Xor = *I;
      if (Xor.getOpcode() != X86::XOR32rr &&
          Xor.getOpcode() != X86::XOR32rr_REV) {
        ++I;
        continue;
      }
      if (Xor.getOperand(1).getReg() != Xor.getOperand(2).getReg()) {
        ++I;
        continue;
      }

      Register ZeroReg = Xor.getOperand(0).getReg();
      MCPhysReg ZeroReg8 = get8bitSubReg(ZeroReg);
      if (!ZeroReg8) { ++I; continue; }

      auto SetI = std::next(I);
      if (SetI == E) { ++I; continue; }

      if (SetI->getOpcode() != X86::SETCCr) { ++I; continue; }
      if (SetI->getOperand(0).getReg() != ZeroReg8) { ++I; continue; }

      X86::CondCode CC = X86::getCondFromSETCC(*SetI);
      if (CC == X86::COND_INVALID) { ++I; continue; }

      auto MovI = std::next(SetI);
      if (MovI == E) { ++I; continue; }

      // Match MOV32rr EAX, reg or MOV32rr_REV EAX, reg
      if (MovI->getOpcode() != X86::MOV32rr &&
          MovI->getOpcode() != X86::MOV32rr_REV) {
        ++I;
        continue;
      }
      if (MovI->getOperand(0).getReg() != X86::EAX) { ++I; continue; }
      if (MovI->getOperand(1).getReg() != ZeroReg) { ++I; continue; }

      auto RetI = std::next(MovI);
      if (RetI == E) { ++I; continue; }

      bool IsRet = (RetI->getOpcode() == X86::RET ||
                    RetI->getOpcode() == X86::RET32 ||
                    RetI->getOpcode() == X86::RETI32 ||
                    RetI->isReturn());
      if (!IsRet) { ++I; continue; }

      // We have the full pattern. Rewrite to branch form.
      // Create a new MBB for the "false" (xor eax,eax) case.
      X86::CondCode InvCC = invertCC(CC);
      if (InvCC == X86::COND_INVALID) { ++I; continue; }

      DebugLoc DL = Xor.getDebugLoc();

      // Create false block after current block
      MachineBasicBlock *FalseMBB = MF.CreateMachineBasicBlock();
      FalseMBB->setLabelMustBeEmitted();
      MF.insert(std::next(MachineFunction::iterator(&MBB)), FalseMBB);

      // Move the RET to the false block (it will follow xor eax,eax)
      // But first, clone the ret for the true path
      unsigned RetOpc = RetI->getOpcode();

      // Build in current block: JCC FalseMBB, inverted_CC; MOV32ri EAX, 1; RET
      BuildMI(MBB, Xor, DL, TII->get(X86::JCC_1))
          .addMBB(FalseMBB)
          .addImm(InvCC);
      BuildMI(MBB, Xor, DL, TII->get(X86::MOV32ri), X86::EAX)
          .addImm(1);
      // Clone the RET for the true path (before the xor/setcc/mov we'll delete)
      {
        auto TrueRet = BuildMI(MBB, Xor, DL, TII->get(RetI->getOpcode()));
        for (const auto &MO : RetI->operands())
          TrueRet.add(MO);
      }

      // Build in false block: XOR32rr EAX, EAX
      // The false block falls through or has its own ret
      BuildMI(*FalseMBB, FalseMBB->end(), DL,
              TII->get(X86::XOR32rr_REV), X86::EAX)
          .addReg(X86::EAX, RegState::Undef)
          .addReg(X86::EAX, RegState::Undef);

      // Transfer successors from current block to false block
      // and add FalseMBB as successor of current block
      FalseMBB->transferSuccessorsAndUpdatePHIs(&MBB);
      MBB.addSuccessor(FalseMBB);

      // Move the RET and everything after it to the false block
      FalseMBB->splice(FalseMBB->end(), &MBB, RetI, MBB.end());

      // Remove old xor, setcc, mov (they're now replaced by jcc + mov eax,1)
      MovI->eraseFromParent();
      SetI->eraseFromParent();
      I = Xor.getIterator();
      auto NextI = std::next(I);
      Xor.eraseFromParent();
      I = NextI;

      Changed = true;
      break; // restart the block scan since we modified the block
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferBranchBoolPass() {
  return new X86PreferBranchBoolPass();
}

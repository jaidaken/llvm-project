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
      // Pattern 1: XOR32rr reg, reg; SETCCr reg_lo; MOV32rr EAX, reg; RET
      // Pattern 2: MOV32ri EAX, 0; SETCCr AL; RET
      MachineInstr &ZeroInst = *I;

      X86::CondCode CC = X86::COND_INVALID;
      MachineBasicBlock::iterator SetI, RetI;
      SmallVector<MachineInstr *, 4> ToRemove;

      // Try pattern 1: XOR32rr + SETCCr + MOV32rr EAX + RET
      if (ZeroInst.getOpcode() == X86::XOR32rr ||
          ZeroInst.getOpcode() == X86::XOR32rr_REV) {
        if (ZeroInst.getOperand(1).getReg() != ZeroInst.getOperand(2).getReg()) {
          ++I; continue;
        }
        Register ZeroReg = ZeroInst.getOperand(0).getReg();
        MCPhysReg ZeroReg8 = get8bitSubReg(ZeroReg);
        if (!ZeroReg8) { ++I; continue; }

        SetI = std::next(I);
        if (SetI == E || SetI->getOpcode() != X86::SETCCr ||
            SetI->getOperand(0).getReg() != ZeroReg8) {
          ++I; continue;
        }
        CC = X86::getCondFromSETCC(*SetI);

        auto MovI = std::next(SetI);
        if (MovI == E) { ++I; continue; }
        if ((MovI->getOpcode() != X86::MOV32rr &&
             MovI->getOpcode() != X86::MOV32rr_REV) ||
            MovI->getOperand(0).getReg() != X86::EAX ||
            MovI->getOperand(1).getReg() != ZeroReg) {
          ++I; continue;
        }

        RetI = std::next(MovI);
        ToRemove = {&ZeroInst, &*SetI, &*MovI};
      }
      // Try pattern 2: MOV32ri EAX, 0; SETCCr AL; RET
      else if (ZeroInst.getOpcode() == X86::MOV32ri &&
               ZeroInst.getOperand(0).getReg() == X86::EAX &&
               ZeroInst.getOperand(1).isImm() &&
               ZeroInst.getOperand(1).getImm() == 0) {
        SetI = std::next(I);
        if (SetI == E || SetI->getOpcode() != X86::SETCCr ||
            SetI->getOperand(0).getReg() != X86::AL) {
          ++I; continue;
        }
        CC = X86::getCondFromSETCC(*SetI);

        RetI = std::next(SetI);
        ToRemove = {&ZeroInst, &*SetI};
      }
      else {
        ++I; continue;
      }

      if (CC == X86::COND_INVALID) { ++I; continue; }

      // Find the RET: must be next in same block
      bool IsRet = false;
      if (RetI != E) {
        IsRet = (RetI->getOpcode() == X86::RET ||
                 RetI->getOpcode() == X86::RET32 ||
                 RetI->getOpcode() == X86::RETI32 ||
                 RetI->isReturn());
      }
      // Also check: setcc is last instruction and block falls through to ret
      if (!IsRet && RetI == E) {
        MachineBasicBlock *NextMBB = MBB.getNextNode();
        if (NextMBB && !NextMBB->empty() && NextMBB->front().isReturn() &&
            NextMBB->pred_size() > 0) {
          // The ret is in the successor. Clone it into our new false block.
          RetI = NextMBB->front().getIterator();
          IsRet = true;
        }
      }
      if (!IsRet) { ++I; continue; }

      // We have the full pattern. Rewrite to branch form.
      // Create a new MBB for the "false" (xor eax,eax) case.
      X86::CondCode InvCC = invertCC(CC);
      if (InvCC == X86::COND_INVALID) { ++I; continue; }

      DebugLoc DL = ZeroInst.getDebugLoc();

      // Create false block after current block
      MachineBasicBlock *FalseMBB = MF.CreateMachineBasicBlock();
      FalseMBB->setLabelMustBeEmitted();
      MF.insert(std::next(MachineFunction::iterator(&MBB)), FalseMBB);

      // Move the RET to the false block (it will follow xor eax,eax)
      // But first, clone the ret for the true path
      unsigned RetOpc = RetI->getOpcode();

      // Insert point is before the first instruction to remove
      MachineInstr *InsertBefore = ToRemove[0];

      // Build in current block: JCC FalseMBB, inverted_CC; MOV32ri EAX, 1; RET
      BuildMI(MBB, *InsertBefore, DL, TII->get(X86::JCC_1))
          .addMBB(FalseMBB)
          .addImm(InvCC);
      BuildMI(MBB, *InsertBefore, DL, TII->get(X86::MOV32ri), X86::EAX)
          .addImm(1);
      // Clone the RET for the true path
      {
        auto TrueRet = BuildMI(MBB, *InsertBefore, DL, TII->get(RetI->getOpcode()));
        for (const auto &MO : RetI->operands())
          TrueRet.add(MO);
      }

      // Build in false block: XOR32rr EAX, EAX
      BuildMI(*FalseMBB, FalseMBB->end(), DL,
              TII->get(X86::XOR32rr_REV), X86::EAX)
          .addReg(X86::EAX, RegState::Undef)
          .addReg(X86::EAX, RegState::Undef);

      // Transfer successors and add false block
      FalseMBB->transferSuccessorsAndUpdatePHIs(&MBB);
      MBB.addSuccessor(FalseMBB);

      // Move or clone the RET into the false block
      if (RetI->getParent() == &MBB) {
        // RET is in our block - move it and everything after
        FalseMBB->splice(FalseMBB->end(), &MBB, RetI, MBB.end());
      } else {
        // RET is in successor block - clone it
        auto ClonedRet = BuildMI(*FalseMBB, FalseMBB->end(), DL,
                                  TII->get(RetI->getOpcode()));
        for (const auto &MO : RetI->operands())
          ClonedRet.add(MO);
      }

      // Remove old instructions
      for (MachineInstr *MI : ToRemove)
        MI->eraseFromParent();
      I = MBB.begin(); // restart scan

      Changed = true;
      break; // restart the block scan since we modified the block
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferBranchBoolPass() {
  return new X86PreferBranchBoolPass();
}

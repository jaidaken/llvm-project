//===--- X86PreventSetccMerge.cpp - Split xor+setcc into branch form ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: MSVC 6.0 generates separate true/false return paths:
//
//   test eax, eax
//   jne .true
//   xor eax, eax       ; return 0
//   ret
//   .true:
//   mov eax, 1          ; return 1
//   ret
//
// Clang merges these into an xor+setcc pattern (pre-zeroes EAX with XOR):
//
//   test eax, eax
//   xor eax, eax
//   setne al            ; return 0 or 1
//   ret
//
// This pass detects XOR32rr/XOR32rr_REV EAX,EAX + SETCCr AL + RET sequences
// and replaces them with the branch-based form, gated on the
// prevent_setcc_merge function attribute.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prevent-setcc-merge"

namespace {
class X86PreventSetccMergePass : public MachineFunctionPass {
public:
  static char ID;
  X86PreventSetccMergePass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 prevent setcc+movzx merge";
  }
};
} // end anonymous namespace

char X86PreventSetccMergePass::ID = 0;

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
  case X86::COND_S:  return X86::COND_NS;
  case X86::COND_NS: return X86::COND_S;
  case X86::COND_P:  return X86::COND_NP;
  case X86::COND_NP: return X86::COND_P;
  case X86::COND_O:  return X86::COND_NO;
  case X86::COND_NO: return X86::COND_O;
  default: return X86::COND_INVALID;
  }
}

/// Return true if MI is a RET or RETI32 instruction.
static bool isRetInstruction(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  return Opc == X86::RET || Opc == X86::RET32 || Opc == X86::RETI32 ||
         MI.isReturn();
}

/// Advance iterator past any debug/pseudo instructions. Returns E if none
/// found.
static MachineBasicBlock::iterator
skipNonReal(MachineBasicBlock::iterator It, MachineBasicBlock::iterator E) {
  while (It != E && (It->isDebugInstr() || It->isPseudo()))
    ++It;
  return It;
}

bool X86PreventSetccMergePass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("prevent_setcc_merge"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E; ) {
      MachineInstr &XorMI = *I;

      // Step 1: Look for XOR32rr/XOR32rr_REV EAX, EAX
      if (XorMI.getOpcode() != X86::XOR32rr &&
          XorMI.getOpcode() != X86::XOR32rr_REV) {
        ++I;
        continue;
      }

      // Both source operands must be the same register (zeroing idiom).
      if (XorMI.getOperand(1).getReg() != XorMI.getOperand(2).getReg()) {
        ++I;
        continue;
      }

      // Destination must be EAX.
      if (XorMI.getOperand(0).getReg() != X86::EAX) {
        ++I;
        continue;
      }

      unsigned XorOpcode = XorMI.getOpcode();

      // Step 1b: Check if there is a MOV32rr/MOV32rr_REV before the XOR that
      // saves EAX to a scratch register. This happens when the compiler saves
      // the call result to another register before zeroing EAX for setcc.
      // Pattern: MOV32rr TmpReg, EAX; XOR EAX,EAX; TEST TmpReg,TmpReg; SETCC AL
      MachineInstr *DeadMov = nullptr;
      Register TmpReg;
      if (I != MBB.begin()) {
        auto PrevI = std::prev(I);
        while (PrevI != MBB.begin() &&
               (PrevI->isDebugInstr() || PrevI->isPseudo()))
          --PrevI;
        if ((PrevI->getOpcode() == X86::MOV32rr ||
             PrevI->getOpcode() == X86::MOV32rr_REV) &&
            PrevI->getOperand(1).getReg() == X86::EAX) {
          DeadMov = &*PrevI;
          TmpReg = PrevI->getOperand(0).getReg();
        }
      }

      // Step 2: Next non-pseudo instruction must be SETCCr AL. The compiler
      // typically places a flag-setting instruction (TEST/CMP) between the
      // XOR and SETCCr, so skip past one optional compare instruction.
      auto NextI = skipNonReal(std::next(I), E);
      MachineBasicBlock::iterator FlagSetI;
      bool HasFlagSetter = false;

      if (NextI != E && NextI->isCompare()) {
        HasFlagSetter = true;
        FlagSetI = NextI;
        NextI = skipNonReal(std::next(NextI), E);
      }

      auto SetI = NextI;
      if (SetI == E || SetI->getOpcode() != X86::SETCCr) {
        ++I;
        continue;
      }

      if (SetI->getOperand(0).getReg() != X86::AL) {
        ++I;
        continue;
      }

      X86::CondCode CC = X86::getCondFromSETCC(*SetI);
      if (CC == X86::COND_INVALID) {
        ++I;
        continue;
      }

      // Step 3: After SETCCr, skip any POP32r instructions (callee-save
      // restores from forced_callee_saves), then expect RET/RETI32.
      auto AfterSet = skipNonReal(std::next(SetI), E);
      SmallVector<MachineInstr *, 4> Pops;
      while (AfterSet != E && AfterSet->getOpcode() == X86::POP32r) {
        Pops.push_back(&*AfterSet);
        AfterSet = skipNonReal(std::next(AfterSet), E);
      }

      MachineBasicBlock::iterator RetI;
      bool RetInSameBlock = false;

      if (AfterSet != E && isRetInstruction(*AfterSet)) {
        RetI = AfterSet;
        RetInSameBlock = true;
      } else if (AfterSet == E) {
        // Check the layout successor for a RET.
        MachineBasicBlock *NextMBB = MBB.getNextNode();
        if (NextMBB && !NextMBB->empty() &&
            isRetInstruction(NextMBB->front())) {
          RetI = NextMBB->front().getIterator();
        } else {
          ++I;
          continue;
        }
      } else {
        ++I;
        continue;
      }

      // Step 4: We have the pattern:
      //   XOR32rr EAX, EAX; [TEST/CMP]; SETCCr AL, CC; [POPs]; RET
      // Replace with:
      //   TEST/CMP                   -- flag-setter (if present) stays
      //   Jcc .false, inverted(CC)   -- jump to false path if CC does NOT hold
      //   MOV32ri EAX, 1             -- true path
      //   [POPs]                     -- callee-save restores (cloned)
      //   RET                        -- true path return
      //   .false:
      //   XOR32rr EAX, EAX           -- false path (preserving original encoding)
      //   [POPs]                     -- callee-save restores (cloned)
      //   RET                        -- false path return

      X86::CondCode InvCC = invertCC(CC);
      if (InvCC == X86::COND_INVALID) {
        ++I;
        continue;
      }

      DebugLoc DL = XorMI.getDebugLoc();

      // Create the "false" block after the current block.
      MachineBasicBlock *FalseMBB = MF.CreateMachineBasicBlock();
      FalseMBB->setLabelMustBeEmitted();
      MF.insert(std::next(MachineFunction::iterator(&MBB)), FalseMBB);

      // Insert before the SETCCr. The flag-setting instruction (TEST/CMP),
      // if present, remains in place before the Jcc so that EFLAGS are set
      // for the branch. The XOR is removed from the main block and placed
      // only in the false path.
      //   [TEST/CMP]                 -- stays (already here)
      //   JCC_1 FalseMBB, inverted(CC)
      //   MOV32ri EAX, 1
      //   RET (cloned)
      MachineInstr *InsertBefore = &*SetI;

      BuildMI(MBB, *InsertBefore, DL, TII->get(X86::JCC_1))
          .addMBB(FalseMBB)
          .addImm(InvCC);
      BuildMI(MBB, *InsertBefore, DL, TII->get(X86::MOV32ri), X86::EAX)
          .addImm(1);

      // Clone the POPs for the true path (callee-save restores).
      for (MachineInstr *PopMI : Pops) {
        auto Clone = BuildMI(MBB, *InsertBefore, DL,
                              TII->get(PopMI->getOpcode()));
        for (const auto &MO : PopMI->operands())
          Clone.add(MO);
      }

      // Clone the RET for the true path.
      {
        auto TrueRet = BuildMI(MBB, *InsertBefore, DL,
                                TII->get(RetI->getOpcode()));
        for (const auto &MO : RetI->operands())
          TrueRet.add(MO);
      }

      // Build in false block: XOR EAX, EAX using the same encoding (REV or
      // not) as the original.
      BuildMI(*FalseMBB, FalseMBB->end(), DL,
              TII->get(XorOpcode), X86::EAX)
          .addReg(X86::EAX, RegState::Undef)
          .addReg(X86::EAX, RegState::Undef);

      // Clone the POPs for the false path (callee-save restores).
      for (MachineInstr *PopMI : Pops) {
        auto Clone = BuildMI(*FalseMBB, FalseMBB->end(), DL,
                              TII->get(PopMI->getOpcode()));
        for (const auto &MO : PopMI->operands())
          Clone.add(MO);
      }

      // Transfer successors from MBB to FalseMBB and wire up.
      FalseMBB->transferSuccessorsAndUpdatePHIs(&MBB);
      MBB.addSuccessor(FalseMBB);

      // Move or clone the RET into the false block.
      if (RetInSameBlock) {
        FalseMBB->splice(FalseMBB->end(), &MBB, RetI, MBB.end());
      } else {
        auto ClonedRet = BuildMI(*FalseMBB, FalseMBB->end(), DL,
                                  TII->get(RetI->getOpcode()));
        for (const auto &MO : RetI->operands())
          ClonedRet.add(MO);
      }

      // Remove the old XOR, SETCCr, and POP instructions.
      for (MachineInstr *PopMI : Pops)
        PopMI->eraseFromParent();
      SetI->eraseFromParent();
      XorMI.eraseFromParent();

      // Step 5: Clean up dead MOV if present. If the flag setter tests
      // TmpReg (e.g. TEST ECX,ECX) and we have MOV TmpReg, EAX before,
      // rewrite the flag setter to use EAX directly and remove the MOV.
      if (DeadMov && HasFlagSetter && TmpReg != X86::NoRegister) {
        bool UsesTmpReg = false;
        for (unsigned i = 0, e = FlagSetI->getNumOperands(); i < e; ++i) {
          if (FlagSetI->getOperand(i).isReg() &&
              FlagSetI->getOperand(i).getReg() == TmpReg) {
            UsesTmpReg = true;
            break;
          }
        }
        if (UsesTmpReg) {
          for (unsigned i = 0, e = FlagSetI->getNumOperands(); i < e; ++i) {
            MachineOperand &MO = FlagSetI->getOperand(i);
            if (MO.isReg() && MO.getReg() == TmpReg)
              MO.setReg(X86::EAX);
          }
          DeadMov->eraseFromParent();
        }
      }

      Changed = true;
      I = MBB.begin(); // restart scan on this block
      break;           // break inner loop to restart
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreventSetccMergePass() {
  return new X86PreventSetccMergePass();
}

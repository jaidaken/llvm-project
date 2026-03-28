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
// Pattern 2: Multi-comparison AND pattern.
//
// Clang generates:
//   [CMP/TEST]    ; flag setter 1
//   setcc1 al     ; result of cmp 1
//   [CMP/TEST]    ; flag setter 2
//   setcc2 cl     ; result of cmp 2
//   and al, cl    ; combine
//   movzx eax, al ; extend
//   ret
//
// MSVC 6.0 generates:
//   [CMP/TEST]          ; flag setter 1
//   jcc1 .return_1      ; if true, return 1
//   [CMP/TEST]          ; flag setter 2
//   jcc2 .return_1      ; if true, return 1
//   xor eax, eax        ; both false: return 0
//   ret
//   .return_1:
//   mov eax, 1
//   ret
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

/// Try to match and transform the multi-comparison AND pattern in MBB
/// starting at iterator I. Returns true if the pattern was matched and
/// transformed.
///
/// Pattern:
///   [CMP/TEST 1]       ; optional flag setter for setcc 1
///   SETCCr AL, CC1
///   [CMP/TEST 2]       ; optional flag setter for setcc 2
///   SETCCr CL, CC2
///   AND8rr AL, CL
///   MOVZX32rr8 EAX, AL
///   [POP32r ...]       ; optional callee-save restores
///   RET
///
/// Replaced with:
///   [CMP/TEST 1]       ; flag setter 1 stays
///   JCC_1 TrueMBB, CC1 ; jump to true if condition 1 holds
///   [CMP/TEST 2]       ; flag setter 2 stays
///   JCC_1 TrueMBB, CC2 ; jump to true if condition 2 holds
///   XOR32rr EAX, EAX   ; false path: return 0
///   [POPs]
///   RET
///   TrueMBB:
///   MOV32ri EAX, 1     ; true path: return 1
///   [POPs]
///   RET
static bool tryTransformMultiSetccAnd(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator I,
                                      const X86InstrInfo *TII,
                                      MachineFunction &MF) {
  auto E = MBB.end();

  // Step 1: Find first SETCCr. There may be a flag-setting instruction
  // (CMP/TEST) immediately before it.
  auto It = I;

  // Locate first flag-setter (optional) + SETCCr pair.
  MachineBasicBlock::iterator FlagSet1;
  bool HasFlagSet1 = false;
  if (It != E && It->isCompare()) {
    HasFlagSet1 = true;
    FlagSet1 = It;
    It = skipNonReal(std::next(It), E);
  }

  if (It == E || It->getOpcode() != X86::SETCCr)
    return false;
  auto Set1I = It;
  Register Set1Reg = Set1I->getOperand(0).getReg();
  X86::CondCode CC1 = X86::getCondFromSETCC(*Set1I);
  if (CC1 == X86::COND_INVALID)
    return false;

  // Step 2: Find second flag-setter (optional) + SETCCr pair.
  It = skipNonReal(std::next(Set1I), E);

  MachineBasicBlock::iterator FlagSet2;
  bool HasFlagSet2 = false;
  if (It != E && It->isCompare()) {
    HasFlagSet2 = true;
    FlagSet2 = It;
    It = skipNonReal(std::next(It), E);
  }

  if (It == E || It->getOpcode() != X86::SETCCr)
    return false;
  auto Set2I = It;
  Register Set2Reg = Set2I->getOperand(0).getReg();
  X86::CondCode CC2 = X86::getCondFromSETCC(*Set2I);
  if (CC2 == X86::COND_INVALID)
    return false;

  // The two SETCCr results must go to different 8-bit registers.
  if (Set1Reg == Set2Reg)
    return false;

  // Step 3: Expect AND8rr combining the two setcc results.
  It = skipNonReal(std::next(Set2I), E);
  if (It == E || It->getOpcode() != X86::AND8rr)
    return false;
  auto AndI = It;

  // Verify the AND uses the two setcc destination registers.
  Register AndDst = AndI->getOperand(0).getReg();
  Register AndSrc1 = AndI->getOperand(1).getReg();
  Register AndSrc2 = AndI->getOperand(2).getReg();
  if (!((AndSrc1 == Set1Reg && AndSrc2 == Set2Reg) ||
        (AndSrc1 == Set2Reg && AndSrc2 == Set1Reg)))
    return false;

  // Step 4: Expect MOVZX32rr8 extending the AND result to EAX.
  It = skipNonReal(std::next(AndI), E);
  if (It == E || It->getOpcode() != X86::MOVZX32rr8)
    return false;
  auto MovzxI = It;
  if (MovzxI->getOperand(0).getReg() != X86::EAX)
    return false;
  if (MovzxI->getOperand(1).getReg() != AndDst)
    return false;

  // Step 5: Skip optional POP32r instructions, then expect RET.
  It = skipNonReal(std::next(MovzxI), E);
  SmallVector<MachineInstr *, 4> Pops;
  while (It != E && It->getOpcode() == X86::POP32r) {
    Pops.push_back(&*It);
    It = skipNonReal(std::next(It), E);
  }

  MachineBasicBlock::iterator RetI;
  bool RetInSameBlock = false;

  if (It != E && isRetInstruction(*It)) {
    RetI = It;
    RetInSameBlock = true;
  } else if (It == E) {
    MachineBasicBlock *NextMBB = MBB.getNextNode();
    if (NextMBB && !NextMBB->empty() && isRetInstruction(NextMBB->front())) {
      RetI = NextMBB->front().getIterator();
    } else {
      return false;
    }
  } else {
    return false;
  }

  // Step 6: Build the replacement.
  //   [CMP/TEST 1]         ; stays in place
  //   JCC_1 TrueMBB, CC1
  //   [CMP/TEST 2]         ; stays in place
  //   JCC_1 TrueMBB, CC2
  //   XOR32rr EAX, EAX     ; false path
  //   [POPs]
  //   RET
  //   TrueMBB:
  //   MOV32ri EAX, 1       ; true path
  //   [POPs]
  //   RET

  DebugLoc DL = Set1I->getDebugLoc();

  // Create the "true" block (return 1) after the current block.
  MachineBasicBlock *TrueMBB = MF.CreateMachineBasicBlock();
  TrueMBB->setLabelMustBeEmitted();
  MF.insert(std::next(MachineFunction::iterator(&MBB)), TrueMBB);

  // Replace first SETCCr with JCC to TrueMBB using its condition.
  BuildMI(MBB, *Set1I, DL, TII->get(X86::JCC_1))
      .addMBB(TrueMBB)
      .addImm(CC1);

  // Replace second SETCCr with JCC to TrueMBB using its condition.
  // The flag-setter for the second SETCCr (if present) stays in place
  // between the first JCC and the second JCC.
  BuildMI(MBB, *Set2I, DL, TII->get(X86::JCC_1))
      .addMBB(TrueMBB)
      .addImm(CC2);

  // False path: XOR EAX, EAX (return 0) in the current block.
  // Insert before the AND instruction position (which we are about to erase).
  // Use the insertion point just after the second JCC we added.
  MachineInstr *InsertBefore = &*AndI;
  BuildMI(MBB, *InsertBefore, DL, TII->get(X86::XOR32rr), X86::EAX)
      .addReg(X86::EAX, RegState::Undef)
      .addReg(X86::EAX, RegState::Undef);

  // Clone the POPs for the false path.
  for (MachineInstr *PopMI : Pops) {
    MachineInstr *Clone = MF.CloneMachineInstr(PopMI);
    MBB.insert(InsertBefore, Clone);
  }

  // Clone the RET for the false path.
  {
    MachineInstr *FalseRet = MF.CloneMachineInstr(&*RetI);
    MBB.insert(InsertBefore, FalseRet);
  }

  // Build the true block: MOV EAX, 1; [POPs]; RET.
  BuildMI(*TrueMBB, TrueMBB->end(), DL, TII->get(X86::MOV32ri), X86::EAX)
      .addImm(1);

  for (MachineInstr *PopMI : Pops) {
    MachineInstr *Clone = MF.CloneMachineInstr(PopMI);
    TrueMBB->insert(TrueMBB->end(), Clone);
  }

  // Transfer successors from MBB to TrueMBB and wire up.
  TrueMBB->transferSuccessorsAndUpdatePHIs(&MBB);
  MBB.addSuccessor(TrueMBB);

  // Move or clone the RET into the true block.
  if (RetInSameBlock) {
    TrueMBB->splice(TrueMBB->end(), &MBB, RetI, MBB.end());
  } else {
    MachineInstr *ClonedRet = MF.CloneMachineInstr(&*RetI);
    TrueMBB->insert(TrueMBB->end(), ClonedRet);
  }

  // Erase the old instructions: SETCCr x2, AND8rr, MOVZX, POPs.
  for (MachineInstr *PopMI : Pops)
    PopMI->eraseFromParent();
  MovzxI->eraseFromParent();
  AndI->eraseFromParent();
  Set2I->eraseFromParent();
  Set1I->eraseFromParent();

  return true;
}

bool X86PreventSetccMergePass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("prevent_setcc_merge"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  // Pattern 2: Multi-comparison AND pattern.
  // Scan each block for SETCCr + SETCCr + AND8rr + MOVZX + RET sequences.
  // This runs first because it is more specific; Pattern 1 (single XOR+SETCCr)
  // could otherwise consume one of the SETCCr instructions prematurely.
  for (MachineBasicBlock &MBB : MF) {
    bool BlockChanged = true;
    while (BlockChanged) {
      BlockChanged = false;
      for (auto I = MBB.begin(), E = MBB.end(); I != E; ++I) {
        // Look for a SETCCr or a compare that precedes one. We try to match
        // starting at each instruction that could be the first flag-setter
        // or the first SETCCr.
        if (I->isCompare() || I->getOpcode() == X86::SETCCr) {
          if (tryTransformMultiSetccAnd(MBB, I, TII, MF)) {
            Changed = true;
            BlockChanged = true;
            break; // restart scan on this block
          }
        }
      }
    }
  }

  // Pattern 1: Single XOR32rr + SETCCr + RET.
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
      // Use CloneMachineInstr to avoid duplicating implicit operands
      // (BuildMI adds implicits from MCInstrDesc, then copying all
      // operands from the original would double them).
      for (MachineInstr *PopMI : Pops) {
        MachineInstr *Clone = MF.CloneMachineInstr(PopMI);
        MBB.insert(InsertBefore, Clone);
      }

      // Clone the RET for the true path.
      {
        MachineInstr *TrueRet = MF.CloneMachineInstr(&*RetI);
        MBB.insert(InsertBefore, TrueRet);
      }

      // Build in false block: XOR EAX, EAX using the same encoding (REV or
      // not) as the original.
      BuildMI(*FalseMBB, FalseMBB->end(), DL,
              TII->get(XorOpcode), X86::EAX)
          .addReg(X86::EAX, RegState::Undef)
          .addReg(X86::EAX, RegState::Undef);

      // Clone the POPs for the false path (callee-save restores).
      for (MachineInstr *PopMI : Pops) {
        MachineInstr *Clone = MF.CloneMachineInstr(PopMI);
        FalseMBB->insert(FalseMBB->end(), Clone);
      }

      // Transfer successors from MBB to FalseMBB and wire up.
      FalseMBB->transferSuccessorsAndUpdatePHIs(&MBB);
      MBB.addSuccessor(FalseMBB);

      // Move or clone the RET into the false block.
      if (RetInSameBlock) {
        FalseMBB->splice(FalseMBB->end(), &MBB, RetI, MBB.end());
      } else {
        MachineInstr *ClonedRet = MF.CloneMachineInstr(&*RetI);
        FalseMBB->insert(FalseMBB->end(), ClonedRet);
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

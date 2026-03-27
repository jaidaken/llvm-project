//===--- X86InterleaveAddPush.cpp - Interleave ADD/LEA between PUSHes -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: MSVC 6.0 loads two fields from the same sub-object pointer,
// where the second load overwrites the base register, then interleaves an
// ADD/LEA between pushes of the loaded values:
//
//   mov eax, [edx+0x58]       ; load field A from info pointer
//   mov edx, [edx+0x5c]       ; load field B (overwrites base reg edx!)
//   add eax, 0x10             ; compute on field A
//   push edx                  ; push field B
//   push eax                  ; push computed A
//
// Clang schedules differently - it completes the arithmetic before the pushes
// or places it after both pushes:
//
//   mov eax, [edx+0x58]       ; load field A
//   mov edx, [edx+0x5c]       ; load field B
//   push edx                  ; push field B first
//   add eax, 0x10             ; then compute
//   push eax                  ; push computed A
//
// This pass finds ADD/LEA instructions that sit between two consecutive PUSH
// instructions and moves the ADD/LEA to before both PUSHes, matching MSVC 6.0's
// interleaved scheduling pattern.
//
// Gate: function attribute "interleave_add_push".
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "MCTargetDesc/X86MCTargetDesc.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "x86-interleave-add-push"

namespace {
class X86InterleaveAddPushPass : public MachineFunctionPass {
public:
  static char ID;
  X86InterleaveAddPushPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 interleave ADD/LEA between PUSHes";
  }

private:
  const X86InstrInfo *TII = nullptr;

  bool processBlock(MachineBasicBlock &MBB);
};
} // end anonymous namespace

char X86InterleaveAddPushPass::ID = 0;

// ---------------------------------------------------------------------------
// Helper predicates
// ---------------------------------------------------------------------------

/// Check if an instruction is any PUSH (immediate, register, or memory).
static bool isAnyPush(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case X86::PUSH32i8:
  case X86::PUSH32i:
  case X86::PUSH32r:
  case X86::PUSH32rmm:
  case X86::PUSH32rmr:
    return true;
  default:
    return false;
  }
}

/// Check if an instruction is a PUSH of a register.
static bool isPushReg(const MachineInstr &MI) {
  return MI.getOpcode() == X86::PUSH32r;
}

/// Check if an instruction is an ADD or LEA with an immediate operand.
/// For ADD: ADD32ri, ADD32ri8 (reg = reg + imm)
/// For LEA: LEA32r (reg = [base + disp])
static bool isAddOrLeaImm(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case X86::ADD32ri:
  case X86::ADD32ri8:
    return true;
  case X86::LEA32r:
    // LEA32r operands: dest(0), base(1), scale(2), index(3), disp(4)
    // Accept LEA with no index register (simple base + disp).
    return MI.getOperand(3).getReg() == X86::NoRegister;
  default:
    return false;
  }
}

/// Get the destination register of an ADD or LEA instruction.
static Register getAddLeaDest(const MachineInstr &MI) {
  return MI.getOperand(0).getReg();
}

/// Get the source register of an ADD instruction (operand 1) or
/// the base register of a LEA instruction (operand 1).
static Register getAddLeaSrc(const MachineInstr &MI) {
  return MI.getOperand(1).getReg();
}

/// Check if the ADD/LEA modifies a register in-place (dest == src).
static bool isInPlaceAddLea(const MachineInstr &MI) {
  if (MI.getOpcode() == X86::ADD32ri || MI.getOpcode() == X86::ADD32ri8) {
    // ADD32ri: operand 0 is def, operand 1 is tied use (same reg).
    return true; // ADD always modifies in place for X86
  }
  if (MI.getOpcode() == X86::LEA32r) {
    return MI.getOperand(0).getReg() == MI.getOperand(1).getReg();
  }
  return false;
}

// ---------------------------------------------------------------------------
// Main transformation
// ---------------------------------------------------------------------------
//
// We scan for the pattern (walking backward from the second PUSH):
//
//   [some instructions...]
//   PUSH32r RegB               ; first push (of second field)
//   ADD/LEA RegA, RegA, imm    ; arithmetic on first field
//   PUSH32r RegA               ; second push (of computed value)
//
// And reorder to:
//
//   [some instructions...]
//   ADD/LEA RegA, RegA, imm    ; arithmetic moved before both pushes
//   PUSH32r RegB               ; first push
//   PUSH32r RegA               ; second push
//
// The ADD/LEA must not use RegB (otherwise moving it before the PUSH of RegB
// would change semantics).

bool X86InterleaveAddPushPass::processBlock(MachineBasicBlock &MBB) {
  bool Changed = false;

  for (auto I = MBB.begin(), E = MBB.end(); I != E; ) {
    MachineInstr &MI = *I;
    ++I; // advance early since we may splice

    // Look for: PUSH32r (the last push in a sequence)
    if (!isPushReg(MI))
      continue;

    Register PushAReg = MI.getOperand(0).getReg(); // reg being pushed last

    // Walk backward to find what's before this PUSH.
    auto PrevIt = MachineBasicBlock::iterator(MI);
    if (PrevIt == MBB.begin())
      continue;
    --PrevIt;
    while (PrevIt != MBB.begin() && (PrevIt->isPseudo() || PrevIt->isDebugInstr()))
      --PrevIt;

    // --- Pattern 1: PUSH_reg B, ADD/LEA A, PUSH_reg A ---
    // Clang: PUSH B, ADD A, PUSH A -> MSVC: ADD A, PUSH B, PUSH A
    if (isAddOrLeaImm(*PrevIt)) {
      MachineInstr &AddMI = *PrevIt;
      Register AddDest = getAddLeaDest(AddMI);
      if (AddDest == PushAReg) {
        auto PushBIt = PrevIt;
        if (PushBIt != MBB.begin()) {
          --PushBIt;
          while (PushBIt != MBB.begin() &&
                 (PushBIt->isPseudo() || PushBIt->isDebugInstr()))
            --PushBIt;

          if (isPushReg(*PushBIt)) {
            Register PushBReg = PushBIt->getOperand(0).getReg();
            Register AddSrc = getAddLeaSrc(AddMI);
            if (AddSrc != PushBReg && AddDest != PushBReg) {
              bool Conflict = false;
              for (const MachineOperand &MO : AddMI.operands()) {
                if (MO.isReg() && MO.getReg() != X86::NoRegister) {
                  const TargetRegisterInfo *TRI = &TII->getRegisterInfo();
                  if (TRI->regsOverlap(MO.getReg(), PushBReg) &&
                      MO.getReg() != AddDest && MO.getReg() != AddSrc) {
                    Conflict = true;
                    break;
                  }
                }
              }
              if (!Conflict) {
                LLVM_DEBUG(dbgs() << "InterleaveAddPush(P1): moving "
                    << TII->getName(AddMI.getOpcode())
                    << " before PUSH " << printReg(PushBReg, &TII->getRegisterInfo())
                    << " in " << MBB.getParent()->getName() << "\n");
                MBB.splice(MachineBasicBlock::iterator(*PushBIt), &MBB,
                           MachineBasicBlock::iterator(AddMI),
                           std::next(MachineBasicBlock::iterator(AddMI)));
                Changed = true;
                continue;
              }
            }
          }
        }
      }
    }

    // --- Pattern 2: ADD/LEA A, PUSH_reg B, PUSH_imm, PUSH_reg A ---
    // Clang: ADD A, PUSH B, PUSH imm, PUSH A
    //     -> MSVC: PUSH B, PUSH imm, ADD A, PUSH A
    // The ADD is before all three pushes; move it between push 2 and push 3.
    if (isAnyPush(*PrevIt) && !isPushReg(*PrevIt)) {
      // PrevIt is a PUSH_imm (push 2)

      auto PushBIt = PrevIt;
      if (PushBIt != MBB.begin()) {
        --PushBIt;
        while (PushBIt != MBB.begin() &&
               (PushBIt->isPseudo() || PushBIt->isDebugInstr()))
          --PushBIt;

        if (isPushReg(*PushBIt)) {
          // PushBIt is PUSH_reg B (push 1)
          Register PushBReg = PushBIt->getOperand(0).getReg();

          auto AddIt = PushBIt;
          if (AddIt != MBB.begin()) {
            --AddIt;
            while (AddIt != MBB.begin() &&
                   (AddIt->isPseudo() || AddIt->isDebugInstr()))
              --AddIt;

            if (isAddOrLeaImm(*AddIt)) {
              MachineInstr &AddMI = *AddIt;
              Register AddDest = getAddLeaDest(AddMI);
              Register AddSrc = getAddLeaSrc(AddMI);

              if (AddDest == PushAReg &&
                  AddSrc != PushBReg && AddDest != PushBReg) {
                LLVM_DEBUG(dbgs() << "InterleaveAddPush(P2): moving "
                    << TII->getName(AddMI.getOpcode())
                    << " after PUSH_imm, before PUSH "
                    << printReg(PushAReg, &TII->getRegisterInfo())
                    << " in " << MBB.getParent()->getName() << "\n");

                // Move ADD to just before the last PUSH (MI).
                MBB.splice(MachineBasicBlock::iterator(MI), &MBB,
                           MachineBasicBlock::iterator(AddMI),
                           std::next(MachineBasicBlock::iterator(AddMI)));
                Changed = true;
                continue;
              }
            }
          }
        }
      }
    }
  }

  return Changed;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

bool X86InterleaveAddPushPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("interleave_add_push"))
    return false;

  TII = MF.getSubtarget<X86Subtarget>().getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF)
    Changed |= processBlock(MBB);

  return Changed;
}

FunctionPass *llvm::createX86InterleaveAddPushPass() {
  return new X86InterleaveAddPushPass();
}

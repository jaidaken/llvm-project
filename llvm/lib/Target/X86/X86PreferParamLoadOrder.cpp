//===--- X86PreferParamLoadOrder.cpp - Force stack param load order --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: Some MSVC 6.0 functions load stack parameters in a specific
// order for non-CALL patterns (e.g., ADD-to-memory operations).  The existing
// prefer_sequential_param_load only works backward from CALL instructions.
//
// This pass is controlled by the "prefer_param_load_order" attribute with a
// format like "4,8" meaning: the stack param at ESP+4 must be loaded before
// the stack param at ESP+8.  It scans the function for ESP-relative loads
// at the specified offsets and reorders them so that the first offset in the
// attribute string is loaded first in program order.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-param-load-order"

namespace {
class X86PreferParamLoadOrderPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferParamLoadOrderPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 prefer param load order";
  }
};
} // end anonymous namespace

char X86PreferParamLoadOrderPass::ID = 0;

/// Check if MI is a 32-bit register load from [ESP + disp].
/// Returns the ESP displacement via OutDisp if true.
static bool isEspLoad(const MachineInstr &MI, int64_t &OutDisp) {
  if (MI.getOpcode() != X86::MOV32rm)
    return false;
  // Operand layout: dest, base, scale, index, disp, segment
  if (!MI.getOperand(1).isReg() ||
      MI.getOperand(1).getReg() != X86::ESP)
    return false;
  if (MI.getOperand(2).getImm() != 1) // scale = 1
    return false;
  if (MI.getOperand(3).getReg() != X86::NoRegister) // no index
    return false;
  OutDisp = MI.getOperand(4).getImm();
  return OutDisp > 0; // stack params have positive offsets from ESP
}

/// Check if an instruction is any PUSH.
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

/// Parse comma-separated integer offsets from the attribute string.
static void parseOffsets(StringRef AttrVal, SmallVectorImpl<int64_t> &Offsets) {
  SmallVector<StringRef, 4> Parts;
  AttrVal.split(Parts, ',');
  for (StringRef Part : Parts) {
    int64_t Val = 0;
    if (!Part.trim().getAsInteger(0, Val))
      Offsets.push_back(Val);
  }
}

bool X86PreferParamLoadOrderPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("prefer_param_load_order"))
    return false;

  StringRef AttrVal =
      MF.getFunction().getFnAttribute("prefer_param_load_order")
          .getValueAsString();

  SmallVector<int64_t, 4> DesiredOrder;
  parseOffsets(AttrVal, DesiredOrder);
  if (DesiredOrder.size() < 2)
    return false;

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Collect all ESP loads with their offsets and positions.
    struct LoadInfo {
      MachineInstr *MI;
      int64_t Disp;
    };
    SmallVector<LoadInfo, 8> EspLoads;

    for (MachineInstr &MI : MBB) {
      if (MI.isPseudo() || MI.isDebugInstr())
        continue;
      int64_t Disp = 0;
      if (isEspLoad(MI, Disp)) {
        EspLoads.push_back({&MI, Disp});
      }
    }

    if (EspLoads.size() < 2)
      continue;

    // Find loads matching the desired offsets.
    SmallVector<MachineInstr *, 4> MatchedLoads;
    for (int64_t DesiredDisp : DesiredOrder) {
      for (auto &LI : EspLoads) {
        if (LI.Disp == DesiredDisp) {
          MatchedLoads.push_back(LI.MI);
          break;
        }
      }
    }

    // All desired offsets must be present.
    if ((int64_t)MatchedLoads.size() != (int64_t)DesiredOrder.size())
      continue;

    // Check if they are already in the desired program order.
    bool AlreadyOrdered = true;
    for (size_t i = 1; i < MatchedLoads.size(); i++) {
      // Check if MatchedLoads[i-1] comes before MatchedLoads[i] in MBB.
      bool FoundFirst = false;
      for (MachineInstr &MI : MBB) {
        if (&MI == MatchedLoads[i - 1]) {
          FoundFirst = true;
          break;
        }
        if (&MI == MatchedLoads[i]) {
          // i came before i-1, wrong order.
          AlreadyOrdered = false;
          break;
        }
      }
      if (!FoundFirst && !AlreadyOrdered)
        break;
      if (!AlreadyOrdered)
        break;
    }
    if (AlreadyOrdered)
      continue;

    // Safety: verify no PUSHes between the matched loads (would change ESP
    // offsets if reordered).
    MachineInstr *FirstLoad = nullptr;
    MachineInstr *LastLoad = nullptr;
    for (MachineInstr &MI : MBB) {
      for (MachineInstr *ML : MatchedLoads) {
        if (&MI == ML) {
          if (!FirstLoad)
            FirstLoad = ML;
          LastLoad = ML;
        }
      }
    }

    bool HasPushBetween = false;
    if (FirstLoad && LastLoad && FirstLoad != LastLoad) {
      bool InRange = false;
      for (MachineInstr &MI : MBB) {
        if (&MI == FirstLoad)
          InRange = true;
        if (InRange && &MI != FirstLoad && &MI != LastLoad && isAnyPush(MI)) {
          HasPushBetween = true;
          break;
        }
        if (&MI == LastLoad)
          break;
      }
    }
    if (HasPushBetween)
      continue;

    // For the two-load case: find the consumer of each load (the instruction
    // immediately following that reads the loaded register) and move both
    // load+consumer pairs into the desired order.
    if (MatchedLoads.size() == 2) {
      MachineInstr *FirstDesired = MatchedLoads[0]; // should come first
      MachineInstr *SecondDesired = MatchedLoads[1]; // should come second

      // Find which is currently first in program order.
      MachineInstr *CurrentFirst = nullptr;
      MachineInstr *CurrentSecond = nullptr;
      for (MachineInstr &MI : MBB) {
        if (&MI == FirstDesired || &MI == SecondDesired) {
          if (!CurrentFirst)
            CurrentFirst = &MI;
          else
            CurrentSecond = &MI;
        }
      }

      if (!CurrentFirst || !CurrentSecond)
        continue;

      // If FirstDesired is already first in program order, nothing to do.
      if (CurrentFirst == FirstDesired)
        continue;

      // CurrentFirst is SecondDesired, CurrentSecond is FirstDesired.
      // We need to move FirstDesired (and its consumer) before SecondDesired.

      // Find consumer of FirstDesired: next non-pseudo instruction that reads
      // the register loaded by FirstDesired.
      Register FirstReg = FirstDesired->getOperand(0).getReg();
      MachineInstr *FirstConsumer = nullptr;
      {
        auto It = MachineBasicBlock::iterator(*FirstDesired);
        ++It;
        while (It != MBB.end()) {
          if (!It->isPseudo() && !It->isDebugInstr()) {
            for (const MachineOperand &MO : It->operands()) {
              if (MO.isReg() && MO.isUse() && MO.getReg() == FirstReg) {
                FirstConsumer = &*It;
                break;
              }
            }
            break; // Only check the immediately following non-pseudo.
          }
          ++It;
        }
      }

      // Move FirstDesired before SecondDesired (CurrentFirst).
      auto InsertPt = MachineBasicBlock::iterator(*CurrentFirst);
      MBB.splice(InsertPt, &MBB,
                 MachineBasicBlock::iterator(*FirstDesired),
                 std::next(MachineBasicBlock::iterator(*FirstDesired)));

      // Move FirstConsumer right after FirstDesired (before SecondDesired).
      if (FirstConsumer && FirstConsumer != SecondDesired) {
        InsertPt = MachineBasicBlock::iterator(*CurrentFirst);
        MBB.splice(InsertPt, &MBB,
                   MachineBasicBlock::iterator(*FirstConsumer),
                   std::next(MachineBasicBlock::iterator(*FirstConsumer)));
      }

      LLVM_DEBUG(dbgs() << "PreferParamLoadOrder: reordered in "
                        << MBB.getParent()->getName() << "\n");
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferParamLoadOrderPass() {
  return new X86PreferParamLoadOrderPass();
}

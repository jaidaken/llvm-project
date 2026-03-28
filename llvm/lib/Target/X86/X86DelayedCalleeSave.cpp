//===--- X86DelayedCalleeSave.cpp - Delay callee-save push ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: MSVC 6.0 sometimes delays a callee-save push until after the
// first TEST/CMP in the entry block, interleaving it between the flag-setting
// instruction and its conditional branch:
//
//   push esi              ; unconditional callee save (at prologue)
//   mov esi, ecx          ; save this
//   mov ecx, [esi+0x8c]   ; load field
//   test ecx, ecx         ; test field
//   push edi              ; DELAYED callee save (between test and branch)
//   mov [esi], 0x008a9a64 ; vtable write (also between test and branch)
//   je skip               ; branch
//
// This pass is gated on the "delayed_callee_save" function attribute.
// Format: delayed_callee_save("edi:after_first_test")
//
// The pass:
// 1. Finds the specified register's PUSH in the entry block prologue
//    (placed there by forced_callee_saves)
// 2. Removes it from the prologue
// 3. Re-inserts it between the first TEST/CMP and its following Jcc
// 4. Ensures the corresponding POP remains before each RET
//    (already handled by forced_callee_saves epilogue)
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "x86-delayed-callee-save"

namespace {
class X86DelayedCalleeSavePass : public MachineFunctionPass {
public:
  static char ID;
  X86DelayedCalleeSavePass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 delayed callee-save push (MSVC 6.0)";
  }
};
} // end anonymous namespace

char X86DelayedCalleeSavePass::ID = 0;

/// Parse a register name to an MCPhysReg.
static MCPhysReg parseRegName(StringRef Name) {
  return StringSwitch<MCPhysReg>(Name.trim().lower())
      .Case("eax", X86::EAX)
      .Case("ecx", X86::ECX)
      .Case("edx", X86::EDX)
      .Case("ebx", X86::EBX)
      .Case("esi", X86::ESI)
      .Case("edi", X86::EDI)
      .Case("ebp", X86::EBP)
      .Default(0);
}

/// Check if MI is a TEST or CMP instruction that sets EFLAGS.
static bool isTestOrCmp(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case X86::TEST8rr:  case X86::TEST16rr:  case X86::TEST32rr:
  case X86::TEST8ri:  case X86::TEST16ri:  case X86::TEST32ri:
  case X86::TEST8mi:  case X86::TEST16mi:  case X86::TEST32mi:
  case X86::CMP8rr:   case X86::CMP16rr:   case X86::CMP32rr:
  case X86::CMP8ri:   case X86::CMP16ri:   case X86::CMP32ri:
  case X86::CMP8ri8:  case X86::CMP16ri8:  case X86::CMP32ri8:
  case X86::CMP8mi:   case X86::CMP16mi:   case X86::CMP32mi:
  case X86::CMP8mi8:  case X86::CMP16mi8:  case X86::CMP32mi8:
  case X86::CMP8rm:   case X86::CMP16rm:   case X86::CMP32rm:
  case X86::CMP8mr:   case X86::CMP16mr:   case X86::CMP32mr:
    return true;
  default:
    return false;
  }
}

/// Parse the attribute value. Format: "edi:after_first_test"
/// Returns true on success, sets Reg and Placement.
static bool parseAttribute(StringRef AttrVal, MCPhysReg &Reg,
                           StringRef &Placement) {
  auto [RegStr, PlaceStr] = AttrVal.split(':');
  if (RegStr.empty() || PlaceStr.empty())
    return false;

  Reg = parseRegName(RegStr);
  if (Reg == 0)
    return false;

  Placement = PlaceStr;
  return true;
}

bool X86DelayedCalleeSavePass::runOnMachineFunction(MachineFunction &MF) {
  const Function &F = MF.getFunction();
  if (!F.hasFnAttribute("delayed_callee_save"))
    return false;

  StringRef AttrVal =
      F.getFnAttribute("delayed_callee_save").getValueAsString();

  MCPhysReg DelayReg;
  StringRef Placement;
  if (!parseAttribute(AttrVal, DelayReg, Placement))
    return false;

  // Supported placements: "after_first_test", "after_first_add".
  if (Placement != "after_first_test" && Placement != "after_first_add")
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  MachineBasicBlock *EntryBlock = &MF.front();
  DebugLoc DL;

  // Phase 1: Find and remove the PUSH of DelayReg from the entry block
  // prologue. The forced_callee_saves attribute placed it at the start.
  MachineInstr *PushToRemove = nullptr;
  for (MachineInstr &MI : *EntryBlock) {
    if (MI.getOpcode() == X86::PUSH32r) {
      Register PushReg = MI.getOperand(0).getReg();
      if (TRI->regsOverlap(PushReg, DelayReg)) {
        PushToRemove = &MI;
        break;
      }
    }
    // Stop searching after we pass the prologue region (first non-push,
    // non-debug instruction that isn't a frame setup).
    if (!MI.isPseudo() && !MI.isDebugInstr() &&
        MI.getOpcode() != X86::PUSH32r)
      break;
  }

  if (!PushToRemove) {
    LLVM_DEBUG(dbgs() << "DelayedCalleeSave: no PUSH for "
                      << printReg(DelayReg, TRI) << " found in prologue of "
                      << MF.getName() << "\n");
    return false;
  }

  // Phase 2: Find the insertion point based on placement mode.
  MachineBasicBlock::iterator InsertBefore = EntryBlock->end();

  if (Placement == "after_first_test") {
    // Find the first TEST/CMP and insert before the following JCC.
    for (auto I = EntryBlock->begin(), E = EntryBlock->end(); I != E; ++I) {
      if (isTestOrCmp(*I)) {
        // Find the JCC after the TEST/CMP (skipping debug instrs).
        auto JccIt = std::next(I);
        while (JccIt != E && (JccIt->isDebugInstr() || JccIt->isPseudo()))
          ++JccIt;
        if (JccIt != E &&
            (JccIt->getOpcode() == X86::JCC_1 ||
             JccIt->getOpcode() == X86::JCC_4)) {
          InsertBefore = JccIt;
        }
        break;
      }
    }
    if (InsertBefore == EntryBlock->end()) {
      LLVM_DEBUG(dbgs() << "DelayedCalleeSave: no TEST/CMP + JCC found in "
                        << MF.getName() << "\n");
      return false;
    }
  } else if (Placement == "after_first_add") {
    // Find the first ADD instruction and insert right after it.
    for (auto I = EntryBlock->begin(), E = EntryBlock->end(); I != E; ++I) {
      if (I->isDebugInstr() || I->isPseudo())
        continue;
      unsigned Opc = I->getOpcode();
      if (Opc == X86::ADD32ri || Opc == X86::ADD32ri8 ||
          Opc == X86::ADD32rr) {
        InsertBefore = std::next(I);
        break;
      }
    }
    if (InsertBefore == EntryBlock->end()) {
      LLVM_DEBUG(dbgs() << "DelayedCalleeSave: no ADD found in "
                        << MF.getName() << "\n");
      return false;
    }
  }

  LLVM_DEBUG(dbgs() << "DelayedCalleeSave: delaying PUSH "
                    << printReg(DelayReg, TRI) << " in "
                    << MF.getName() << " (placement: " << Placement << ")\n");

  // Phase 3: Remove the original PUSH from the prologue. Before removing,
  // adjust ESP-relative displacements between the push and the insertion
  // point, since removing the push changes the stack by -4 (ESP is 4 higher
  // without the push).
  //
  // Walk from right after the removed PUSH to the insertion point and adjust
  // ESP-based displacements by -4.
  {
    auto StartIt = std::next(MachineBasicBlock::iterator(PushToRemove));
    auto EndIt = InsertBefore;
    for (auto It = StartIt; It != EndIt; ++It) {
      if (It->isDebugInstr() || It->isPseudo())
        continue;
      // Adjust ESP-based memory operands: the PUSH was removed so ESP is
      // 4 bytes higher than expected. Subtract 4 from displacements.
      int MemOpIdx = X86II::getMemoryOperandNo(It->getDesc().TSFlags);
      if (MemOpIdx < 0)
        continue;
      MemOpIdx += X86II::getOperandBias(It->getDesc());
      unsigned BaseIdx = MemOpIdx + X86::AddrBaseReg;
      unsigned DispIdx = MemOpIdx + X86::AddrDisp;
      if (BaseIdx >= It->getNumOperands() || DispIdx >= It->getNumOperands())
        continue;
      MachineOperand &BaseMO = It->getOperand(BaseIdx);
      MachineOperand &DispMO = It->getOperand(DispIdx);
      if (BaseMO.isReg() && BaseMO.getReg() == X86::ESP && DispMO.isImm())
        DispMO.setImm(DispMO.getImm() - 4);
    }
  }

  // Remove the original PUSH.
  PushToRemove->eraseFromParent();

  // Phase 4: Insert the PUSH at the determined insertion point.
  BuildMI(*EntryBlock, InsertBefore, DL, TII->get(X86::PUSH32r))
      .addReg(DelayReg, RegState::Undef);

  // Phase 5: The corresponding POP is already in place from
  // forced_callee_saves. No additional POP insertion needed.

  return true;
}

FunctionPass *llvm::createX86DelayedCalleeSavePass() {
  return new X86DelayedCalleeSavePass();
}

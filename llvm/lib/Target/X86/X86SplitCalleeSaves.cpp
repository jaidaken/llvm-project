//===--- X86SplitCalleeSaves.cpp - Split callee-saves for MSVC 6.0 --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: Splits callee-save pushes into unconditional and conditional
// groups, matching MSVC 6.0's asymmetric prologue pattern.
//
// Gated behind the "split_callee_saves" function attribute.
// Format: split_callee_saves("ecx,esi|ebx,ebp,edi")
//   - Group 1 (before |): pushed unconditionally at entry block start
//   - Group 2 (after |): pushed at start of the taken-path successor of
//     the first conditional branch in the entry block
//   - Returns reachable from the taken path pop all (group 2 then group 1)
//   - Early-exit returns (not via taken path) pop only group 1
//
// Requires no_callee_saves to suppress the automatic prologue/epilogue.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "MCTargetDesc/X86BaseInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-split-callee-saves"
#define PASS_NAME "X86 split callee-saves pass"

/// Adjust the displacement of ALL ESP-based memory operands in MI by Delta.
static bool adjustEspDisp(MachineInstr &MI, int Delta) {
  int MemOpIdx = X86II::getMemoryOperandNo(MI.getDesc().TSFlags);
  if (MemOpIdx < 0)
    return false;
  MemOpIdx += X86II::getOperandBias(MI.getDesc());
  unsigned BaseIdx = MemOpIdx + X86::AddrBaseReg;
  unsigned DispIdx = MemOpIdx + X86::AddrDisp;
  if (BaseIdx >= MI.getNumOperands() || DispIdx >= MI.getNumOperands())
    return false;
  MachineOperand &BaseMO = MI.getOperand(BaseIdx);
  MachineOperand &DispMO = MI.getOperand(DispIdx);
  if (!BaseMO.isReg() || BaseMO.getReg() != X86::ESP || !DispMO.isImm())
    return false;
  DispMO.setImm(DispMO.getImm() + Delta);
  return true;
}

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

/// Parse "ecx,esi|ebx,ebp,edi" into two register groups.
static bool parseAttribute(StringRef Attr,
                           SmallVectorImpl<MCPhysReg> &Group1,
                           SmallVectorImpl<MCPhysReg> &Group2) {
  auto [Left, Right] = Attr.split('|');
  if (Right.empty())
    return false;

  SmallVector<StringRef, 4> Names1, Names2;
  Left.split(Names1, ',');
  Right.split(Names2, ',');

  for (StringRef N : Names1) {
    MCPhysReg R = parseRegName(N);
    if (R == 0)
      return false;
    Group1.push_back(R);
  }
  for (StringRef N : Names2) {
    MCPhysReg R = parseRegName(N);
    if (R == 0)
      return false;
    Group2.push_back(R);
  }
  return true;
}

/// Collect all blocks reachable from a given block via BFS/DFS,
/// without crossing through certain excluded blocks.
static void collectReachable(MachineBasicBlock *Start,
                             SmallPtrSetImpl<MachineBasicBlock *> &Visited) {
  SmallVector<MachineBasicBlock *, 16> Worklist;
  Worklist.push_back(Start);
  while (!Worklist.empty()) {
    MachineBasicBlock *MBB = Worklist.pop_back_val();
    if (!Visited.insert(MBB).second)
      continue;
    for (MachineBasicBlock *Succ : MBB->successors())
      Worklist.push_back(Succ);
  }
}

namespace {
class X86SplitCalleeSavesPass : public MachineFunctionPass {
public:
  static char ID;
  X86SplitCalleeSavesPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return PASS_NAME; }
};
} // end anonymous namespace

char X86SplitCalleeSavesPass::ID = 0;

bool X86SplitCalleeSavesPass::runOnMachineFunction(MachineFunction &MF) {
  const Function &F = MF.getFunction();
  if (!F.hasFnAttribute("split_callee_saves"))
    return false;

  StringRef AttrVal =
      F.getFnAttribute("split_callee_saves").getValueAsString();
  SmallVector<MCPhysReg, 4> Group1, Group2;
  if (!parseAttribute(AttrVal, Group1, Group2))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  MachineBasicBlock *EntryBlock = &MF.front();
  DebugLoc DL;

  // ===== Phase 1: Insert group 1 pushes at entry block start =====
  // Push in the order listed in the attribute (first listed = first pushed).
  int Group1StackDelta = Group1.size() * 4;
  for (unsigned i = 0; i < Group1.size(); ++i) {
    auto InsertPt = EntryBlock->begin();
    // Skip past any existing pushes we already inserted
    for (unsigned j = 0; j < i; ++j)
      ++InsertPt;
    BuildMI(*EntryBlock, InsertPt, DL, TII->get(X86::PUSH32r))
        .addReg(Group1[i], RegState::Undef);
  }

  // Adjust ESP-relative operands in entry block after the group 1 pushes.
  // Skip the push instructions we just inserted.
  {
    auto It = EntryBlock->begin();
    for (unsigned i = 0; i < Group1.size(); ++i)
      ++It;
    for (auto E = EntryBlock->end(); It != E; ++It)
      adjustEspDisp(*It, Group1StackDelta);
  }

  // ===== Phase 2: Find the first conditional branch in entry block =====
  MachineInstr *FirstCondBranch = nullptr;
  for (MachineInstr &MI : *EntryBlock) {
    if (MI.getOpcode() == X86::JCC_1 || MI.getOpcode() == X86::JCC_4) {
      FirstCondBranch = &MI;
      break;
    }
  }
  if (!FirstCondBranch)
    return true; // No conditional branch found; group 1 pushes are in place

  // The taken-path successor is the branch target.
  MachineBasicBlock *TakenBlock = FirstCondBranch->getOperand(0).getMBB();

  // The early-exit path is the other successor (fallthrough).
  MachineBasicBlock *EarlyExitBlock = nullptr;
  for (MachineBasicBlock *Succ : EntryBlock->successors()) {
    if (Succ != TakenBlock) {
      EarlyExitBlock = Succ;
      break;
    }
  }

  // ===== Phase 3: Insert group 2 pushes at start of taken-path block =====
  int Group2StackDelta = Group2.size() * 4;
  for (unsigned i = 0; i < Group2.size(); ++i) {
    auto InsertPt = TakenBlock->begin();
    for (unsigned j = 0; j < i; ++j)
      ++InsertPt;
    BuildMI(*TakenBlock, InsertPt, DL, TII->get(X86::PUSH32r))
        .addReg(Group2[i], RegState::Undef);
  }

  // Adjust ESP-relative operands in the taken block after the group 2 pushes.
  {
    auto It = TakenBlock->begin();
    for (unsigned i = 0; i < Group2.size(); ++i)
      ++It;
    for (auto E = TakenBlock->end(); It != E; ++It)
      adjustEspDisp(*It, Group2StackDelta);
  }

  // Adjust ESP-relative operands in all blocks reachable from TakenBlock
  // (except TakenBlock itself, already handled above).
  SmallPtrSet<MachineBasicBlock *, 32> TakenReachable;
  collectReachable(TakenBlock, TakenReachable);

  for (MachineBasicBlock *MBB : TakenReachable) {
    if (MBB == TakenBlock)
      continue;
    for (MachineInstr &MI : *MBB)
      adjustEspDisp(MI, Group2StackDelta);
  }

  // ===== Phase 4: Insert pops before returns =====
  // Blocks reachable from TakenBlock that contain RET get full pops.
  // Other blocks with RET (reachable only from early exit) get group 1 pops.
  for (MachineBasicBlock &MBB : MF) {
    MachineInstr *RetMI = nullptr;
    for (MachineInstr &MI : MBB) {
      if (MI.getOpcode() == X86::RET32 || MI.getOpcode() == X86::RET64) {
        RetMI = &MI;
        break;
      }
    }
    if (!RetMI)
      continue;

    bool OnTakenPath = TakenReachable.count(&MBB) != 0;

    if (OnTakenPath) {
      // Pop group 2 in reverse order, then group 1 in reverse order.
      for (int i = Group2.size() - 1; i >= 0; --i)
        BuildMI(MBB, RetMI, DL, TII->get(X86::POP32r), Group2[i]);
      for (int i = Group1.size() - 1; i >= 0; --i)
        BuildMI(MBB, RetMI, DL, TII->get(X86::POP32r), Group1[i]);
    } else {
      // Early exit: pop only group 1 in reverse order.
      for (int i = Group1.size() - 1; i >= 0; --i)
        BuildMI(MBB, RetMI, DL, TII->get(X86::POP32r), Group1[i]);
    }
  }

  return true;
}

FunctionPass *llvm::createX86SplitCalleeSavesPass() {
  return new X86SplitCalleeSavesPass();
}

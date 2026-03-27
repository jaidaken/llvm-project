//===--- X86ReorderSubEsp.cpp - Move SUB ESP before callee-save pushes ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: Post-regalloc pass that reorders SUB ESP relative to callee-save
// PUSHes in the prologue/epilogue.
//
// MSVC 6.0 sometimes allocates stack space BEFORE saving callee-saved regs:
//   sub esp, N       ; stack allocation
//   push esi         ; callee-save
//   push edi         ; callee-save
//
// Clang always does it the other way:
//   push esi         ; callee-save
//   push edi         ; callee-save
//   sub esp, N       ; stack allocation
//
// This pass:
//   1. Gates on the reorder_sub_esp attribute.
//   2. In the prologue: finds PUSH32r; ...; PUSH32r; SUB32ri/SUB32ri8 ESP, N
//      and moves the SUB before the PUSHes. Adjusts any ESP-relative operands
//      in the PUSHes (unlikely for callee-saves but checked for safety).
//   3. In the epilogue: finds ADD32ri/ADD32ri8 ESP, N; POP32r; ...; POP32r
//      and moves the ADD after the POPs.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-reorder-sub-esp"

namespace {
class X86ReorderSubEspPass : public MachineFunctionPass {
public:
  static char ID;
  X86ReorderSubEspPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 reorder SUB ESP before callee-save pushes";
  }

private:
  bool reorderPrologue(MachineBasicBlock &MBB);
  bool reorderEpilogue(MachineBasicBlock &MBB);
};
} // end anonymous namespace

char X86ReorderSubEspPass::ID = 0;

/// Check if the instruction is a callee-save PUSH32r.
static bool isCalleeSavePush(const MachineInstr &MI) {
  if (MI.getOpcode() != X86::PUSH32r)
    return false;
  Register Reg = MI.getOperand(0).getReg();
  // Typical callee-saved regs on x86-32: ESI, EDI, EBX, EBP.
  return Reg == X86::ESI || Reg == X86::EDI ||
         Reg == X86::EBX || Reg == X86::EBP;
}

/// Check if the instruction is a callee-save POP32r.
static bool isCalleeSavePop(const MachineInstr &MI) {
  if (MI.getOpcode() != X86::POP32r)
    return false;
  Register Reg = MI.getOperand(0).getReg();
  return Reg == X86::ESI || Reg == X86::EDI ||
         Reg == X86::EBX || Reg == X86::EBP;
}

/// Check if the instruction is SUB32ri or SUB32ri8 with ESP as dest/src.
static bool isSubEsp(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  if (Opc != X86::SUB32ri && Opc != X86::SUB32ri8)
    return false;
  return MI.getOperand(0).getReg() == X86::ESP &&
         MI.getOperand(1).getReg() == X86::ESP;
}

/// Check if the instruction is ADD32ri or ADD32ri8 with ESP as dest/src.
static bool isAddEsp(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  if (Opc != X86::ADD32ri && Opc != X86::ADD32ri8)
    return false;
  return MI.getOperand(0).getReg() == X86::ESP &&
         MI.getOperand(1).getReg() == X86::ESP;
}

/// Skip past CFI instructions from the given iterator.
static MachineBasicBlock::iterator
skipCFI(MachineBasicBlock::iterator I, MachineBasicBlock::iterator E) {
  while (I != E && I->getOpcode() == TargetOpcode::CFI_INSTRUCTION)
    ++I;
  return I;
}

/// Adjust ESP-relative operands in a PUSH32r instruction. If the push source
/// references ESP (e.g., push [esp+X]), we need to account for the SUB ESP
/// that now appears before it. In practice, callee-save pushes are register
/// pushes (push esi, push edi) and never reference ESP, but this handles the
/// edge case.
static void adjustPushEspOperands(MachineInstr &MI, int64_t SubAmount) {
  // PUSH32r has a single register operand - it cannot reference ESP memory.
  // Only PUSH32rmm/PUSH32rmr would have memory operands. Callee-save pushes
  // are always PUSH32r, so this is a no-op for them.
  if (MI.getOpcode() != X86::PUSH32r)
    return;
  // Nothing to adjust for a register push.
}

/// Adjust ESP-relative operands in a POP32r instruction. Same reasoning as
/// above - callee-save pops are always register form.
static void adjustPopEspOperands(MachineInstr &MI, int64_t AddAmount) {
  if (MI.getOpcode() != X86::POP32r)
    return;
  // Nothing to adjust for a register pop.
}

bool X86ReorderSubEspPass::reorderPrologue(MachineBasicBlock &MBB) {
  // Pattern to match in the entry block:
  //   [CFI...]
  //   PUSH32r callee-save   ; one or more
  //   [CFI...]
  //   PUSH32r callee-save
  //   [CFI...]
  //   SUB32ri/SUB32ri8 ESP, N
  //   [CFI...]
  //
  // Transform to:
  //   [CFI...]
  //   SUB32ri/SUB32ri8 ESP, N
  //   [CFI for SUB if any]
  //   PUSH32r callee-save
  //   [CFI...]
  //   PUSH32r callee-save
  //   [CFI...]

  auto I = MBB.begin();
  auto E = MBB.end();

  // Skip leading CFI.
  I = skipCFI(I, E);
  if (I == E)
    return false;

  // Collect consecutive callee-save PUSHes (with interspersed CFI).
  SmallVector<MachineInstr *, 4> Pushes;
  auto PushStart = I;

  while (I != E) {
    I = skipCFI(I, E);
    if (I == E)
      break;
    if (!isCalleeSavePush(*I))
      break;
    Pushes.push_back(&*I);
    ++I;
  }

  if (Pushes.empty())
    return false;

  // After the pushes (and any trailing CFI), expect SUB ESP, N.
  I = skipCFI(I, E);
  if (I == E || !isSubEsp(*I))
    return false;

  MachineInstr *SubInstr = &*I;
  int64_t SubAmount = SubInstr->getOperand(2).getImm();

  // Collect any CFI instructions that immediately follow the SUB.
  // These are typically the CFA adjustment for the SUB and need to move with it.
  SmallVector<MachineInstr *, 2> SubCFIs;
  auto AfterSub = std::next(MachineBasicBlock::iterator(SubInstr));
  while (AfterSub != E &&
         AfterSub->getOpcode() == TargetOpcode::CFI_INSTRUCTION) {
    SubCFIs.push_back(&*AfterSub);
    ++AfterSub;
  }

  // Move the SUB (and its CFI) before the first PUSH.
  // First, find the insertion point: just before the first PUSH instruction.
  MachineBasicBlock::iterator InsertPt(Pushes.front());

  // Move SUB before the first push.
  MBB.splice(InsertPt, &MBB, MachineBasicBlock::iterator(SubInstr));

  // Move the SUB's CFI instructions right after the SUB (before first push).
  auto AfterSubNew = std::next(MachineBasicBlock::iterator(SubInstr));
  for (MachineInstr *CFI : SubCFIs) {
    MBB.splice(AfterSubNew, &MBB, MachineBasicBlock::iterator(CFI));
    AfterSubNew = std::next(MachineBasicBlock::iterator(CFI));
  }

  // Adjust ESP-relative operands in the pushes. Each push after the SUB will
  // see ESP already decremented by SubAmount. For callee-save register pushes,
  // this is a no-op.
  for (MachineInstr *Push : Pushes)
    adjustPushEspOperands(*Push, SubAmount);

  return true;
}

bool X86ReorderSubEspPass::reorderEpilogue(MachineBasicBlock &MBB) {
  // Pattern to match at the end of each block (before RET):
  //   ADD32ri/ADD32ri8 ESP, N
  //   [CFI...]
  //   POP32r callee-save
  //   [CFI...]
  //   POP32r callee-save
  //   [CFI...]
  //   RET / RETL
  //
  // Transform to:
  //   POP32r callee-save
  //   [CFI...]
  //   POP32r callee-save
  //   [CFI...]
  //   ADD32ri/ADD32ri8 ESP, N
  //   [CFI for ADD if any]
  //   RET / RETL

  // Find the return instruction at the end of the block.
  auto E = MBB.end();
  if (MBB.empty())
    return false;

  auto RetIt = E;
  --RetIt;

  // Skip past trailing CFI to find the actual terminator.
  while (RetIt != MBB.begin() &&
         RetIt->getOpcode() == TargetOpcode::CFI_INSTRUCTION)
    --RetIt;

  // Must end with a return.
  if (!RetIt->isReturn())
    return false;

  // Walk backwards from the return to find POPs and ADD ESP.
  auto I = RetIt;

  // Collect POPs in reverse order (walking backwards).
  SmallVector<MachineInstr *, 4> Pops;
  while (I != MBB.begin()) {
    --I;
    // Skip CFI.
    while (I != MBB.begin() &&
           I->getOpcode() == TargetOpcode::CFI_INSTRUCTION)
      --I;

    if (isCalleeSavePop(*I)) {
      Pops.push_back(&*I);
    } else {
      break;
    }
  }

  if (Pops.empty())
    return false;

  // I should now point to the ADD ESP instruction.
  // Skip backwards past any CFI that might be between the ADD and the POPs.
  while (I != MBB.begin() &&
         I->getOpcode() == TargetOpcode::CFI_INSTRUCTION)
    --I;

  if (!isAddEsp(*I))
    return false;

  MachineInstr *AddInstr = &*I;
  int64_t AddAmount = AddInstr->getOperand(2).getImm();

  // Collect any CFI instructions that immediately follow the ADD.
  SmallVector<MachineInstr *, 2> AddCFIs;
  auto AfterAdd = std::next(MachineBasicBlock::iterator(AddInstr));
  while (AfterAdd != E &&
         AfterAdd->getOpcode() == TargetOpcode::CFI_INSTRUCTION) {
    // Only collect CFIs up to the first POP.
    bool IsPop = false;
    for (MachineInstr *Pop : Pops) {
      if (&*AfterAdd == Pop) {
        IsPop = true;
        break;
      }
    }
    if (IsPop)
      break;
    AddCFIs.push_back(&*AfterAdd);
    ++AfterAdd;
  }

  // Move the ADD (and its CFI) to just before the RET, after all POPs.
  // Insert point: just before the return instruction.
  MachineBasicBlock::iterator InsertPt(RetIt);

  MBB.splice(InsertPt, &MBB, MachineBasicBlock::iterator(AddInstr));
  auto AfterAddNew = std::next(MachineBasicBlock::iterator(AddInstr));
  for (MachineInstr *CFI : AddCFIs) {
    MBB.splice(AfterAddNew, &MBB, MachineBasicBlock::iterator(CFI));
    AfterAddNew = std::next(MachineBasicBlock::iterator(CFI));
  }

  // Adjust ESP-relative operands in the pops. For callee-save register pops,
  // this is a no-op.
  for (MachineInstr *Pop : Pops)
    adjustPopEspOperands(*Pop, AddAmount);

  return true;
}

bool X86ReorderSubEspPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("reorder_sub_esp"))
    return false;

  bool Changed = false;

  // Reorder prologue in the entry block.
  Changed |= reorderPrologue(MF.front());

  // Reorder epilogue in any block that ends with a return.
  for (MachineBasicBlock &MBB : MF)
    Changed |= reorderEpilogue(MBB);

  return Changed;
}

FunctionPass *llvm::createX86ReorderSubEspPass() {
  return new X86ReorderSubEspPass();
}

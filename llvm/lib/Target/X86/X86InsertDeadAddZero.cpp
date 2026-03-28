//===--- X86InsertDeadAddZero.cpp - Insert dead xor+add on else paths -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: MSVC 6.0 generates the same "add offset" template on both paths
// of a branch, even when one path has offset zero. The result is dead
// arithmetic on the else path:
//
//   ; Taken path: compute real offset
//   mov eax, [ecx+0x48]     ; load base
//   mov edx, [eax+0x08]     ; load offset
//   add eax, edx            ; base + offset
//
//   ; Else path: zero offset (dead arithmetic)
//   xor.s edx, edx          ; edx = 0
//   add.s eax, edx          ; eax += 0 (no-op)
//
// Modern compilers eliminate the dead xor+add. This pass reinserts it at merge
// points where one predecessor has ADD32rr EAX, reg and the other does not.
//
// Gate: function attribute "insert_dead_add_zero" with a value specifying the
// register and an optional filter separated by colon:
//
//   insert_dead_add_zero("edx")             - all merge points (legacy)
//   insert_dead_add_zero("edx:non_return")  - only merge points with successors
//   insert_dead_add_zero("edx:N")           - only the first N merge points
//
// The "non_return" filter matches the MSVC 6.0 pattern: dead arithmetic appears
// at merge points that flow into more computation, not at return blocks.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "x86-insert-dead-add-zero"

namespace {

class X86InsertDeadAddZeroPass : public MachineFunctionPass {
public:
  static char ID;
  X86InsertDeadAddZeroPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 insert dead xor+add zero on else branches";
  }
};

} // end anonymous namespace

char X86InsertDeadAddZeroPass::ID = 0;

/// Parse register name from attribute value. Returns the X86 register or 0.
static Register parseRegName(StringRef Name) {
  return StringSwitch<Register>(Name.lower())
      .Case("eax", X86::EAX)
      .Case("ebx", X86::EBX)
      .Case("ecx", X86::ECX)
      .Case("edx", X86::EDX)
      .Case("esi", X86::ESI)
      .Case("edi", X86::EDI)
      .Case("ebp", X86::EBP)
      .Default(0);
}

/// Check if a block contains ADD32rr or ADD32rr_REV with EAX as destination
/// and the specified register as the source operand.
static bool blockHasAddEax(const MachineBasicBlock &MBB, Register ZeroReg) {
  for (const MachineInstr &MI : MBB) {
    unsigned Opc = MI.getOpcode();
    if (Opc != X86::ADD32rr && Opc != X86::ADD32rr_REV)
      continue;
    // ADD32rr: operand 0 = dst (tied to op1), op1 = src1, op2 = src2
    if (MI.getOperand(0).getReg() == X86::EAX &&
        MI.getOperand(2).getReg() == ZeroReg)
      return true;
  }
  return false;
}

/// Check if a block is a return block (contains a RET instruction).
static bool isReturnBlock(const MachineBasicBlock &MBB) {
  for (const MachineInstr &MI : MBB) {
    if (MI.isReturn())
      return true;
  }
  return false;
}

/// Filter mode for which merge points receive dead add insertion.
enum class MergeFilter {
  All,       // Insert at every qualifying merge point (legacy behavior).
  NonReturn, // Skip merge points that are return blocks.
  Count,     // Insert at only the first N qualifying merge points.
};

/// Parse the attribute value into register, filter mode, and count limit.
/// Format: "reg" | "reg:non_return" | "reg:N"
/// Returns true on success.
static bool parseAttrValue(StringRef AttrVal, Register &ZeroReg,
                           MergeFilter &Filter, unsigned &CountLimit) {
  Filter = MergeFilter::All;
  CountLimit = 0;

  auto [RegPart, FilterPart] = AttrVal.split(':');
  ZeroReg = parseRegName(RegPart);
  if (!ZeroReg)
    return false;

  if (FilterPart.empty())
    return true;

  if (FilterPart == "non_return") {
    Filter = MergeFilter::NonReturn;
    return true;
  }

  // Try to parse as a count.
  if (!FilterPart.getAsInteger(10, CountLimit) && CountLimit > 0) {
    Filter = MergeFilter::Count;
    return true;
  }

  return false;
}

bool X86InsertDeadAddZeroPass::runOnMachineFunction(MachineFunction &MF) {
  const Function &F = MF.getFunction();
  if (!F.hasFnAttribute("insert_dead_add_zero"))
    return false;

  StringRef AttrVal =
      F.getFnAttribute("insert_dead_add_zero").getValueAsString();

  Register ZeroReg;
  MergeFilter Filter;
  unsigned CountLimit;
  if (!parseAttrValue(AttrVal, ZeroReg, Filter, CountLimit))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;
  unsigned InsertCount = 0;

  for (MachineBasicBlock &MBB : MF) {
    // Only look at merge points: blocks with 2+ predecessors.
    if (MBB.pred_size() < 2)
      continue;

    // Apply filter: skip return blocks when using non_return mode.
    if (Filter == MergeFilter::NonReturn && isReturnBlock(MBB))
      continue;

    // Apply filter: stop after inserting at CountLimit merge points.
    if (Filter == MergeFilter::Count && InsertCount >= CountLimit)
      break;

    // Check if any predecessor has the ADD32rr EAX, ZeroReg pattern.
    SmallVector<MachineBasicBlock *, 4> HasAdd;
    SmallVector<MachineBasicBlock *, 4> LacksAdd;

    for (MachineBasicBlock *Pred : MBB.predecessors()) {
      if (blockHasAddEax(*Pred, ZeroReg))
        HasAdd.push_back(Pred);
      else
        LacksAdd.push_back(Pred);
    }

    // We need at least one predecessor with ADD and at least one without.
    if (HasAdd.empty() || LacksAdd.empty())
      continue;

    // Insert XOR32rr_REV + ADD32rr_REV at the end of each predecessor that
    // lacks the ADD, just before the terminator.
    for (MachineBasicBlock *Pred : LacksAdd) {
      // Find the insertion point: before the first terminator.
      auto InsertPt = Pred->getFirstTerminator();
      DebugLoc DL;
      if (InsertPt != Pred->end())
        DL = InsertPt->getDebugLoc();

      // XOR32rr_REV ZeroReg, ZeroReg, ZeroReg  (sets ZeroReg = 0)
      BuildMI(*Pred, InsertPt, DL, TII->get(X86::XOR32rr_REV), ZeroReg)
          .addReg(ZeroReg, RegState::Undef)
          .addReg(ZeroReg, RegState::Undef);

      // ADD32rr_REV EAX, EAX, ZeroReg  (eax += 0, dead arithmetic)
      BuildMI(*Pred, InsertPt, DL, TII->get(X86::ADD32rr_REV), X86::EAX)
          .addReg(X86::EAX)
          .addReg(ZeroReg);

      LLVM_DEBUG(dbgs() << "  Inserted dead XOR+ADD in " << Pred->getName()
                        << " before merge into " << MBB.getName() << "\n");
      Changed = true;
    }

    ++InsertCount;
  }

  return Changed;
}

FunctionPass *llvm::createX86InsertDeadAddZeroPass() {
  return new X86InsertDeadAddZeroPass();
}

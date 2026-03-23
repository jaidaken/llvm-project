//===--- X86ReorderStores.cpp - Reorder stores to match MSVC 6.0 ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: MSVC 6.0 emits stores in source/declaration order, which is
// often NOT ascending by memory offset. Clang sorts independent stores to
// the same base register by ascending offset.
//
// This pass reorders consecutive stores to the same base register within a
// basic block to match a user-specified offset order given by the
// "store_order" function attribute.
//
// Example attribute:
//   __attribute__((store_order("0x2c,0x28,0x24,0x1c,0x18,0x14,0x0c,0x08,0x04")))
//
// The pass:
// 1. Parses the comma-separated hex offsets from the attribute.
// 2. Scans each basic block for consecutive stores to the same base register.
// 3. Reorders those stores to match the specified offset sequence.
// 4. Preserves data dependencies (won't reorder a store before a defining
//    instruction for its source operand).
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "x86-reorder-stores"

namespace {
class X86ReorderStoresPass : public MachineFunctionPass {
public:
  static char ID;
  X86ReorderStoresPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 reorder stores to match MSVC 6.0 order";
  }

private:
  const X86InstrInfo *TII = nullptr;

  /// Parse "0x2c,0x28,0x24,..." into a vector of offsets.
  static bool parseOffsets(StringRef AttrVal,
                           SmallVectorImpl<int64_t> &Offsets);

  /// Check if an instruction is a store we can reorder.
  static bool isReorderableStore(const MachineInstr &MI);

  /// Get the memory base register from a store instruction.
  static Register getStoreBase(const MachineInstr &MI);

  /// Get the displacement from a store instruction.
  static int64_t getStoreDisp(const MachineInstr &MI);

  /// Check if moving a store before another would violate data dependencies.
  /// Returns true if MI defines a register that any instruction between
  /// RangeBegin (inclusive) and RangeEnd (exclusive) uses.
  static bool hasDependencyBetween(const MachineInstr &MI,
                                   MachineBasicBlock::iterator RangeBegin,
                                   MachineBasicBlock::iterator RangeEnd);

  bool reorderStoresInBlock(MachineBasicBlock &MBB,
                            const SmallVectorImpl<int64_t> &Offsets);
};
} // end anonymous namespace

char X86ReorderStoresPass::ID = 0;

bool X86ReorderStoresPass::parseOffsets(StringRef AttrVal,
                                        SmallVectorImpl<int64_t> &Offsets) {
  SmallVector<StringRef, 16> Parts;
  AttrVal.split(Parts, ',');

  for (StringRef Part : Parts) {
    Part = Part.trim();
    if (Part.empty())
      continue;

    int64_t Val;
    // Handle "0x" prefix for hex values.
    StringRef NumStr = Part;
    if (NumStr.starts_with("0x") || NumStr.starts_with("0X")) {
      NumStr = NumStr.drop_front(2);
      if (NumStr.getAsInteger(16, Val))
        return false;
    } else {
      if (NumStr.getAsInteger(10, Val))
        return false;
    }
    Offsets.push_back(Val);
  }

  return !Offsets.empty();
}

bool X86ReorderStoresPass::isReorderableStore(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case X86::MOV32mr:
  case X86::MOV16mr:
  case X86::MOV8mr:
  case X86::MOVSSmr:
  case X86::MOVSDmr:
  case X86::MOV32mi:
  case X86::MOV16mi:
  case X86::MOV8mi:
    return true;
  default:
    return false;
  }
}

Register X86ReorderStoresPass::getStoreBase(const MachineInstr &MI) {
  // MOVxxmr/MOVxxmi operand layout: base(0), scale(1), index(2), disp(3),
  // segment(4), src(5)
  return MI.getOperand(0).getReg();
}

int64_t X86ReorderStoresPass::getStoreDisp(const MachineInstr &MI) {
  return MI.getOperand(3).getImm();
}

bool X86ReorderStoresPass::hasDependencyBetween(
    const MachineInstr &MI, MachineBasicBlock::iterator RangeBegin,
    MachineBasicBlock::iterator RangeEnd) {
  // Collect registers that MI's source operand uses.
  // For MOVxxmr: operand 5 is the source register.
  // For MOVxxmi: operand 5 is an immediate, no register dependency.
  // Also check the base/index registers of MI's memory operand.
  SmallVector<Register, 4> MIUses;

  // Source operand.
  if (MI.getOperand(5).isReg() && MI.getOperand(5).getReg() != X86::NoRegister)
    MIUses.push_back(MI.getOperand(5).getReg());

  // Base register.
  if (MI.getOperand(0).isReg() && MI.getOperand(0).getReg() != X86::NoRegister)
    MIUses.push_back(MI.getOperand(0).getReg());

  // Index register.
  if (MI.getOperand(2).isReg() && MI.getOperand(2).getReg() != X86::NoRegister)
    MIUses.push_back(MI.getOperand(2).getReg());

  // Check if any instruction in the range defines a register that MI uses.
  for (auto It = RangeBegin; It != RangeEnd; ++It) {
    for (const MachineOperand &MO : It->operands()) {
      if (!MO.isReg() || !MO.isDef())
        continue;
      Register DefReg = MO.getReg();
      if (DefReg == X86::NoRegister)
        continue;
      for (Register Use : MIUses) {
        if (Use == DefReg)
          return true;
      }
    }
  }

  return false;
}

bool X86ReorderStoresPass::reorderStoresInBlock(
    MachineBasicBlock &MBB, const SmallVectorImpl<int64_t> &Offsets) {
  bool Changed = false;

  auto I = MBB.begin();
  while (I != MBB.end()) {
    // Skip non-store instructions.
    if (!isReorderableStore(*I)) {
      ++I;
      continue;
    }

    // Found a store. Collect consecutive stores to the same base register.
    Register Base = getStoreBase(*I);
    SmallVector<MachineInstr *, 16> Stores;
    auto SeqStart = I;

    while (I != MBB.end() && isReorderableStore(*I) &&
           getStoreBase(*I) == Base) {
      Stores.push_back(&*I);
      ++I;
    }

    // Need at least 2 stores to reorder.
    if (Stores.size() < 2)
      continue;

    // Build a map from offset to position in the desired order.
    // The attribute specifies the target order of offsets.
    DenseMap<int64_t, unsigned> DesiredPosition;
    for (unsigned Idx = 0; Idx < Offsets.size(); ++Idx)
      DesiredPosition[Offsets[Idx]] = Idx;

    // Check if all stores in this group have offsets present in the attribute.
    bool AllFound = true;
    for (MachineInstr *Store : Stores) {
      int64_t Disp = getStoreDisp(*Store);
      if (!DesiredPosition.count(Disp)) {
        AllFound = false;
        break;
      }
    }

    if (!AllFound) {
      LLVM_DEBUG(dbgs() << "ReorderStores: skipping group - not all offsets "
                           "found in attribute\n");
      continue;
    }

    // Sort stores by the desired position from the attribute.
    SmallVector<MachineInstr *, 16> Sorted(Stores);
    llvm::sort(Sorted, [&](MachineInstr *A, MachineInstr *B) {
      return DesiredPosition[getStoreDisp(A)] <
             DesiredPosition[getStoreDisp(B)];
    });

    // Check if already in the desired order.
    bool AlreadyOrdered = true;
    for (unsigned Idx = 0; Idx < Stores.size(); ++Idx) {
      if (Stores[Idx] != Sorted[Idx]) {
        AlreadyOrdered = false;
        break;
      }
    }

    if (AlreadyOrdered)
      continue;

    // Verify no data dependencies would be violated by the reorder.
    // Since these are all stores to the same base register with different
    // offsets, the main concern is that a store's source register might be
    // defined by a preceding store's side effect. For simple MOV stores
    // to the same base, this is uncommon, but we check anyway.
    //
    // We check conservatively: for each store in the new order, verify it
    // doesn't depend on any store that would come after it in the new order
    // but came before it in the original order.
    bool Safe = true;
    for (unsigned NewIdx = 0; NewIdx < Sorted.size() && Safe; ++NewIdx) {
      MachineInstr *StoreMI = Sorted[NewIdx];
      // Find the original position of this store.
      unsigned OldIdx = 0;
      for (unsigned J = 0; J < Stores.size(); ++J) {
        if (Stores[J] == StoreMI) {
          OldIdx = J;
          break;
        }
      }

      // If this store moved earlier (NewIdx < OldIdx), check that none of the
      // instructions between NewIdx and OldIdx in the original order define
      // registers this store uses.
      if (NewIdx < OldIdx) {
        auto RangeBegin = MachineBasicBlock::iterator(Stores[NewIdx]);
        auto RangeEnd = MachineBasicBlock::iterator(StoreMI);
        if (hasDependencyBetween(*StoreMI, RangeBegin, RangeEnd)) {
          Safe = false;
          LLVM_DEBUG(dbgs() << "ReorderStores: dependency prevents reorder\n");
        }
      }
    }

    if (!Safe)
      continue;

    // Perform the reorder by splicing stores into the desired order.
    // Insert point is at SeqStart (the position of the first store).
    MachineBasicBlock::iterator InsertPt = SeqStart;
    for (MachineInstr *StoreMI : Sorted) {
      if (&*InsertPt == StoreMI) {
        // Already in the right place, advance insert point.
        ++InsertPt;
        continue;
      }
      MBB.splice(InsertPt, &MBB, MachineBasicBlock::iterator(StoreMI));
      // InsertPt stays valid - splice inserts before it.
    }

    LLVM_DEBUG(dbgs() << "ReorderStores: reordered " << Stores.size()
                      << " stores in " << MBB.getParent()->getName() << "\n");
    Changed = true;
  }

  return Changed;
}

bool X86ReorderStoresPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("store_order"))
    return false;

  StringRef AttrVal =
      MF.getFunction().getFnAttribute("store_order").getValueAsString();
  if (AttrVal.empty())
    return false;

  SmallVector<int64_t, 16> Offsets;
  if (!parseOffsets(AttrVal, Offsets))
    return false;

  TII = MF.getSubtarget<X86Subtarget>().getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF)
    Changed |= reorderStoresInBlock(MBB, Offsets);

  return Changed;
}

FunctionPass *llvm::createX86ReorderStoresPass() {
  return new X86ReorderStoresPass();
}

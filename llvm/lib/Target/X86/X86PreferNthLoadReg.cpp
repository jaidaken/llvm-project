//===--- X86PreferNthLoadReg.cpp - Force Nth load into specific register ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: Post-regalloc pass that ensures the Nth memory load in the entry
// block produces its result in a specific register.
//
// MSVC 6.0 has specific register preferences for memory loads that go beyond
// just the first load. For example:
//   - The 2nd load might need to be in EAX (overwriting a flags value)
//   - The 3rd load might need to be in EDX
//
// The pass parses the "prefer_nth_load_reg" function attribute, which has the
// format "2:eax,3:edx" meaning the 2nd load goes to EAX, the 3rd to EDX.
//
// For each specified Nth load, if its destination doesn't match the requested
// register, the pass inserts a MOV32rr right after the load and locally
// rewrites forward uses of the load's result to use the desired register,
// stopping when the original register is redefined by another instruction.
//
// This local approach (vs. a global swap) is safe with chained loads and
// does not conflict with forced_callee_saves PUSH/POP instructions.
//
// Gated on the "prefer_nth_load_reg" function attribute.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-nth-load-reg"
#define PASS_NAME "X86 prefer Nth load into specific register"

namespace {
class X86PreferNthLoadRegPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferNthLoadRegPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return PASS_NAME; }
};
} // end anonymous namespace

char X86PreferNthLoadRegPass::ID = 0;

/// Map a register from OldReg32's family to NewReg32's family.
/// For example, if OldReg32=EAX, NewReg32=EDX, then AL -> DL, AX -> DX, etc.
/// Returns 0 if the register doesn't belong to OldReg32's family.
static unsigned mapRegToFamily(unsigned Reg, unsigned OldReg32,
                               unsigned NewReg32) {
  if (Reg == OldReg32)
    return NewReg32;

  // Build sub-register tables for both families.
  struct RegFamily {
    unsigned R32, R16, RLo, RHi;
  };
  auto getFamily = [](unsigned R32) -> RegFamily {
    switch (R32) {
    case X86::EAX: return {X86::EAX, X86::AX, X86::AL, X86::AH};
    case X86::EDX: return {X86::EDX, X86::DX, X86::DL, X86::DH};
    case X86::ECX: return {X86::ECX, X86::CX, X86::CL, X86::CH};
    case X86::EBX: return {X86::EBX, X86::BX, X86::BL, X86::BH};
    case X86::ESI: return {X86::ESI, X86::SI, X86::SIL, 0};
    case X86::EDI: return {X86::EDI, X86::DI, X86::DIL, 0};
    case X86::EBP: return {X86::EBP, X86::BP, X86::BPL, 0};
    default: return {0, 0, 0, 0};
    }
  };

  RegFamily Old = getFamily(OldReg32);
  RegFamily New = getFamily(NewReg32);
  if (Old.R32 == 0 || New.R32 == 0)
    return 0;

  if (Reg == Old.R16) return New.R16;
  if (Reg == Old.RLo) return New.RLo;
  if (Old.RHi && Reg == Old.RHi) return New.RHi ? New.RHi : 0;

  return 0;
}

/// Parse a register name string to an X86 32-bit register.
/// Supports: eax, ebx, ecx, edx, esi, edi, ebp.
static unsigned parseReg32(StringRef Name) {
  return StringSwitch<unsigned>(Name.lower())
      .Case("eax", X86::EAX)
      .Case("ebx", X86::EBX)
      .Case("ecx", X86::ECX)
      .Case("edx", X86::EDX)
      .Case("esi", X86::ESI)
      .Case("edi", X86::EDI)
      .Case("ebp", X86::EBP)
      .Default(0);
}

/// Parse the spec string "2:eax,3:edx" into a vector of (N, Reg32) pairs.
static bool parseSpec(StringRef Spec,
                      SmallVectorImpl<std::pair<unsigned, unsigned>> &Entries) {
  SmallVector<StringRef, 4> Parts;
  Spec.split(Parts, ',');
  for (StringRef Part : Parts) {
    Part = Part.trim();
    if (Part.empty())
      continue;
    auto [NStr, RegStr] = Part.split(':');
    unsigned N;
    if (NStr.trim().getAsInteger(10, N) || N == 0)
      return false;
    unsigned Reg32 = parseReg32(RegStr.trim());
    if (Reg32 == 0)
      return false;
    Entries.push_back({N, Reg32});
  }
  return !Entries.empty();
}

/// Check if an instruction is a prologue/epilogue PUSH/POP from
/// forced_callee_saves. These have FrameSetup or FrameDestroy flags.
static bool isPrologueEpilogue(const MachineInstr &MI) {
  return MI.getFlag(MachineInstr::FrameSetup) ||
         MI.getFlag(MachineInstr::FrameDestroy);
}

/// Check if the instruction defines (writes to) a register that overlaps
/// with the given 32-bit register.
static bool definesReg(const MachineInstr &MI, unsigned Reg32,
                       const TargetRegisterInfo *TRI) {
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isReg() && MO.isDef() && MO.getReg() != 0 &&
        TRI->regsOverlap(MO.getReg(), Reg32))
      return true;
  }
  // Also check implicit defs.
  if (MI.getDesc().hasImplicitDefOfPhysReg(Reg32, TRI))
    return true;
  return false;
}

bool X86PreferNthLoadRegPass::runOnMachineFunction(MachineFunction &MF) {
  Attribute Attr =
      MF.getFunction().getFnAttribute("prefer_nth_load_reg");
  if (!Attr.isStringAttribute())
    return false;

  StringRef Spec = Attr.getValueAsString();
  SmallVector<std::pair<unsigned, unsigned>, 4> Entries;
  if (!parseSpec(Spec, Entries))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  MachineBasicBlock &EntryMBB = MF.front();

  // Collect all MOV32rm instructions in entry block order.
  SmallVector<MachineInstr *, 8> Loads;
  for (MachineInstr &MI : EntryMBB) {
    if (MI.isPseudo())
      continue;
    if (MI.getOpcode() == X86::MOV32rm)
      Loads.push_back(&MI);
  }

  // Sort entries by N to process in order.
  llvm::sort(Entries, [](const auto &A, const auto &B) {
    return A.first < B.first;
  });

  bool Changed = false;
  for (auto &[N, DesiredReg] : Entries) {
    // N is 1-based.
    if (N > Loads.size())
      continue;

    MachineInstr *LoadMI = Loads[N - 1];
    Register ActualDest = LoadMI->getOperand(0).getReg();

    if (ActualDest == DesiredReg)
      continue; // Already in the right register.

    LLVM_DEBUG(dbgs() << "PreferNthLoadReg: load #" << N << " in "
                      << MF.getName() << ": " << printReg(ActualDest, TRI)
                      << " -> " << printReg(DesiredReg, TRI) << "\n");

    // Insert MOV32rr DesiredReg, ActualDest right after the load.
    auto InsertPt = std::next(MachineBasicBlock::iterator(LoadMI));
    BuildMI(EntryMBB, InsertPt, LoadMI->getDebugLoc(),
            TII->get(X86::MOV32rr), DesiredReg)
        .addReg(ActualDest);

    // Walk forward from the insertion point through the entry block and
    // rewrite uses of ActualDest to DesiredReg. Stop when ActualDest is
    // redefined by another instruction (meaning a new value is written).
    for (auto It = InsertPt, E = EntryMBB.end(); It != E; ++It) {
      MachineInstr &MI = *It;

      // Skip prologue/epilogue PUSH/POP from forced_callee_saves.
      if (isPrologueEpilogue(MI))
        continue;

      // If this instruction redefines ActualDest, stop rewriting.
      // The register now holds a different value.
      if (definesReg(MI, ActualDest, TRI))
        break;

      // Rewrite uses of ActualDest (and its sub-registers) to DesiredReg.
      for (MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || MO.getReg() == 0)
          continue;
        if (!MO.isUse())
          continue;
        unsigned Mapped = mapRegToFamily(MO.getReg(), ActualDest, DesiredReg);
        if (Mapped != 0) {
          MO.setReg(Mapped);
          Changed = true;
        }
      }
    }

    Changed = true;
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferNthLoadRegPass() {
  return new X86PreferNthLoadRegPass();
}

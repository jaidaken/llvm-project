//===--- X86PreferNthLoadReg.cpp - Force Nth load into specific register ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: Post-regalloc pass that forces the Nth memory load in the entry
// block into a specific register.
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
// register, the pass does a global register swap throughout the entire function.
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

/// Get the 16-bit, low-8, and high-8 sub-registers for a 32-bit GPR.
/// Returns false if the register is not a supported 32-bit GPR.
static bool getSubRegs(unsigned Reg32, unsigned &Reg16, unsigned &RegLo,
                       unsigned &RegHi) {
  switch (Reg32) {
  case X86::EAX: Reg16 = X86::AX;  RegLo = X86::AL;  RegHi = X86::AH;  return true;
  case X86::EDX: Reg16 = X86::DX;  RegLo = X86::DL;  RegHi = X86::DH;  return true;
  case X86::ECX: Reg16 = X86::CX;  RegLo = X86::CL;  RegHi = X86::CH;  return true;
  case X86::EBX: Reg16 = X86::BX;  RegLo = X86::BL;  RegHi = X86::BH;  return true;
  case X86::ESI: Reg16 = X86::SI;  RegLo = X86::SIL; RegHi = 0;        return true;
  case X86::EDI: Reg16 = X86::DI;  RegLo = X86::DIL; RegHi = 0;        return true;
  case X86::EBP: Reg16 = X86::BP;  RegLo = X86::BPL; RegHi = 0;        return true;
  default: return false;
  }
}

/// Swap register Reg between the OldReg32 family and the NewReg32 family.
/// Handles 32-bit, 16-bit, and 8-bit sub-register variants.
/// Returns Reg unchanged if it belongs to neither family.
static unsigned swapReg(unsigned Reg, unsigned OldReg32, unsigned NewReg32) {
  if (Reg == OldReg32) return NewReg32;
  if (Reg == NewReg32) return OldReg32;

  unsigned Old16, OldLo, OldHi;
  unsigned New16, NewLo, NewHi;
  if (!getSubRegs(OldReg32, Old16, OldLo, OldHi) ||
      !getSubRegs(NewReg32, New16, NewLo, NewHi))
    return Reg;

  if (Reg == Old16) return New16;
  if (Reg == New16) return Old16;
  if (Reg == OldLo) return NewLo;
  if (Reg == NewLo) return OldLo;
  if (OldHi && Reg == OldHi) return NewHi ? NewHi : Reg;
  if (NewHi && Reg == NewHi) return OldHi ? OldHi : Reg;

  return Reg;
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

bool X86PreferNthLoadRegPass::runOnMachineFunction(MachineFunction &MF) {
  Attribute Attr =
      MF.getFunction().getFnAttribute("prefer_nth_load_reg");
  if (!Attr.isStringAttribute())
    return false;

  StringRef Spec = Attr.getValueAsString();
  SmallVector<std::pair<unsigned, unsigned>, 4> Entries;
  if (!parseSpec(Spec, Entries))
    return false;

  MachineBasicBlock &EntryMBB = MF.front();

  // Collect all MOV32rm instructions in entry block order.
  SmallVector<MachineInstr *, 8> Loads;
  for (MachineInstr &MI : EntryMBB) {
    if (MI.isPseudo())
      continue;
    if (MI.getOpcode() == X86::MOV32rm)
      Loads.push_back(&MI);
  }

  // Process each entry. Apply swaps in order of ascending N so that earlier
  // swaps don't interfere with later load identification (we track loads by
  // position, and a global swap doesn't change instruction order).
  //
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
    Register CurReg = LoadMI->getOperand(0).getReg();

    if (CurReg == DesiredReg)
      continue; // Already in the right register.

    // Swap CurReg <-> DesiredReg throughout the entire function.
    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : MBB) {
        for (MachineOperand &MO : MI.operands()) {
          if (!MO.isReg())
            continue;
          unsigned NewReg = swapReg(MO.getReg(), CurReg, DesiredReg);
          if (NewReg != MO.getReg()) {
            MO.setReg(NewReg);
            Changed = true;
          }
        }
      }
    }

    // Update the Loads vector: the swap changed registers globally, so any
    // load that was targeting DesiredReg is now targeting CurReg, and vice
    // versa. We need to keep the Loads vector accurate for subsequent entries.
    // (The instruction pointers are still valid - only register operands changed.)
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferNthLoadRegPass() {
  return new X86PreferNthLoadRegPass();
}

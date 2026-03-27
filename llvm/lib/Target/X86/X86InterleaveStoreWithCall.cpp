//===--- X86InterleaveStoreWithCall.cpp - Interleave stores in vtable calls ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: MSVC 6.0 places field stores BETWEEN push arguments and the
// vtable call. Clang schedules stores BEFORE the vtable load, since the store
// doesn't depend on the vtable register and moving it earlier is valid for ILP.
//
// Clang generates:
//   mov [esi+0x58], dx     ; store field (BEFORE vtable load)
//   mov eax, [esi]         ; vtable load
//   push 0x6b              ; push arg
//   mov ecx, esi           ; this setup
//   call [eax+0x8e8]       ; vtable call
//
// MSVC 6.0 generates:
//   mov eax, [esi]         ; vtable load
//   push 0x6b              ; push arg
//   mov [esi+0x58], dx     ; store field (BETWEEN push and call)
//   mov ecx, esi           ; this setup
//   call [eax+0x8e8]       ; vtable call
//
// This pass finds store instructions that are before the vtable load and moves
// them into the push/call sequence. The exact placement is controlled by the
// string argument of the "interleave_store_with_call" function attribute:
//
//   "after_push"   - place the store after all PUSHes, before ECX setup/call
//   "before_call"  - place the store immediately before the CALL instruction
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

#define DEBUG_TYPE "x86-interleave-store-with-call"

namespace {
class X86InterleaveStoreWithCallPass : public MachineFunctionPass {
public:
  static char ID;
  X86InterleaveStoreWithCallPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 interleave store with vtable call";
  }

private:
  const X86InstrInfo *TII = nullptr;

  bool processBlock(MachineBasicBlock &MBB, StringRef Mode);
};
} // end anonymous namespace

char X86InterleaveStoreWithCallPass::ID = 0;

// ---------------------------------------------------------------------------
// Helper predicates
// ---------------------------------------------------------------------------

/// Check if an instruction is an indirect call through memory.
static bool isCallMem(const MachineInstr &MI) {
  return MI.getOpcode() == X86::CALL32m;
}

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

/// Check if an instruction is a store to memory (register or immediate source).
static bool isMemStore(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case X86::MOV8mr:
  case X86::MOV16mr:
  case X86::MOV32mr:
  case X86::MOV8mi:
  case X86::MOV16mi:
  case X86::MOV32mi:
    return true;
  default:
    return false;
  }
}

/// Check if a store writes to [BaseReg + offset] (simple addressing mode).
static bool isStoreToBase(const MachineInstr &MI, Register BaseReg) {
  // MOVxmr operand layout: [base, scale, index, disp, seg, src]
  return MI.getOperand(0).isReg() &&
         MI.getOperand(0).getReg() == BaseReg &&
         MI.getOperand(1).getImm() == 1 &&       // scale = 1
         MI.getOperand(2).getReg() == X86::NoRegister; // no index
}

/// Check if a store's source register uses any part of the given register.
/// For example, if VtReg is EAX, a store using AL/AX/EAX would conflict.
static bool storeUsesReg(const MachineInstr &MI, Register VtReg,
                         const X86InstrInfo *TII) {
  // The source operand of MOVxmr is operand 5.
  Register SrcReg = MI.getOperand(5).getReg();
  if (SrcReg == X86::NoRegister)
    return false;

  const TargetRegisterInfo *TRI = &TII->getRegisterInfo();
  return TRI->regsOverlap(SrcReg, VtReg);
}

/// Check if a MOV32rm loads the vtable pointer: MOV32rm Dest, [Base + disp].
/// Returns the base register (thisReg) if it's a simple load.
static bool isVtableLoad(const MachineInstr &MI, Register ExpectedDest) {
  return MI.getOpcode() == X86::MOV32rm &&
         MI.getOperand(0).getReg() == ExpectedDest &&
         MI.getOperand(1).isReg() &&
         MI.getOperand(2).getImm() == 1 &&        // scale = 1
         MI.getOperand(3).getReg() == X86::NoRegister; // no index
}

/// Check if an instruction is a MOV32rr copying to ECX (this-pointer setup).
static bool isEcxSetup(const MachineInstr &MI) {
  return (MI.getOpcode() == X86::MOV32rr ||
          MI.getOpcode() == X86::MOV32rr_REV) &&
         MI.getOperand(0).getReg() == X86::ECX;
}

// ---------------------------------------------------------------------------
// Main transformation
// ---------------------------------------------------------------------------

bool X86InterleaveStoreWithCallPass::processBlock(MachineBasicBlock &MBB,
                                                   StringRef Mode) {
  bool Changed = false;

  for (auto CallIt = MBB.begin(), E = MBB.end(); CallIt != E; ++CallIt) {
    MachineInstr &CallMI = *CallIt;

    // Step 1: Find an indirect call through memory: CALL32m [reg + disp].
    if (!isCallMem(CallMI))
      continue;
    if (!CallMI.getOperand(0).isReg())
      continue;
    Register CallBase = CallMI.getOperand(0).getReg();
    if (CallBase == X86::NoRegister || CallBase == X86::ESP)
      continue;

    // Step 2: Walk backward from the CALL, collecting the instruction sequence.
    // We expect to see (closest to CALL first):
    //   [optional] MOV32rr ECX, thisReg  (this-pointer setup)
    //   [0..N]     PUSHes
    //   [1]        MOV32rm CallBase, [thisReg + vtDisp]  (vtable load)
    //   [before]   store instructions to [thisReg + N]
    //
    // We scan backward collecting up to 16 preceding non-pseudo instructions.
    SmallVector<MachineInstr *, 16> Prev;
    {
      auto WalkIt = MachineBasicBlock::iterator(CallMI);
      while (Prev.size() < 16 && WalkIt != MBB.begin()) {
        --WalkIt;
        if (WalkIt->isTerminator())
          break;
        if (!WalkIt->isPseudo())
          Prev.push_back(&*WalkIt);
      }
    }
    // Prev[0] = instruction right before CALL, etc.

    if (Prev.size() < 2)
      continue;

    int cursor = 0;

    // Step 3: Optional ECX setup (MOV32rr ECX, thisReg).
    int EcxIdx = -1;
    if (cursor < (int)Prev.size() && isEcxSetup(*Prev[cursor])) {
      EcxIdx = cursor;
      cursor++;
    }

    // Step 4: Collect PUSHes.
    SmallVector<MachineInstr *, 4> Pushes;
    while (cursor < (int)Prev.size() && isAnyPush(*Prev[cursor])) {
      Pushes.push_back(Prev[cursor]);
      cursor++;
    }

    if (cursor >= (int)Prev.size())
      continue;

    // Step 5: Find the vtable load: MOV32rm CallBase, [thisReg + vtDisp].
    int VtableIdx = -1;
    if (isVtableLoad(*Prev[cursor], CallBase)) {
      VtableIdx = cursor;
      cursor++;
    } else {
      continue; // vtable load is required
    }

    MachineInstr &VtableMI = *Prev[VtableIdx];
    Register ThisReg = VtableMI.getOperand(1).getReg();
    if (ThisReg == X86::NoRegister || ThisReg == X86::ESP)
      continue;

    // Step 6: Scan backward from the vtable load for store instructions
    // that write to [thisReg + offset] and don't depend on CallBase.
    // These are the stores that Clang hoisted above the vtable load.
    SmallVector<MachineInstr *, 4> StoresToMove;

    while (cursor < (int)Prev.size()) {
      MachineInstr *MI = Prev[cursor];

      if (!isMemStore(*MI))
        break;
      if (!isStoreToBase(*MI, ThisReg))
        break;
      if (storeUsesReg(*MI, CallBase, TII))
        break;

      StoresToMove.push_back(MI);
      cursor++;
    }

    if (StoresToMove.empty())
      continue;

    LLVM_DEBUG(dbgs() << "InterleaveStoreWithCall: found "
                      << StoresToMove.size() << " store(s) to move in "
                      << MBB.getParent()->getName()
                      << " (mode=" << Mode << ")\n");

    // Step 7: Move stores to the target position.
    // "after_push"  - after all pushes, before ECX setup (or before call if
    //                 no ECX setup)
    // "before_call" - immediately before the CALL instruction
    MachineBasicBlock::iterator InsertPt;

    if (Mode == "before_call") {
      // Insert right before the CALL.
      InsertPt = MachineBasicBlock::iterator(CallMI);
    } else {
      // "after_push" (default): insert after the last push, before ECX/call.
      if (EcxIdx >= 0) {
        // Insert before ECX setup.
        InsertPt = MachineBasicBlock::iterator(*Prev[EcxIdx]);
      } else {
        // No ECX setup found - insert before the CALL.
        InsertPt = MachineBasicBlock::iterator(CallMI);
      }
    }

    // Move stores in reverse order (StoresToMove[0] is closest to the vtable
    // load, which was the last store before it). We want to preserve their
    // relative order, so we insert each one at the same InsertPt, building
    // the sequence from last to first.
    for (int i = (int)StoresToMove.size() - 1; i >= 0; --i) {
      MachineInstr *StoreMI = StoresToMove[i];
      MBB.splice(InsertPt, &MBB,
                 MachineBasicBlock::iterator(*StoreMI),
                 std::next(MachineBasicBlock::iterator(*StoreMI)));
    }

    Changed = true;
  }

  return Changed;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

bool X86InterleaveStoreWithCallPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("interleave_store_with_call"))
    return false;

  StringRef Mode =
      MF.getFunction()
          .getFnAttribute("interleave_store_with_call")
          .getValueAsString();
  if (Mode.empty())
    return false;

  if (Mode != "after_push" && Mode != "before_call") {
    LLVM_DEBUG(dbgs() << "InterleaveStoreWithCall: unknown mode '"
                      << Mode << "', skipping\n");
    return false;
  }

  TII = MF.getSubtarget<X86Subtarget>().getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF)
    Changed |= processBlock(MBB, Mode);

  return Changed;
}

FunctionPass *llvm::createX86InterleaveStoreWithCallPass() {
  return new X86InterleaveStoreWithCallPass();
}

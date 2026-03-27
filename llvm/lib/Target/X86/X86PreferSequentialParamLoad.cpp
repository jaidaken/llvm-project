//===--- X86PreferSequentialParamLoad.cpp - Sequential stack param loads ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: MSVC 6.0 loads stack parameters one at a time in ascending
// ESP offset order (param1 first), interleaving each load with its
// immediate use. Clang loads multiple parameters upfront in an order
// determined by the scheduler, often reversed.
//
// Clang generates:
//   mov eax, [esp+8]       ; load param2 FIRST
//   mov edx, [esp+4]       ; load param1
//   mov [ecx+0xbc], edx    ; store param1
//   mov edx, [ecx]         ; vtable
//   push 0x8f
//   push eax               ; push param2
//   call [edx+0x990]
//
// MSVC 6.0 generates:
//   mov eax, [esp+4]       ; load param1
//   mov [ecx+0xbc], eax    ; store param1
//   mov eax, [esp+8]       ; load param2
//   mov edx, [ecx]         ; vtable
//   push 0x8f
//   push eax               ; push param2
//   call [edx+0x990]
//
// The pass scans backward from CALL instructions, identifies stack
// parameter loads from [ESP+N], and reorders them into ascending offset
// order. Register assignments are adjusted to match the MSVC 6.0 pattern:
// the first (lowest offset) load uses EAX.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "MCTargetDesc/X86MCTargetDesc.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-sequential-param-load"

namespace {
class X86PreferSequentialParamLoadPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferSequentialParamLoadPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 prefer sequential stack param load";
  }

private:
  const X86InstrInfo *TII = nullptr;

  bool processBlock(MachineBasicBlock &MBB);
};
} // end anonymous namespace

char X86PreferSequentialParamLoadPass::ID = 0;

// ---------------------------------------------------------------------------
// Helper predicates
// ---------------------------------------------------------------------------

/// Check if MI is a 32-bit register load from [ESP + disp].
/// Returns the ESP displacement via OutDisp if true.
static bool isEspLoad(const MachineInstr &MI, int64_t &OutDisp) {
  if (MI.getOpcode() != X86::MOV32rm)
    return false;
  // Operand layout: dest, base, scale, index, disp, segment
  if (!MI.getOperand(1).isReg() ||
      MI.getOperand(1).getReg() != X86::ESP)
    return false;
  if (MI.getOperand(2).getImm() != 1)       // scale = 1
    return false;
  if (MI.getOperand(3).getReg() != X86::NoRegister)  // no index
    return false;
  OutDisp = MI.getOperand(4).getImm();
  return OutDisp > 0;  // stack params have positive offsets from ESP
}

/// Check if an instruction is an indirect call through memory.
static bool isCallMem(const MachineInstr &MI) {
  return MI.getOpcode() == X86::CALL32m ||
         MI.getOpcode() == X86::FARCALL32m;
}

/// Check if an instruction is a direct call.
static bool isCallDirect(const MachineInstr &MI) {
  return MI.getOpcode() == X86::CALLpcrel32;
}

/// Check if an instruction is any kind of call.
static bool isAnyCall(const MachineInstr &MI) {
  return isCallMem(MI) || isCallDirect(MI) || MI.isCall();
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

/// Check if an instruction reads the given register.
static bool readsReg(const MachineInstr &MI, Register Reg) {
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isReg() && MO.isUse() && MO.getReg() == Reg)
      return true;
  }
  return false;
}

/// Check if an instruction defines the given register.
static bool defsReg(const MachineInstr &MI, Register Reg) {
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isReg() && MO.isDef() && MO.getReg() == Reg)
      return true;
  }
  return false;
}

/// Get the 32-bit super register for a given register, or the register
/// itself if already 32-bit.
static Register getSuperReg32(Register Reg) {
  switch (Reg) {
  case X86::AL: case X86::AH: case X86::AX: case X86::EAX: return X86::EAX;
  case X86::DL: case X86::DH: case X86::DX: case X86::EDX: return X86::EDX;
  case X86::CL: case X86::CH: case X86::CX: case X86::ECX: return X86::ECX;
  case X86::BL: case X86::BH: case X86::BX: case X86::EBX: return X86::EBX;
  case X86::ESI: case X86::SI: return X86::ESI;
  case X86::EDI: case X86::DI: return X86::EDI;
  case X86::EBP: case X86::BP: return X86::EBP;
  case X86::ESP: case X86::SP: return X86::ESP;
  default: return Reg;
  }
}

/// Check if a register overlaps with another (same or sub/super).
static bool regOverlaps(Register A, Register B) {
  return getSuperReg32(A) == getSuperReg32(B);
}

/// Describes a stack parameter load and its consumer.
struct ParamLoadInfo {
  MachineInstr *LoadMI;      // The MOV32rm from [ESP+disp]
  int64_t EspDisp;           // The ESP displacement
  Register DestReg;          // Register the load writes to
  MachineInstr *ConsumerMI;  // The instruction that uses the loaded value
                             // (may be nullptr if consumed by push later)
};

// ---------------------------------------------------------------------------
// Main pass logic
// ---------------------------------------------------------------------------

bool X86PreferSequentialParamLoadPass::processBlock(MachineBasicBlock &MBB) {
  bool Changed = false;

  for (auto I = MBB.begin(), E = MBB.end(); I != E; ++I) {
    MachineInstr &CallMI = *I;

    if (!isAnyCall(CallMI))
      continue;

    // Walk backward from the CALL, collecting instructions up to a limit.
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
    // Prev[0] = instruction right before CALL, Prev[1] = next earlier, etc.

    if (Prev.size() < 2)
      continue;

    // Identify all ESP loads in this pre-call region.
    SmallVector<ParamLoadInfo, 4> ParamLoads;
    for (int i = 0; i < (int)Prev.size(); i++) {
      int64_t Disp = 0;
      if (isEspLoad(*Prev[i], Disp)) {
        ParamLoadInfo PLI;
        PLI.LoadMI = Prev[i];
        PLI.EspDisp = Disp;
        PLI.DestReg = Prev[i]->getOperand(0).getReg();
        PLI.ConsumerMI = nullptr;

        // Find the consumer: the next instruction (closer to CALL) that reads
        // the loaded register. It must be between the load and the call.
        for (int j = i - 1; j >= 0; j--) {
          if (readsReg(*Prev[j], PLI.DestReg)) {
            PLI.ConsumerMI = Prev[j];
            break;
          }
          // If something redefines the register before a read, stop.
          if (defsReg(*Prev[j], PLI.DestReg))
            break;
        }

        ParamLoads.push_back(PLI);
      }
    }

    // Need at least 2 ESP loads to have an ordering issue.
    if (ParamLoads.size() < 2)
      continue;

    // ParamLoads is in backward order (closest-to-call first). Check if
    // they are already in ascending ESP offset order when read in program
    // order (reversed). Program order = last element first.
    bool AlreadyAscending = true;
    for (int i = (int)ParamLoads.size() - 1; i > 0; i--) {
      // In program order, ParamLoads[i] comes before ParamLoads[i-1].
      if (ParamLoads[i].EspDisp >= ParamLoads[i - 1].EspDisp) {
        AlreadyAscending = false;
        break;
      }
    }
    if (AlreadyAscending)
      continue;

    // For the simple two-load case: reorder so the lower-offset load
    // comes first in program order, and adjust register usage.
    if (ParamLoads.size() != 2)
      continue;

    // ParamLoads[0] = closer to CALL (later in program order)
    // ParamLoads[1] = further from CALL (earlier in program order)
    ParamLoadInfo &Later = ParamLoads[0];   // closer to CALL
    ParamLoadInfo &Earlier = ParamLoads[1]; // further from CALL

    // We want ascending order: the load with the smaller ESP offset should
    // be earlier in program order. If Earlier already has the smaller offset,
    // no reorder is needed.
    if (Earlier.EspDisp < Later.EspDisp)
      continue;

    // Earlier has the larger offset, Later has the smaller offset.
    // We need to swap their positions so the smaller-offset load runs first.
    // This means: move Later.LoadMI + its consumer before Earlier.LoadMI.
    //
    // Current program order:
    //   Earlier.LoadMI  (high offset, e.g., [esp+8])
    //   ... (Earlier's consumer, possibly)
    //   Later.LoadMI    (low offset, e.g., [esp+4])
    //   ... (Later's consumer, possibly)
    //   PUSHes / other
    //   CALL
    //
    // Target MSVC 6.0 order:
    //   Later.LoadMI -> renamed to EAX   (low offset [esp+4])
    //   Later's consumer                 (uses EAX)
    //   Earlier.LoadMI -> reuses EAX     (high offset [esp+8])
    //   Earlier's consumer               (uses EAX)
    //   PUSHes / other
    //   CALL

    // Verify that the consumer of the earlier (high-offset) load comes
    // between the two loads, and the consumer of the later (low-offset)
    // load comes between the later load and the call.
    // Also check that rearranging won't break other dependencies.

    // Safety: verify that no instruction between the two loads reads or
    // writes ESP in a way that would change semantics when reordered.
    // (PUSHes modify ESP, but the loads should both be before any pushes.)
    bool HasPushBetween = false;
    {
      auto EarlierIt = MachineBasicBlock::iterator(*Earlier.LoadMI);
      auto LaterIt = MachineBasicBlock::iterator(*Later.LoadMI);
      for (auto It = std::next(EarlierIt);
           It != MBB.end() && It != LaterIt; ++It) {
        if (!It->isPseudo() && isAnyPush(*It)) {
          HasPushBetween = true;
          break;
        }
      }
    }
    if (HasPushBetween)
      continue;

    LLVM_DEBUG(dbgs() << "PreferSequentialParamLoad: reordering in "
                      << MBB.getParent()->getName() << "\n"
                      << "  Earlier (high offset): " << *Earlier.LoadMI
                      << "  Later (low offset): " << *Later.LoadMI);

    // Strategy: MSVC 6.0 uses EAX for every param load, reusing it
    // sequentially. The first load (low offset) loads into EAX, its
    // consumer uses EAX. Then the second load (high offset) loads into
    // EAX again, and its consumer also uses EAX.
    //
    // But if the second param is pushed (still alive at push time), we
    // can't reuse EAX for the second load. In that case, MSVC 6.0 still
    // loads both into EAX but the consumer of the first is a store (which
    // completes before the second load).
    //
    // Determine the target register mapping:
    // - If the low-offset param has a consumer that fully consumes the value
    //   (store), both loads can use EAX.
    // - Otherwise, use EAX for the first, keep the second's reg as-is.

    Register TargetReg = X86::EAX;

    // Check: does the low-offset (first) load's consumer fully consume it?
    // A store (MOV32mr, MOV16mr) fully consumes - the value is written to
    // memory and the register is free. A PUSH also consumes.
    bool FirstFullyConsumed = false;
    if (Later.ConsumerMI) {
      unsigned ConsOp = Later.ConsumerMI->getOpcode();
      if (ConsOp == X86::MOV32mr || ConsOp == X86::MOV16mr ||
          ConsOp == X86::MOV8mr || isAnyPush(*Later.ConsumerMI))
        FirstFullyConsumed = true;
    }

    // Determine which register each load should use after reorder.
    Register FirstLoadReg = TargetReg;  // low-offset load -> EAX
    Register SecondLoadReg;
    if (FirstFullyConsumed) {
      SecondLoadReg = TargetReg;  // can reuse EAX
    } else {
      // Keep the original register of the high-offset load.
      SecondLoadReg = Earlier.DestReg;
    }

    // Rewrite register assignments on the loads and their consumers.

    // First load (low offset, currently Later): change dest to FirstLoadReg.
    Register OldFirstReg = Later.DestReg;
    if (OldFirstReg != FirstLoadReg) {
      Later.LoadMI->getOperand(0).setReg(FirstLoadReg);
      // Update consumer's use of the old register.
      if (Later.ConsumerMI) {
        for (MachineOperand &MO : Later.ConsumerMI->operands()) {
          if (MO.isReg() && MO.isUse() && regOverlaps(MO.getReg(), OldFirstReg)) {
            // Replace with the corresponding sub/super of FirstLoadReg.
            if (MO.getReg() == OldFirstReg)
              MO.setReg(FirstLoadReg);
            else {
              // Handle sub-register cases (e.g., AX, AL for EAX).
              unsigned SubSize = 0;
              if (MO.getReg() == X86::AX || MO.getReg() == X86::DX ||
                  MO.getReg() == X86::CX)
                SubSize = 16;
              else if (MO.getReg() == X86::AL || MO.getReg() == X86::DL ||
                       MO.getReg() == X86::CL)
                SubSize = 8;
              if (SubSize > 0)
                MO.setReg(getX86SubSuperRegister(FirstLoadReg, SubSize));
            }
          }
        }
      }
    }

    // Second load (high offset, currently Earlier): change dest to SecondLoadReg.
    Register OldSecondReg = Earlier.DestReg;
    if (OldSecondReg != SecondLoadReg) {
      Earlier.LoadMI->getOperand(0).setReg(SecondLoadReg);
      // Update consumer's use of the old register.
      if (Earlier.ConsumerMI) {
        for (MachineOperand &MO : Earlier.ConsumerMI->operands()) {
          if (MO.isReg() && MO.isUse() && regOverlaps(MO.getReg(), OldSecondReg)) {
            if (MO.getReg() == OldSecondReg)
              MO.setReg(SecondLoadReg);
            else {
              unsigned SubSize = 0;
              if (MO.getReg() == X86::AX || MO.getReg() == X86::DX ||
                  MO.getReg() == X86::CX)
                SubSize = 16;
              else if (MO.getReg() == X86::AL || MO.getReg() == X86::DL ||
                       MO.getReg() == X86::CL)
                SubSize = 8;
              if (SubSize > 0)
                MO.setReg(getX86SubSuperRegister(SecondLoadReg, SubSize));
            }
          }
        }
      }
    }

    // Reorder: move the low-offset load (and its consumer) before the
    // high-offset load.
    //
    // Currently in program order:
    //   [Earlier.LoadMI]    high offset
    //   [Earlier.Consumer]  (optional)
    //   ...intervening...
    //   [Later.LoadMI]      low offset
    //   [Later.Consumer]    (optional)
    //
    // Target:
    //   [Later.LoadMI]      low offset (now first)
    //   [Later.Consumer]    (optional)
    //   [Earlier.LoadMI]    high offset (now second)
    //   [Earlier.Consumer]  (optional)

    auto EarlierLoadIt = MachineBasicBlock::iterator(*Earlier.LoadMI);

    // Move the low-offset load before the high-offset load.
    MBB.splice(EarlierLoadIt, &MBB,
               MachineBasicBlock::iterator(*Later.LoadMI),
               std::next(MachineBasicBlock::iterator(*Later.LoadMI)));

    // Move the low-offset load's consumer right after its load
    // (before the high-offset load).
    if (Later.ConsumerMI && Later.ConsumerMI != Earlier.LoadMI) {
      // Re-fetch iterator since splice invalidated it.
      EarlierLoadIt = MachineBasicBlock::iterator(*Earlier.LoadMI);
      MBB.splice(EarlierLoadIt, &MBB,
                 MachineBasicBlock::iterator(*Later.ConsumerMI),
                 std::next(MachineBasicBlock::iterator(*Later.ConsumerMI)));
    }

    // The high-offset load and its consumer are already in the right
    // relative position (after the low-offset load+consumer).

    Changed = true;
  }

  return Changed;
}

bool X86PreferSequentialParamLoadPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("prefer_sequential_param_load"))
    return false;

  TII = MF.getSubtarget<X86Subtarget>().getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF)
    Changed |= processBlock(MBB);

  return Changed;
}

FunctionPass *llvm::createX86PreferSequentialParamLoadPass() {
  return new X86PreferSequentialParamLoadPass();
}

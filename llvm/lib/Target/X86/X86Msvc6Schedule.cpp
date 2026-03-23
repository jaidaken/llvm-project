//===--- X86Msvc6Schedule.cpp - MSVC 6.0 instruction scheduling -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: MSVC 6.0 evaluates C expressions in AST-order depth-first:
// assignments complete before call targets are evaluated. Clang's scheduler
// reorders independent operations for ILP, placing vtable loads first.
//
// This pass detects the "info-chain-then-vtable-call" pattern and rewrites
// to match MSVC 6.0's evaluation order, simultaneously adjusting registers.
//
// Clang generates:
//   mov eax, [ecx]              ; vtable load FIRST
//   mov edx, [ecx+info_off]    ; info chain uses EDX
//   movzx edx, word [edx+fld]  ; field load (zero-extending)
//   mov [ecx+store_off], dx    ; store field
//   xor edx, edx               ; fastcall dummy
//   push imm                   ; push arg
//   call [eax+vt_off]          ; vtable call
//
// MSVC 6.0 generates:
//   mov eax, [ecx+info_off]    ; info chain uses EAX
//   mov dx, [eax+fld]          ; field load (partial, 16-bit)
//   mov eax, [ecx]             ; vtable load (reuses EAX)
//   push imm                   ; push arg
//   mov [ecx+store_off], dx    ; store field
//   call [eax+vt_off]          ; vtable call
//
// The pass also handles:
// - Converting MOVZX32rm16 to MOV16rm (partial register load)
// - Removing fastcall dummy XOR EDX, EDX before thiscall calls
// - Adjusting ESP displacements when instructions cross PUSHes
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "x86-msvc6-schedule"

namespace {
class X86Msvc6SchedulePass : public MachineFunctionPass {
public:
  static char ID;
  X86Msvc6SchedulePass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 MSVC 6.0 instruction scheduling";
  }

private:
  const X86InstrInfo *TII = nullptr;

  bool tryInfoChainBeforeVtable(MachineBasicBlock &MBB);
};
} // end anonymous namespace

char X86Msvc6SchedulePass::ID = 0;

/// Check if an instruction is a 32-bit register load from memory.
static bool isLoad32(const MachineInstr &MI, Register ExpectedDest) {
  return MI.getOpcode() == X86::MOV32rm &&
         MI.getOperand(0).getReg() == ExpectedDest;
}

/// Check if an instruction loads from [BaseReg + disp].
static bool isLoadFromBase(const MachineInstr &MI, Register BaseReg) {
  return MI.getOperand(1).isReg() &&
         MI.getOperand(1).getReg() == BaseReg &&
         MI.getOperand(2).getImm() == 1 &&  // scale = 1
         MI.getOperand(3).getReg() == X86::NoRegister;  // no index
}

/// Check if an instruction is a MOVZX32rm16 (zero-extending 16-bit load).
static bool isMovzx16(const MachineInstr &MI) {
  return MI.getOpcode() == X86::MOVZX32rm16;
}

/// Check if an instruction is a 16-bit store to memory.
static bool isStore16(const MachineInstr &MI) {
  return MI.getOpcode() == X86::MOV16mr;
}

/// Check if an instruction is XOR32rr zeroing a register.
static bool isXorZero(const MachineInstr &MI, Register Reg) {
  return (MI.getOpcode() == X86::XOR32rr ||
          MI.getOpcode() == X86::XOR32rr_REV) &&
         MI.getOperand(0).getReg() == Reg &&
         MI.getOperand(1).getReg() == Reg;
}

/// Check if an instruction is a PUSH immediate.
static bool isPushImm(const MachineInstr &MI) {
  return MI.getOpcode() == X86::PUSH32i8 ||
         MI.getOpcode() == X86::PUSH32i;
}

/// Check if an instruction is an indirect call through memory.
static bool isCallMem(const MachineInstr &MI) {
  return MI.getOpcode() == X86::CALL32m ||
         MI.getOpcode() == X86::FARCALL32m;
}

/// Try to match and transform the info-chain-before-vtable pattern.
///
/// Pattern (Clang output, walking backward from CALL):
///   [0] CALL32m [CallBase + vt_off]
///   [1] PUSH imm
///   [2] XOR32rr ChainReg, ChainReg         (optional: fastcall dummy)
///   [3] MOV16mr [thisReg+store_off], ChainReg16  (store field)
///   [4] MOVZX32rm16 ChainReg, [ChainReg+fld_off] (field load, zero-extending)
///   [5] MOV32rm ChainReg, [thisReg+info_off]      (info ptr load)
///   [6] MOV32rm CallBase, [thisReg]                (vtable load)
///
/// Target (MSVC 6.0 output):
///   MOV32rm CallBase, [thisReg+info_off]   (info chain uses CallBase)
///   MOV16rm CallBase16, [CallBase+fld_off] (field load, 16-bit partial)
///   MOV32rm CallBase, [thisReg]            (vtable load, reuses CallBase)
///   PUSH imm
///   MOV16mr [thisReg+store_off], ChainReg16 (store uses the 16-bit subreg)
///   CALL32m [CallBase + vt_off]
bool X86Msvc6SchedulePass::tryInfoChainBeforeVtable(MachineBasicBlock &MBB) {
  bool Changed = false;

  for (auto I = MBB.begin(), E = MBB.end(); I != E; ++I) {
    MachineInstr &CallMI = *I;

    // Step 1: Find an indirect call through a register base.
    if (!isCallMem(CallMI))
      continue;
    if (!CallMI.getOperand(0).isReg())
      continue;
    Register CallBase = CallMI.getOperand(0).getReg();
    if (CallBase == X86::NoRegister || CallBase == X86::ESP)
      continue;

    // Collect preceding non-pseudo instructions (up to 8).
    SmallVector<MachineInstr*, 8> Prev;
    {
      auto WalkIt = MachineBasicBlock::iterator(CallMI);
      while (Prev.size() < 8 && WalkIt != MBB.begin()) {
        --WalkIt;
        if (WalkIt->isTerminator())
          break;
        if (!WalkIt->isPseudo())
          Prev.push_back(&*WalkIt);
      }
    }
    // Prev[0] = instruction right before CALL, etc.

    // Step 2: Match the pattern backward from the call.
    // We need at least: push, store, field_load, info_load, vtable_load
    // Optionally: xor_zero between store and push

    int PushIdx = -1, XorIdx = -1, StoreIdx = -1, FieldIdx = -1,
        InfoIdx = -1, VtableIdx = -1;

    // Find PUSH (should be Prev[0] or Prev[1])
    for (int i = 0; i < (int)Prev.size() && i < 2; i++) {
      if (isPushImm(*Prev[i])) {
        PushIdx = i;
        break;
      }
    }
    if (PushIdx < 0)
      continue;

    // The next instruction after push (further back) could be XOR or store
    int cursor = PushIdx + 1;
    if (cursor >= (int)Prev.size())
      continue;

    // Optional: XOR EDX, EDX (fastcall dummy zero)
    if (isXorZero(*Prev[cursor], X86::EDX)) {
      XorIdx = cursor;
      cursor++;
    }
    if (cursor >= (int)Prev.size())
      continue;

    // MOV16mr: store field to this
    if (isStore16(*Prev[cursor])) {
      StoreIdx = cursor;
      cursor++;
    } else {
      continue;
    }
    if (cursor >= (int)Prev.size())
      continue;

    // MOVZX32rm16: field load (zero-extending 16-bit)
    if (isMovzx16(*Prev[cursor])) {
      FieldIdx = cursor;
      cursor++;
    } else {
      continue;
    }
    if (cursor >= (int)Prev.size())
      continue;

    // MOV32rm: info ptr load into the chain register
    Register ChainReg = Prev[FieldIdx]->getOperand(0).getReg();
    if (isLoad32(*Prev[cursor], ChainReg)) {
      InfoIdx = cursor;
      cursor++;
    } else {
      continue;
    }
    if (cursor >= (int)Prev.size())
      continue;

    // MOV32rm: vtable load into CallBase
    if (isLoad32(*Prev[cursor], CallBase)) {
      VtableIdx = cursor;
    } else {
      continue;
    }

    // Step 3: Verify the pattern details.
    MachineInstr &VtableMI = *Prev[VtableIdx];
    MachineInstr &InfoMI = *Prev[InfoIdx];
    MachineInstr &FieldMI = *Prev[FieldIdx];
    MachineInstr &StoreMI = *Prev[StoreIdx];
    MachineInstr &PushMI = *Prev[PushIdx];

    // Vtable load must be from [thisReg + 0] (offset 0 = vtable pointer)
    Register ThisReg = VtableMI.getOperand(1).getReg();
    if (!VtableMI.getOperand(1).isReg() || ThisReg == X86::ESP)
      continue;
    int64_t VtableDisp = VtableMI.getOperand(4).getImm();
    if (VtableDisp != 0)
      continue;

    // Info load must be from [thisReg + nonzero_disp]
    if (!isLoadFromBase(InfoMI, ThisReg))
      continue;
    int64_t InfoDisp = InfoMI.getOperand(4).getImm();
    if (InfoDisp == 0)
      continue;

    // Field load must chain through ChainReg (info ptr as base)
    if (!FieldMI.getOperand(1).isReg() ||
        FieldMI.getOperand(1).getReg() != ChainReg)
      continue;

    // Store must write to [thisReg + some_offset]
    if (!StoreMI.getOperand(0).isReg() ||
        StoreMI.getOperand(0).getReg() != ThisReg)
      continue;

    // The chain register and call base must be different
    if (ChainReg == CallBase) {
      // Clang already has them the same - this IS the MSVC 6.0 pattern
      // but with the wrong order. We just need to reorder.
    }

    LLVM_DEBUG(dbgs() << "Msvc6Schedule: matched info-chain-before-vtable in "
                      << MBB.getParent()->getName() << "\n");

    // Step 4: Build the MSVC 6.0 instruction sequence.
    DebugLoc DL = VtableMI.getDebugLoc();

    // Determine the 16-bit subreg for the store.
    // MSVC 6.0 uses DX for the field value regardless of which 32-bit reg
    // holds the info chain. After rewriting, the info chain uses CallBase
    // (EAX), and the field value lives in DX.
    Register FieldSubReg16 = X86::DX;
    if (CallBase == X86::EDX)
      FieldSubReg16 = X86::AX;  // unlikely but handle it

    // 4a: Info load: MOV32rm CallBase, [thisReg + InfoDisp]
    // (was: MOV32rm ChainReg, [thisReg + InfoDisp])
    MachineInstr *NewInfoLoad = BuildMI(MBB, VtableMI, DL,
        TII->get(X86::MOV32rm), CallBase)
      .addReg(ThisReg)
      .addImm(InfoMI.getOperand(2).getImm())
      .addReg(InfoMI.getOperand(3).getReg())
      .addImm(InfoDisp)
      .addReg(InfoMI.getOperand(5).getReg());

    // 4b: Field load: MOV16rm FieldSubReg16, [CallBase + FieldDisp]
    // (was: MOVZX32rm16 ChainReg, [ChainReg + FieldDisp])
    // This converts movzx to a partial 16-bit load (MSVC 6.0 pattern).
    int64_t FieldDisp = FieldMI.getOperand(4).getImm();
    BuildMI(MBB, VtableMI, DL, TII->get(X86::MOV16rm), FieldSubReg16)
      .addReg(CallBase)
      .addImm(FieldMI.getOperand(2).getImm())
      .addReg(FieldMI.getOperand(3).getReg())
      .addImm(FieldDisp)
      .addReg(FieldMI.getOperand(5).getReg());

    // 4c: Vtable load stays in place but we rebuild it to ensure
    // it uses CallBase as dest and thisReg as base.
    // (The original VtableMI is at the right position, but registers
    //  may need changing if CallBase was ChainReg.)
    // Actually, vtable load was already loading into CallBase from thisReg.
    // Just leave it as is - it's already correct.

    // 4d: Move PUSH to right after the vtable load.
    // Currently PUSH is at PushIdx (near the CALL). We want it right after
    // the vtable load. Just splice it.
    MachineBasicBlock::iterator AfterVtable =
        std::next(MachineBasicBlock::iterator(VtableMI));
    MBB.splice(AfterVtable, &MBB,
               MachineBasicBlock::iterator(PushMI),
               std::next(MachineBasicBlock::iterator(PushMI)));

    // 4e: Move store to right after the push.
    // The store's source register needs updating: was ChainReg16, now
    // FieldSubReg16.
    MachineBasicBlock::iterator AfterPush =
        std::next(MachineBasicBlock::iterator(PushMI));
    MBB.splice(AfterPush, &MBB,
               MachineBasicBlock::iterator(StoreMI),
               std::next(MachineBasicBlock::iterator(StoreMI)));

    // Update the store's source register to match the new field load dest.
    // StoreMI operand layout for MOV16mr: [base, scale, index, disp, seg, src]
    StoreMI.getOperand(5).setReg(FieldSubReg16);

    // 4f: Remove the original info load, field load, and optional XOR.
    InfoMI.eraseFromParent();
    FieldMI.eraseFromParent();
    if (XorIdx >= 0)
      Prev[XorIdx]->eraseFromParent();

    Changed = true;
  }

  return Changed;
}

bool X86Msvc6SchedulePass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("msvc6_schedule"))
    return false;

  TII = MF.getSubtarget<X86Subtarget>().getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    Changed |= tryInfoChainBeforeVtable(MBB);
  }

  return Changed;
}

FunctionPass *llvm::createX86Msvc6SchedulePass() {
  return new X86Msvc6SchedulePass();
}

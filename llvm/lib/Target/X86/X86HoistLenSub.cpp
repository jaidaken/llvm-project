//===--- X86HoistLenSub.cpp - Hoist len-=k and eliminate stack spills -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: Transforms the adler32 outer loop to match MSVC 6.0's structure.
// MSVC computes len -= k BEFORE the DO16 loop, avoiding stack spills.
//
// This pass:
// 1. Finds "mov [esp+N], ebx" (save len) and "mov [esp+M], eax" (save k)
// 2. Finds the corresponding restores and "sub ebx, eax" (len -= k)
// 3. Inserts "sub ebx, eax" before the trip count setup
// 4. Rewrites the trip count setup to use EAX for remainder (not EBX)
// 5. Deletes stack saves/restores
// 6. Removes "sub esp, 8" / "add esp, 8" frame adjustments
// 7. Adjusts all ESP-relative memory operands (parameter loads get -8)
//
// Gated behind the prefer_div attribute.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "MCTargetDesc/X86BaseInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-hoist-len-sub"
#define X86_HOIST_LEN_SUB_NAME "X86 hoist len subtraction pass"

namespace {
class X86HoistLenSubPass : public MachineFunctionPass {
public:
  static char ID;
  X86HoistLenSubPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return X86_HOIST_LEN_SUB_NAME; }
};
} // end anonymous namespace

char X86HoistLenSubPass::ID = 0;

bool X86HoistLenSubPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::PreferDiv))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  // Step 1: Find and delete "sub esp, 8" in the entry block
  MachineBasicBlock &Entry = MF.front();
  MachineInstr *SubEsp = nullptr;
  for (MachineInstr &MI : Entry) {
    if ((MI.getOpcode() == X86::SUB32ri8 || MI.getOpcode() == X86::SUB32ri) &&
        MI.getOperand(0).getReg() == X86::ESP &&
        MI.getOperand(2).getImm() == 8) {
      SubEsp = &MI;
      break;
    }
  }

  if (!SubEsp)
    return false; // No stack frame to eliminate

  // Step 2: Find and delete "add esp, 8" in all blocks (epilogue)
  SmallVector<MachineInstr *, 4> AddEsps;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if ((MI.getOpcode() == X86::ADD32ri8 || MI.getOpcode() == X86::ADD32ri) &&
          MI.getOperand(0).getReg() == X86::ESP &&
          MI.getOperand(2).getImm() == 8) {
        AddEsps.push_back(&MI);
      }
    }
  }

  if (AddEsps.empty())
    return false;

  // Step 3: Find stack saves and restores
  // Save: MOV32mr [ESP+disp], EBX (len save)
  // Save: MOV32mr [ESP+disp], EAX (k save)
  // Restore: MOV32rm EAX, [ESP+disp] (k restore)
  // Restore: MOV32rm EBX, [ESP+disp] (len restore)
  SmallVector<MachineInstr *, 8> StackSaves;
  SmallVector<MachineInstr *, 8> StackRestores;
  MachineInstr *LenSubInstr = nullptr;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      // Stack save: MOV32mr with ESP base and small displacement
      if (MI.getOpcode() == X86::MOV32mr) {
        int MemOpIdx = X86II::getMemoryOperandNo(MI.getDesc().TSFlags);
        if (MemOpIdx >= 0) {
          MemOpIdx += X86II::getOperandBias(MI.getDesc());
          unsigned BaseIdx = MemOpIdx + X86::AddrBaseReg;
          unsigned DispIdx = MemOpIdx + X86::AddrDisp;
          if (BaseIdx < MI.getNumOperands() &&
              MI.getOperand(BaseIdx).isReg() &&
              MI.getOperand(BaseIdx).getReg() == X86::ESP &&
              DispIdx < MI.getNumOperands() &&
              MI.getOperand(DispIdx).isImm()) {
            int64_t Disp = MI.getOperand(DispIdx).getImm();
            if (Disp >= 0 && Disp < 8) {
              StackSaves.push_back(&MI);
            }
          }
        }
      }
      // Stack restore: MOV32rm from ESP base with small displacement
      if (MI.getOpcode() == X86::MOV32rm) {
        int MemOpIdx = X86II::getMemoryOperandNo(MI.getDesc().TSFlags);
        if (MemOpIdx >= 0) {
          MemOpIdx += X86II::getOperandBias(MI.getDesc());
          unsigned BaseIdx = MemOpIdx + X86::AddrBaseReg;
          unsigned DispIdx = MemOpIdx + X86::AddrDisp;
          if (BaseIdx < MI.getNumOperands() &&
              MI.getOperand(BaseIdx).isReg() &&
              MI.getOperand(BaseIdx).getReg() == X86::ESP &&
              DispIdx < MI.getNumOperands() &&
              MI.getOperand(DispIdx).isImm()) {
            int64_t Disp = MI.getOperand(DispIdx).getImm();
            if (Disp >= 0 && Disp < 8) {
              StackRestores.push_back(&MI);
            }
          }
        }
      }
      // SUB32rr EBX, EAX (len -= k)
      if (MI.getOpcode() == X86::SUB32rr &&
          MI.getOperand(0).getReg() == X86::EBX &&
          MI.getOperand(2).getReg() == X86::EAX) {
        LenSubInstr = &MI;
      }
    }
  }

  if (StackSaves.size() < 1 || StackRestores.size() < 1 || !LenSubInstr)
    return false;

  // Step 4: Find the "mov ebx, 0x4(%esp)" save - this is the len save point.
  // Insert "sub ebx, eax" right after it, then change the save to
  // "sub ebx, eax" (replacing the save).
  MachineInstr *LenSave = nullptr;
  for (MachineInstr *Save : StackSaves) {
    // Check if the source register is EBX
    // MOV32mr format: [base, scale, index, disp, seg], src
    unsigned SrcIdx = Save->getNumOperands() - 1;
    if (Save->getOperand(SrcIdx).isReg() &&
        Save->getOperand(SrcIdx).getReg() == X86::EBX) {
      LenSave = Save;
      break;
    }
  }

  if (!LenSave)
    return false;

  // Step 5: Insert "sub ebx, eax" where the len save was
  DebugLoc DL = LenSave->getDebugLoc();
  MachineBasicBlock *SaveBB = LenSave->getParent();
  BuildMI(*SaveBB, *LenSave, DL, TII->get(X86::SUB32rr), X86::EBX)
      .addReg(X86::EBX)
      .addReg(X86::EAX);

  // Step 5b: Change "cmp ebx, 0x10" to "cmp eax, 0x10" since EBX now
  // holds len-k (not k). EAX still has k.
  for (auto I = MachineBasicBlock::iterator(LenSave); I != SaveBB->end(); ++I) {
    if ((I->getOpcode() == X86::CMP32ri8 || I->getOpcode() == X86::CMP32ri) &&
        I->getOperand(0).getReg() == X86::EBX &&
        I->getOperand(1).getImm() == 16) {
      I->getOperand(0).setReg(X86::EAX);
      break;
    }
  }

  // Step 6: Change trip count setup to use EAX instead of EBX.
  // Find "mov ebx, eax" right after the "jb skip_do16" guard and change
  // all uses of EBX in the trip count computation to EAX.
  // Pattern: mov ebx, eax; mov ebp, ebx; shr ebp, 4; ...; add ebx, edx
  // Change to: (remove mov ebx,eax); mov ebp, eax; shr ebp, 4; ...; add eax, edx
  //
  // Scan ALL blocks to find "mov ebx, eax" that starts the trip count setup.
  // It may be in a different block than the len save.
  MachineInstr *MovEbxEax = nullptr;
  MachineBasicBlock *TripBB = nullptr;
  for (MachineBasicBlock &MBB2 : MF) {
    for (MachineInstr &MI : MBB2) {
      if (MI.getOpcode() == X86::MOV32rr &&
          MI.getOperand(0).getReg() == X86::EBX &&
          MI.getOperand(1).getReg() == X86::EAX) {
        // Check if this is followed by trip count pattern (mov ebp, ebx; shr)
        auto Next = std::next(MachineBasicBlock::iterator(&MI));
        if (Next != MBB2.end() &&
            Next->getOpcode() == X86::MOV32rr &&
            Next->getOperand(0).getReg() == X86::EBP) {
          MovEbxEax = &MI;
          TripBB = &MBB2;
          break;
        }
      }
    }
    if (MovEbxEax) break;
  }

  // If found, rewrite the trip count setup to use EAX
  if (MovEbxEax && TripBB) {
    // Find all instructions between MovEbxEax and the loop body that use EBX
    // and change them to use EAX
    bool inTripCount = false;
    for (auto I = MachineBasicBlock::iterator(MovEbxEax);
         I != TripBB->end(); ++I) {
      if (&*I == MovEbxEax) {
        inTripCount = true;
        continue;
      }
      if (!inTripCount)
        continue;

      // Stop at the loop body (NOP padding or the first XOR)
      if (I->getOpcode() == X86::NOOP || I->getOpcode() == X86::XOR32rr)
        break;

      // Replace EBX with EAX in operands
      for (MachineOperand &MO : I->operands()) {
        if (MO.isReg() && MO.getReg() == X86::EBX)
          MO.setReg(X86::EAX);
        if (MO.isReg() && MO.getReg() == X86::BL)
          MO.setReg(X86::AL);
      }
    }

    // Delete the "mov ebx, eax" since we're using EAX directly now
    MovEbxEax->eraseFromParent();
  }

  // Step 7: Also fix the tail loop entry and post-loop code that reads EBX
  // After the DO16 loop, "test ebx, ebx" tests the remainder.
  // With the change, remainder is now in EAX, so change to "test eax, eax".
  // And "mov ebx, eax" at 0x150 (skip_do16 path) should become just
  // keeping EAX as-is (since EAX already has k, which is the remainder
  // when k < 16).

  // Fix: In the block that has "test ebx, ebx; jne tail", change to test eax.
  // And in the tail loop setup, change EBX references to EAX.
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      // "test ebx, ebx" after DO16 loop -> "test eax, eax"
      if (MI.getOpcode() == X86::TEST32rr &&
          MI.getOperand(0).getReg() == X86::EBX &&
          MI.getOperand(1).getReg() == X86::EBX) {
        // Check if this is after the loop (near a DEC32r EBP)
        // Simple heuristic: if there's a DEC EBP nearby before this TEST
        auto Prev = MachineBasicBlock::iterator(&MI);
        bool nearDec = false;
        for (int i = 0; i < 5 && Prev != MBB.begin(); i++) {
          --Prev;
          if (Prev->getOpcode() == X86::DEC32r &&
              Prev->getOperand(0).getReg() == X86::EBP) {
            nearDec = true;
            break;
          }
        }
        if (nearDec) {
          MI.getOperand(0).setReg(X86::EAX);
          MI.getOperand(1).setReg(X86::EAX);
        }
      }
    }
  }

  // Step 8: Delete all stack saves, restores, the original sub, and frame adj
  for (MachineInstr *MI : StackSaves)
    MI->eraseFromParent();
  for (MachineInstr *MI : StackRestores)
    MI->eraseFromParent();
  LenSubInstr->eraseFromParent();
  SubEsp->eraseFromParent();
  for (MachineInstr *MI : AddEsps)
    MI->eraseFromParent();

  // Step 9: Adjust all remaining ESP-relative memory operands.
  // Removing "sub esp, 8" shifts ESP up by 8 bytes. All parameter loads
  // that were [esp+0x20] become [esp+0x18], etc.
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      int MemOpIdx = X86II::getMemoryOperandNo(MI.getDesc().TSFlags);
      if (MemOpIdx < 0)
        continue;
      MemOpIdx += X86II::getOperandBias(MI.getDesc());
      unsigned BaseIdx = MemOpIdx + X86::AddrBaseReg;
      unsigned DispIdx = MemOpIdx + X86::AddrDisp;
      if (BaseIdx >= MI.getNumOperands() || DispIdx >= MI.getNumOperands())
        continue;
      MachineOperand &BaseMO = MI.getOperand(BaseIdx);
      MachineOperand &DispMO = MI.getOperand(DispIdx);
      if (BaseMO.isReg() && BaseMO.getReg() == X86::ESP && DispMO.isImm()) {
        DispMO.setImm(DispMO.getImm() - 8);
      }
    }
  }

  Changed = true;
  return Changed;
}

FunctionPass *llvm::createX86HoistLenSubPass() {
  return new X86HoistLenSubPass();
}

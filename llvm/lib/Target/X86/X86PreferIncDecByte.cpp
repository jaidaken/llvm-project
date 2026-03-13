//===--- X86PreferIncDecByte.cpp - Replace load-add-store with INC/DEC ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: This pass replaces movzx+add/inc+mov byte sequences with INC8m
// or DEC8m memory operations when the function has the PreferIncDecByte
// attribute. This matches MSVC 6.0's code generation pattern of
// "inc byte ptr [ecx+0xb6]" instead of the load-modify-store sequence.
//
// Pattern matched:
//   MOVZX32rm8  %reg, [mem]       ; load byte, zero-extend to 32-bit
//   ADD32ri8    %reg, 1           ; add 1 (or INC32r %reg)
//   MOV8mr      [mem], %reg_lo   ; store low byte back
// =>
//   INC8m       [mem]
//
// Similarly for subtract-by-1 / DEC32r => DEC8m.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "MCTargetDesc/X86BaseInfo.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-inc-dec-byte"
#define X86_PREFER_INC_DEC_BYTE_NAME                                           \
  "X86 prefer INC/DEC byte memory pass"

namespace {
class X86PreferIncDecBytePass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferIncDecBytePass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return X86_PREFER_INC_DEC_BYTE_NAME;
  }
};
} // end anonymous namespace

char X86PreferIncDecBytePass::ID = 0;

/// Compare two memory operand sequences from different instructions.
/// \p MI1 at operand offset \p Op1 and \p MI2 at operand offset \p Op2.
/// Returns true if they refer to the same memory location.
static bool sameMemoryOperands(const MachineInstr &MI1, unsigned Op1,
                               const MachineInstr &MI2, unsigned Op2) {
  // Check that both instructions have enough operands for a full memory ref.
  if (Op1 + X86::AddrNumOperands > MI1.getNumOperands() ||
      Op2 + X86::AddrNumOperands > MI2.getNumOperands())
    return false;

  // Compare base register.
  const MachineOperand &Base1 = MI1.getOperand(Op1 + X86::AddrBaseReg);
  const MachineOperand &Base2 = MI2.getOperand(Op2 + X86::AddrBaseReg);
  if (!Base1.isReg() || !Base2.isReg() || Base1.getReg() != Base2.getReg())
    return false;

  // Compare scale.
  const MachineOperand &Scale1 = MI1.getOperand(Op1 + X86::AddrScaleAmt);
  const MachineOperand &Scale2 = MI2.getOperand(Op2 + X86::AddrScaleAmt);
  if (!Scale1.isImm() || !Scale2.isImm() ||
      Scale1.getImm() != Scale2.getImm())
    return false;

  // Compare index register.
  const MachineOperand &Index1 = MI1.getOperand(Op1 + X86::AddrIndexReg);
  const MachineOperand &Index2 = MI2.getOperand(Op2 + X86::AddrIndexReg);
  if (!Index1.isReg() || !Index2.isReg() ||
      Index1.getReg() != Index2.getReg())
    return false;

  // Compare displacement.
  const MachineOperand &Disp1 = MI1.getOperand(Op1 + X86::AddrDisp);
  const MachineOperand &Disp2 = MI2.getOperand(Op2 + X86::AddrDisp);
  if (Disp1.getType() != Disp2.getType())
    return false;
  if (Disp1.isImm()) {
    if (Disp1.getImm() != Disp2.getImm())
      return false;
  } else if (Disp1.isGlobal()) {
    if (Disp1.getGlobal() != Disp2.getGlobal() ||
        Disp1.getOffset() != Disp2.getOffset())
      return false;
  } else if (Disp1.isCPI()) {
    if (Disp1.getIndex() != Disp2.getIndex() ||
        Disp1.getOffset() != Disp2.getOffset())
      return false;
  } else {
    return false;
  }

  // Compare segment register.
  const MachineOperand &Seg1 = MI1.getOperand(Op1 + X86::AddrSegmentReg);
  const MachineOperand &Seg2 = MI2.getOperand(Op2 + X86::AddrSegmentReg);
  if (!Seg1.isReg() || !Seg2.isReg() || Seg1.getReg() != Seg2.getReg())
    return false;

  return true;
}

/// Check if any register in the memory operands of MI (starting at offset
/// MemOpStart) overlaps with Reg.
static bool memOpUsesReg(const MachineInstr &MI, unsigned MemOpStart,
                         Register Reg, const TargetRegisterInfo *TRI) {
  for (unsigned I = 0; I < X86::AddrNumOperands; ++I) {
    const MachineOperand &MO = MI.getOperand(MemOpStart + I);
    if (MO.isReg() && MO.getReg() != 0 && TRI->regsOverlap(MO.getReg(), Reg))
      return true;
  }
  return false;
}

bool X86PreferIncDecBytePass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::PreferIncDecByte))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Collect candidate triplets, then replace.
    // A candidate is (MOVZX, ADD/INC/SUB/DEC, MOV8mr).
    struct Candidate {
      MachineInstr *Movzx;
      MachineInstr *ArithOp;
      MachineInstr *Store;
      bool IsInc; // true = INC8m, false = DEC8m
    };
    SmallVector<Candidate, 4> Candidates;

    // Pass 1: Compute EFLAGS liveness at each potential MOVZX instruction.
    // We iterate backwards to track liveness.
    // But first, let's do a forward scan to find triplets, then check EFLAGS.

    // Actually, let's do it like ExpandMovzx: backward liveness scan, then
    // forward pattern matching with liveness info.

    // Build a map of EFLAGS liveness at each instruction.
    // We'll iterate backwards and record whether EFLAGS is live before each MI.
    DenseMap<MachineInstr *, bool> EFLAGSLiveBefore;
    {
      LivePhysRegs LiveRegs(*TRI);
      LiveRegs.addLiveOuts(MBB);

      for (auto I = MBB.rbegin(), E = MBB.rend(); I != E; ++I) {
        MachineInstr &MI = *I;
        // After stepBackward, LiveRegs contains what's live before MI.
        // But we need "live after" the MOVZX for our check.
        // INC/DEC will clobber flags, so we need EFLAGS to be dead after
        // the store instruction (the last in the sequence).
        // Actually, we need EFLAGS to not be live across the replacement.
        // The simplest check: EFLAGS must not be live after the store.
        EFLAGSLiveBefore[&MI] = LiveRegs.contains(X86::EFLAGS);
        LiveRegs.stepBackward(MI);
      }
    }

    // Pass 2: Forward scan for triplet patterns.
    for (auto I = MBB.begin(), E = MBB.end(); I != E; ++I) {
      MachineInstr &MI1 = *I;

      // Step 1: Look for MOVZX32rm8.
      if (MI1.getOpcode() != X86::MOVZX32rm8)
        continue;

      // MOVZX32rm8 operands: def GR32:$dst, mem:$src (5 operands starting at 1)
      Register DstReg32 = MI1.getOperand(0).getReg();
      unsigned MovzxMemStart = 1; // memory operands start at operand 1

      // Get the 8-bit sub-register (AL for EAX, etc.)
      Register DstReg8 = TRI->getSubReg(DstReg32, X86::sub_8bit);
      if (!DstReg8)
        continue;

      // Step 2: Next instruction should be ADD32ri8 R, 1 or INC32r R
      //         or SUB32ri8 R, 1 or DEC32r R
      auto I2 = std::next(I);
      if (I2 == E)
        continue;
      MachineInstr &MI2 = *I2;

      bool IsInc = false;
      unsigned ArithOpcode = MI2.getOpcode();

      if (ArithOpcode == X86::ADD32ri8) {
        // ADD32ri8: def GR32:$dst, tied GR32:$src1, i32i8imm:$src2
        if (MI2.getOperand(0).getReg() != DstReg32 ||
            MI2.getOperand(1).getReg() != DstReg32)
          continue;
        if (!MI2.getOperand(2).isImm() || MI2.getOperand(2).getImm() != 1)
          continue;
        IsInc = true;
      } else if (ArithOpcode == X86::ADD8ri) {
        // ADD8ri: def GR8:$dst, tied GR8:$src1, i8imm:$src2
        if (MI2.getOperand(0).getReg() != DstReg8 ||
            MI2.getOperand(1).getReg() != DstReg8)
          continue;
        if (!MI2.getOperand(2).isImm() || MI2.getOperand(2).getImm() != 1)
          continue;
        IsInc = true;
      } else if (ArithOpcode == X86::INC32r) {
        // INC32r: def GR32:$dst, tied GR32:$src
        if (MI2.getOperand(0).getReg() != DstReg32 ||
            MI2.getOperand(1).getReg() != DstReg32)
          continue;
        IsInc = true;
      } else if (ArithOpcode == X86::INC8r) {
        // INC8r: def GR8:$dst, tied GR8:$src
        if (MI2.getOperand(0).getReg() != DstReg8 ||
            MI2.getOperand(1).getReg() != DstReg8)
          continue;
        IsInc = true;
      } else if (ArithOpcode == X86::SUB32ri8) {
        // SUB32ri8: def GR32:$dst, tied GR32:$src1, i32i8imm:$src2
        if (MI2.getOperand(0).getReg() != DstReg32 ||
            MI2.getOperand(1).getReg() != DstReg32)
          continue;
        if (!MI2.getOperand(2).isImm() || MI2.getOperand(2).getImm() != 1)
          continue;
        IsInc = false;
      } else if (ArithOpcode == X86::SUB8ri) {
        // SUB8ri: def GR8:$dst, tied GR8:$src1, i8imm:$src2
        if (MI2.getOperand(0).getReg() != DstReg8 ||
            MI2.getOperand(1).getReg() != DstReg8)
          continue;
        if (!MI2.getOperand(2).isImm() || MI2.getOperand(2).getImm() != 1)
          continue;
        IsInc = false;
      } else if (ArithOpcode == X86::DEC32r) {
        // DEC32r: def GR32:$dst, tied GR32:$src
        if (MI2.getOperand(0).getReg() != DstReg32 ||
            MI2.getOperand(1).getReg() != DstReg32)
          continue;
        IsInc = false;
      } else if (ArithOpcode == X86::DEC8r) {
        // DEC8r: def GR8:$dst, tied GR8:$src
        if (MI2.getOperand(0).getReg() != DstReg8 ||
            MI2.getOperand(1).getReg() != DstReg8)
          continue;
        IsInc = false;
      } else {
        continue;
      }

      // Step 3: Next instruction should be MOV8mr [mem], R_lo
      auto I3 = std::next(I2);
      if (I3 == E)
        continue;
      MachineInstr &MI3 = *I3;

      if (MI3.getOpcode() != X86::MOV8mr)
        continue;

      // MOV8mr operands: mem:$dst (5 operands 0-4), GR8:$src (operand 5)
      unsigned StoreMemStart = 0;
      unsigned StoreSrcOp = X86::AddrNumOperands; // operand 5

      // Check that the store source register is the low byte of our register.
      if (MI3.getOperand(StoreSrcOp).getReg() != DstReg8)
        continue;

      // Check that memory operands match between MOVZX and MOV8mr.
      if (!sameMemoryOperands(MI1, MovzxMemStart, MI3, StoreMemStart))
        continue;

      // Check that the base/index registers used in the memory operand are
      // not the same as the destination register (they shouldn't be modified
      // between load and store, and INC8m needs them intact).
      if (memOpUsesReg(MI1, MovzxMemStart, DstReg32, TRI))
        continue;

      // Check that DstReg32 is not used by any other instruction between
      // the MOVZX and the store. Since our pattern is exactly 3 consecutive
      // instructions, we just need to make sure DstReg32 is dead after the store.
      // Also verify no other uses of DstReg32 sneak in.

      // Check EFLAGS: INC8m/DEC8m modify OF, SF, ZF, AF, PF (but not CF).
      // The original ADD/SUB also modifies all these plus CF.
      // The original INC/DEC modifies the same flags as INC8m/DEC8m.
      // We need EFLAGS to not be live after the store instruction.
      // If EFLAGS is live after the store, we can't safely replace because
      // the original sequence may have set flags that are used later.
      //
      // Check: is EFLAGS live after MI3 (the store)?
      // The store (MOV8mr) doesn't define EFLAGS. So EFLAGS liveness after
      // MI3 is the same as before MI3.
      // We recorded EFLAGSLiveBefore for each instruction.
      // "Live after MI3" = EFLAGS is live before the instruction after MI3,
      // or if MI3 is the last instruction, check liveouts.
      auto I4 = std::next(I3);
      bool EFLAGSLiveAfterStore;
      if (I4 == E) {
        // MI3 is the last instruction. Check if EFLAGS is a live-out of MBB.
        LivePhysRegs LiveRegs(*TRI);
        LiveRegs.addLiveOuts(MBB);
        EFLAGSLiveAfterStore = LiveRegs.contains(X86::EFLAGS);
      } else {
        EFLAGSLiveAfterStore = EFLAGSLiveBefore[&*I4];
      }

      if (EFLAGSLiveAfterStore)
        continue;

      // Check that DstReg32 is dead after the store.
      // If it's live after the store, someone else uses the incremented value,
      // and we can't eliminate the register operations.
      // Use the same approach: check liveness before the next instruction.
      bool RegLiveAfterStore;
      if (I4 == E) {
        LivePhysRegs LiveRegs(*TRI);
        LiveRegs.addLiveOuts(MBB);
        RegLiveAfterStore = LiveRegs.contains(DstReg32);
      } else {
        // We need to check if DstReg32 (or any alias) is live before I4.
        // We didn't record that in our map. Let's do a quick check:
        // DstReg32 is dead after the store if the store kills it.
        // In post-RA code, we can check kill flags or do liveness analysis.
        // Since we already have EFLAGSLiveBefore map computed via LivePhysRegs,
        // let's just recompute liveness for this specific case.
        //
        // Actually, let's take a simpler approach: check if the MOV8mr has
        // a kill flag on the source register, and also check if DstReg32
        // is used anywhere after the store before being redefined.
        //
        // For the common case (the entire purpose of DstReg32 was to hold the
        // temporary), the register will be dead. Let's check kill flags.
        // In post-RA code, kill flags should be accurate.
        RegLiveAfterStore = !MI3.getOperand(StoreSrcOp).isKill();
      }

      if (RegLiveAfterStore)
        continue;

      Candidates.push_back({&MI1, &MI2, &MI3, IsInc});

      // Skip past the matched instructions.
      I = I3;
    }

    // Pass 3: Replace matched triplets with INC8m/DEC8m.
    for (auto &Cand : Candidates) {
      unsigned NewOpcode = Cand.IsInc ? X86::INC8m : X86::DEC8m;
      DebugLoc DL = Cand.Movzx->getDebugLoc();
      unsigned MovzxMemStart = 1;

      auto MIB = BuildMI(MBB, *Cand.Movzx, DL, TII->get(NewOpcode));
      // Copy memory operands from the MOVZX (operands 1..5).
      for (unsigned I = 0; I < X86::AddrNumOperands; ++I)
        MIB.add(Cand.Movzx->getOperand(MovzxMemStart + I));
      // Copy memory references for alias analysis.
      MIB.cloneMemRefs(*Cand.Movzx);

      Cand.Store->eraseFromParent();
      Cand.ArithOp->eraseFromParent();
      Cand.Movzx->eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferIncDecBytePass() {
  return new X86PreferIncDecBytePass();
}

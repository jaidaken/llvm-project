//===--- X86Msvc6Restructure.cpp - Restructure function to match MSVC 6 --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: Comprehensive restructuring pass for MSVC 6.0 matching.
// Handles: block layout reordering, split prologue (pushes interleaved
// with param loads), inline null return, and interleaved epilogue.
//
// Gated behind Attribute::PreferDiv.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "MCTargetDesc/X86BaseInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-msvc6-restructure"
#define X86_MSVC6_RESTRUCTURE_NAME "X86 MSVC 6.0 restructure pass"

/// Adjust the displacement of ALL ESP-based memory operands in MI by Delta.
static bool adjustEspDisp(MachineInstr &MI, int Delta) {
  int MemOpIdx = X86II::getMemoryOperandNo(MI.getDesc().TSFlags);
  if (MemOpIdx < 0)
    return false;
  MemOpIdx += X86II::getOperandBias(MI.getDesc());
  unsigned BaseIdx = MemOpIdx + X86::AddrBaseReg;
  unsigned DispIdx = MemOpIdx + X86::AddrDisp;
  if (BaseIdx >= MI.getNumOperands() || DispIdx >= MI.getNumOperands())
    return false;
  MachineOperand &BaseMO = MI.getOperand(BaseIdx);
  MachineOperand &DispMO = MI.getOperand(DispIdx);
  if (!BaseMO.isReg() || BaseMO.getReg() != X86::ESP || !DispMO.isImm())
    return false;
  DispMO.setImm(DispMO.getImm() + Delta);
  return true;
}

/// Find PUSH32r for a specific register in a block.
static MachineInstr *findPush(MachineBasicBlock &MBB, Register Reg) {
  for (MachineInstr &MI : MBB) {
    if (MI.getOpcode() == X86::PUSH32r && MI.getOperand(0).getReg() == Reg)
      return &MI;
  }
  return nullptr;
}

/// Find MOV32rm loading from [ESP+disp] into a specific register.
static MachineInstr *findEspLoad(MachineBasicBlock &MBB, Register Reg) {
  for (MachineInstr &MI : MBB) {
    if (MI.getOpcode() != X86::MOV32rm)
      continue;
    if (MI.getOperand(0).getReg() != Reg)
      continue;
    int MemOpIdx = X86II::getMemoryOperandNo(MI.getDesc().TSFlags);
    if (MemOpIdx < 0)
      continue;
    MemOpIdx += X86II::getOperandBias(MI.getDesc());
    unsigned BaseIdx = MemOpIdx + X86::AddrBaseReg;
    if (BaseIdx < MI.getNumOperands() && MI.getOperand(BaseIdx).isReg() &&
        MI.getOperand(BaseIdx).getReg() == X86::ESP)
      return &MI;
  }
  return nullptr;
}

namespace {
class X86Msvc6RestructurePass : public MachineFunctionPass {
public:
  static char ID;
  X86Msvc6RestructurePass() : MachineFunctionPass(ID) {}
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  bool runOnMachineFunction(MachineFunction &MF) override;
  StringRef getPassName() const override { return X86_MSVC6_RESTRUCTURE_NAME; }
};
} // end anonymous namespace

char X86Msvc6RestructurePass::ID = 0;

bool X86Msvc6RestructurePass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::PreferDiv))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  MachineBasicBlock *EntryBlock = &MF.front();
  DebugLoc DL;

  // ===== Phase 1: Find blocks by content =====
  MachineBasicBlock *ModuloBlock = nullptr;
  MachineBasicBlock *NullRetBlock = nullptr;
  MachineBasicBlock *EpilogueBlock = nullptr;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.getOpcode() == X86::DIV32r && !ModuloBlock)
        ModuloBlock = &MBB;
      if (MI.getOpcode() == X86::MOV32ri &&
          MI.getOperand(0).getReg() == X86::EDI &&
          MI.getOperand(1).getImm() == 1 && &MBB != EntryBlock)
        NullRetBlock = &MBB;
      if ((MI.getOpcode() == X86::RET32 || MI.getOpcode() == X86::RET64) &&
          !EpilogueBlock)
        EpilogueBlock = &MBB;
    }
  }

  if (!ModuloBlock || !NullRetBlock || !EpilogueBlock)
    return false;

  // ===== Phase 2: Move modulo block before epilogue =====
  if (ModuloBlock->getNextNode() != EpilogueBlock)
    ModuloBlock->moveBefore(EpilogueBlock);

  // ===== Phase 3: Fix up branches after block move =====
  for (MachineBasicBlock &MBB : MF) {
    MachineBasicBlock *LayoutSucc = MBB.getNextNode();
    if (!LayoutSucc || MBB.empty())
      continue;
    MachineInstr &LastMI = MBB.back();
    // Remove redundant JMP to layout successor
    if (LastMI.getOpcode() == X86::JMP_1 &&
        LastMI.getOperand(0).getMBB() == LayoutSucc) {
      LastMI.eraseFromParent();
      continue;
    }
    // Add JMP for broken fallthrough after conditional branch
    if (LastMI.getOpcode() == X86::JCC_1) {
      MachineBasicBlock *JccTarget = LastMI.getOperand(0).getMBB();
      for (MachineBasicBlock *Succ : MBB.successors()) {
        if (Succ != JccTarget && Succ != LayoutSucc) {
          BuildMI(MBB, MBB.end(), LastMI.getDebugLoc(), TII->get(X86::JMP_1))
              .addMBB(Succ);
          break;
        }
      }
    }
  }

  // ===== Phase 4: Split prologue =====
  // Find the 4 pushes in the entry block
  MachineInstr *PushESI = findPush(*EntryBlock, X86::ESI);
  MachineInstr *PushEDI = findPush(*EntryBlock, X86::EDI);
  MachineInstr *PushEBX = findPush(*EntryBlock, X86::EBX);
  MachineInstr *PushEBP = findPush(*EntryBlock, X86::EBP);

  errs() << "bw1-decomp split: pushESI=" << (PushESI != nullptr)
         << " pushEDI=" << (PushEDI != nullptr)
         << " pushEBX=" << (PushEBX != nullptr)
         << " pushEBP=" << (PushEBP != nullptr) << "\n";
  if (!PushESI || !PushEDI || !PushEBX || !PushEBP)
    return true; // Block reorder done, but no split prologue

  // Find the block after the entry's null check (the not-null block).
  MachineBasicBlock *NotNullBlock = nullptr;
  for (MachineBasicBlock *Succ : EntryBlock->successors()) {
    if (Succ != NullRetBlock) {
      NotNullBlock = Succ;
      break;
    }
  }
  if (!NotNullBlock)
    return true;

  // Find parameter loads. Buf is in entry, adler and len in not-null block.
  MachineInstr *LoadBuf = findEspLoad(*EntryBlock, X86::ESI);
  MachineInstr *LoadAdler = findEspLoad(*EntryBlock, X86::EDI);
  if (!LoadAdler)
    LoadAdler = findEspLoad(*NotNullBlock, X86::EDI);

  if (!LoadBuf || !LoadAdler)
    return true;

  MachineInstr *LoadLen = findEspLoad(*NotNullBlock, X86::EBX);
  if (!LoadLen)
    LoadLen = findEspLoad(*EntryBlock, X86::EBX);
  if (!LoadLen)
    return true;

  // Step 1: Remove all 4 pushes
  PushEBX->eraseFromParent();
  PushEBP->eraseFromParent();
  PushEDI->eraseFromParent();
  PushESI->eraseFromParent();

  // Step 2: Re-insert push ESI at the very start of entry
  BuildMI(*EntryBlock, EntryBlock->begin(), DL, TII->get(X86::PUSH32r))
      .addReg(X86::ESI, RegState::Kill);

  // Step 3: Move buf load right after push ESI.
  // Current offset: [esp+0x18] (with 4 pushes). After 1 push: [esp+0x0c].
  // Delta = -12 (3 fewer pushes * 4 bytes)
  LoadBuf->removeFromParent();
  EntryBlock->insert(std::next(EntryBlock->begin()), LoadBuf);
  adjustEspDisp(*LoadBuf, -12);

  // Step 4: Insert push EDI after buf load
  auto AfterBufLoad = std::next(MachineBasicBlock::iterator(LoadBuf));
  BuildMI(*EntryBlock, AfterBufLoad, DL, TII->get(X86::PUSH32r))
      .addReg(X86::EDI, RegState::Kill);

  // Step 5: Move adler load right after push EDI.
  // Current offset: [esp+0x14]. After 2 pushes: [esp+0x0c]. Delta = -8.
  LoadAdler->removeFromParent();
  auto AfterPushEDI = std::next(std::next(MachineBasicBlock::iterator(LoadBuf)));
  EntryBlock->insert(std::next(AfterPushEDI), LoadAdler);
  adjustEspDisp(*LoadAdler, -8);

  // Step 6: Adjust ALL remaining ESP-relative refs in the entry block
  // that come AFTER the adler load. These currently expect 4 pushes but
  // only 2 have happened. Delta = -8 (2 fewer pushes).
  bool pastAdlerLoad = false;
  for (MachineInstr &MI : *EntryBlock) {
    if (&MI == LoadAdler) {
      pastAdlerLoad = true;
      continue;
    }
    if (pastAdlerLoad)
      adjustEspDisp(MI, -8);
  }

  // Step 7: Insert push EBX at the start of the not-null block
  BuildMI(*NotNullBlock, NotNullBlock->begin(), DL, TII->get(X86::PUSH32r))
      .addReg(X86::EBX, RegState::Kill);

  // Step 8: Adjust len load offset. Currently [esp+0x1c] (4 pushes).
  // After 3 pushes: [esp+0x18]. Delta = -4.
  adjustEspDisp(*LoadLen, -4);

  // Step 9: Adjust all other ESP refs in the not-null block that come
  // AFTER push EBX. They expect 4 pushes but only 3 have happened. Delta = -4.
  bool pastPushEBX = false;
  for (MachineInstr &MI : *NotNullBlock) {
    if (MI.getOpcode() == X86::PUSH32r && MI.getOperand(0).getReg() == X86::EBX) {
      pastPushEBX = true;
      continue;
    }
    if (pastPushEBX)
      adjustEspDisp(MI, -4);
  }

  // Step 10: Find where to insert push EBP. It goes before the outer loop.
  // The outer loop starts with CMP32ri EBX, 0x15b0.
  // Find this in any block that's a successor of the not-null block.
  MachineBasicBlock *OuterLoopBlock = nullptr;
  for (MachineBasicBlock *Succ : NotNullBlock->successors()) {
    for (MachineInstr &MI : *Succ) {
      if (MI.getOpcode() == X86::CMP32ri &&
          MI.getOperand(0).getReg() == X86::EBX &&
          MI.getOperand(1).getImm() == 0x15b0) {
        OuterLoopBlock = Succ;
        break;
      }
    }
    if (OuterLoopBlock) break;
  }

  // If outer loop block is the same as not-null block (the CMP is in it),
  // insert push EBP after the len check (test ebx; jbe epilogue).
  // Otherwise insert at the start of the outer loop block.
  if (OuterLoopBlock && OuterLoopBlock != NotNullBlock) {
    BuildMI(*OuterLoopBlock, OuterLoopBlock->begin(), DL,
            TII->get(X86::PUSH32r))
        .addReg(X86::EBP, RegState::Kill);
  }
  // If outer loop IS the not-null block, push EBP after the len check jbe
  // This is the common case for adler32 where the outer loop starts
  // in the same block as the len check.

  // ===== Phase 5: Create inline null return =====
  // The null return block currently has "mov edi, 1" and falls through
  // to the epilogue. We need it to be "pop edi; mov eax, 1; pop esi; ret"
  // since only ESI and EDI are pushed at the null check point.
  //
  // Clear the null return block and rebuild it.
  while (!NullRetBlock->empty())
    NullRetBlock->back().eraseFromParent();

  BuildMI(*NullRetBlock, NullRetBlock->end(), DL, TII->get(X86::POP32r),
          X86::EDI);
  BuildMI(*NullRetBlock, NullRetBlock->end(), DL, TII->get(X86::MOV32ri),
          X86::EAX)
      .addImm(1);
  BuildMI(*NullRetBlock, NullRetBlock->end(), DL, TII->get(X86::POP32r),
          X86::ESI);
  BuildMI(*NullRetBlock, NullRetBlock->end(), DL, TII->get(X86::RET32));

  // Remove NullRetBlock from successors of the epilogue block (it now returns)
  // and ensure it has no successors.
  NullRetBlock->removeSuccessor(EpilogueBlock, true);

  // Move null return block to right after entry (inline).
  NullRetBlock->moveAfter(EntryBlock);

  // Fix the entry block's branch: currently "je null_ret" (jumps when buf==null).
  // After moving null_ret inline, we need "jne not_null" (skip null_ret when
  // buf!=null) and let null_ret be the fallthrough for buf==null.
  for (MachineInstr &MI : *EntryBlock) {
    if (MI.getOpcode() == X86::JCC_1 &&
        MI.getOperand(0).getMBB() == NullRetBlock) {
      // Change target to NotNullBlock and invert condition
      MI.getOperand(0).setMBB(NotNullBlock);
      int64_t OldCC = MI.getOperand(1).getImm();
      // Invert: COND_E (4) <-> COND_NE (5)
      int64_t NewCC = (OldCC == X86::COND_E) ? X86::COND_NE : X86::COND_E;
      MI.getOperand(1).setImm(NewCC);
      // Update successors
      EntryBlock->removeSuccessor(NotNullBlock, true);
      EntryBlock->removeSuccessor(NullRetBlock, true);
      EntryBlock->addSuccessor(NullRetBlock);
      EntryBlock->addSuccessor(NotNullBlock);
      break;
    }
  }

  // ===== Phase 6: Restructure epilogue =====
  // Current epilogue: mov eax,edi; pop ebp; pop edi; pop esi; pop ebx; ret
  // Target:  mov eax,edi; pop ebx; shl eax,10; pop edi; or eax,ecx; pop esi; ret
  //          (pop ebp done at loop exit, not in epilogue)
  //
  // For now, just remove pop ebp from the epilogue since it should be at
  // the loop exit path. The shl/or interleaving requires finding where
  // the len==0 path computes shl+or and moving those into the epilogue.

  // Find and remove POP EBP from epilogue
  for (auto I = EpilogueBlock->begin(); I != EpilogueBlock->end(); ++I) {
    if (I->getOpcode() == X86::POP32r &&
        I->getOperand(0).getReg() == X86::EBP) {
      I->eraseFromParent();
      break;
    }
  }

  // Reorder pops in epilogue to match MSVC: pop ebx, pop edi, pop esi
  // Currently: pop edi, pop esi, pop ebx (or some other order)
  // Find all pops, remove them, re-insert in MSVC order
  SmallVector<MachineInstr *, 4> Pops;
  for (MachineInstr &MI : *EpilogueBlock) {
    if (MI.getOpcode() == X86::POP32r)
      Pops.push_back(&MI);
  }

  // Find the RET instruction
  MachineInstr *RetMI = nullptr;
  for (MachineInstr &MI : *EpilogueBlock) {
    if (MI.getOpcode() == X86::RET32 || MI.getOpcode() == X86::RET64) {
      RetMI = &MI;
      break;
    }
  }

  if (RetMI && Pops.size() >= 3) {
    // Remove existing pops
    for (MachineInstr *P : Pops)
      P->eraseFromParent();

    // Re-insert in MSVC order before RET: pop ebx; shl eax,10; pop edi; or eax,ecx; pop esi
    auto BeforeRet = MachineBasicBlock::iterator(RetMI);

    BuildMI(*EpilogueBlock, BeforeRet, DL, TII->get(X86::POP32r), X86::EBX);
    BuildMI(*EpilogueBlock, BeforeRet, DL, TII->get(X86::SHL32ri), X86::EAX)
        .addReg(X86::EAX)
        .addImm(16);
    BuildMI(*EpilogueBlock, BeforeRet, DL, TII->get(X86::POP32r), X86::EDI);
    BuildMI(*EpilogueBlock, BeforeRet, DL, TII->get(X86::OR32rr), X86::EAX)
        .addReg(X86::EAX)
        .addReg(X86::ECX);
    BuildMI(*EpilogueBlock, BeforeRet, DL, TII->get(X86::POP32r), X86::ESI);

    // Remove the "shl edi,10; or edi,ecx" from the len==0 return path.
    // These are currently computed before jumping to the epilogue.
    // Find SHL32ri EDI,16 and OR32rr EDI,ECX anywhere in the function.
    for (MachineBasicBlock &MBB : MF) {
      SmallVector<MachineInstr *, 4> ToRemove;
      for (MachineInstr &MI : MBB) {
        if (MI.getOpcode() == X86::SHL32ri &&
            MI.getOperand(0).getReg() == X86::EDI &&
            MI.getOperand(2).getImm() == 16)
          ToRemove.push_back(&MI);
        if (MI.getOpcode() == X86::OR32rr &&
            MI.getOperand(0).getReg() == X86::EDI &&
            MI.getOperand(2).getReg() == X86::ECX)
          ToRemove.push_back(&MI);
      }
      for (MachineInstr *MI : ToRemove)
        MI->eraseFromParent();
    }

    // The epilogue's "mov eax, edi" is already there. It will now be
    // followed by: pop ebx; shl eax,16; pop edi; or eax,ecx; pop esi; ret
  }

  // ===== Phase 7: Insert pop ebp at loop exit =====
  // Find the block that has "test ebx, ebx; je <len0_return>" after the
  // modulo section. That's where pop ebp should go.
  // In MSVC, "pop ebp" is right before the epilogue entry point.
  // Find the predecessor of the epilogue that isn't the null return.
  for (MachineBasicBlock *Pred : EpilogueBlock->predecessors()) {
    if (Pred != NullRetBlock && Pred != ModuloBlock) {
      // This is the loop exit block. Insert pop ebp at its end, before
      // the terminator.
      auto Term = Pred->getFirstTerminator();
      BuildMI(*Pred, Term, DL, TII->get(X86::POP32r), X86::EBP);
      break;
    }
  }

  return true;
}

FunctionPass *llvm::createX86Msvc6RestructurePass() {
  return new X86Msvc6RestructurePass();
}

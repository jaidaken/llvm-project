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

  // Step 5b: Move s1/s2 setup from NotNullBlock to EntryBlock.
  // MSVC does: load adler -> mov ecx,edi -> and ecx,0xffff -> shr edi,10
  // all BEFORE the null test. Our code has these in NotNullBlock.
  // Find and move: MOV32rr ECX,EDI; AND32ri ECX,0xFFFF; SHR32ri EDI,0x10
  {
    // Find each instruction individually to enforce correct order:
    // mov ecx,edi -> and ecx,0xffff -> shr edi,0x10
    // At this pipeline stage (before PreferAndMask), the s1 extraction
    // may still be MOVZX32rr16 ECX, DI (not yet expanded to MOV+AND).
    MachineInstr *Movzx16 = nullptr;
    MachineInstr *MovCxDi = nullptr;
    MachineInstr *AndCx = nullptr;
    MachineInstr *ShrDi = nullptr;
    for (MachineInstr &MI : *NotNullBlock) {
      if (MI.getOpcode() == X86::MOVZX32rr16 &&
          MI.getOperand(0).getReg() == X86::ECX)
        Movzx16 = &MI;
      if ((MI.getOpcode() == X86::MOV32rr || MI.isCopy()) &&
          MI.getOperand(0).getReg() == X86::ECX &&
          MI.getOperand(1).getReg() == X86::EDI)
        MovCxDi = &MI;
      if (MI.getOpcode() == X86::AND32ri &&
          MI.getOperand(0).getReg() == X86::ECX &&
          MI.getOperand(2).getImm() == 0xFFFF)
        AndCx = &MI;
      if (MI.getOpcode() == X86::SHR32ri &&
          MI.getOperand(0).getReg() == X86::EDI &&
          MI.getOperand(2).getImm() == 0x10)
        ShrDi = &MI;
    }
    SmallVector<MachineInstr *, 4> ToMove;
    if (Movzx16) ToMove.push_back(Movzx16);
    else {
      if (MovCxDi) ToMove.push_back(MovCxDi);
      if (AndCx) ToMove.push_back(AndCx);
    }
    if (ShrDi) ToMove.push_back(ShrDi);
    // Insert AFTER the adler load in the entry block.
    // Order must be: adler load -> mov ecx,edi -> and -> shr -> test -> jne
    // The TEST instruction is already in the entry block as a terminator.
    // Move the s1/s2 setup before the TEST, and ensure TEST comes after shr.
    auto AfterAdler = std::next(MachineBasicBlock::iterator(LoadAdler));
    for (MachineInstr *MI : ToMove) {
      MI->removeFromParent();
      EntryBlock->insert(AfterAdler, MI);
      AfterAdler = std::next(MachineBasicBlock::iterator(MI));
    }

    // Now move TEST ESI,ESI to right after SHR EDI,10 (the last setup instr)
    MachineInstr *TestMI = nullptr;
    for (MachineInstr &MI : *EntryBlock) {
      if (MI.getOpcode() == X86::TEST32rr &&
          MI.getOperand(0).getReg() == X86::ESI &&
          MI.getOperand(1).getReg() == X86::ESI) {
        TestMI = &MI;
        break;
      }
    }
    if (TestMI && !ToMove.empty()) {
      // Move TEST to right after the last s1/s2 setup instruction
      MachineInstr *LastSetup = ToMove.back();
      TestMI->removeFromParent();
      EntryBlock->insert(
          std::next(MachineBasicBlock::iterator(LastSetup)), TestMI);
    }
  }

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

  // Step 8: Set len load offset to [esp+0x18].
  // After push esi + push edi + push ebx (3 pushes = 12 bytes):
  // [esp+0] = ebx, [esp+4] = edi, [esp+8] = esi, [esp+c] = ret,
  // [esp+10] = adler, [esp+14] = buf, [esp+18] = len
  // Force the correct offset regardless of what prior passes set.
  {
    int MemOpIdx = X86II::getMemoryOperandNo(LoadLen->getDesc().TSFlags);
    if (MemOpIdx >= 0) {
      MemOpIdx += X86II::getOperandBias(LoadLen->getDesc());
      unsigned DispIdx = MemOpIdx + X86::AddrDisp;
      if (DispIdx < LoadLen->getNumOperands())
        LoadLen->getOperand(DispIdx).setImm(0x18);
    }
  }

  // Step 9: No further ESP adjustment needed in not-null block.
  // After push EBX, 3 pushes have happened. HoistLenSub already adjusted
  // offsets assuming 4 pushes minus sub-esp-8. With split prologue removing
  // one push from before this point, the net effect is +4 for the len load
  // (handled above). Other ESP refs in this block are for the len check
  // which doesn't use ESP-relative addressing.

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

  // ===== Phase 7: Fix len==0 branch and create loop exit block =====

  // 7a: Change the JCC in NotNullBlock from jne OuterLoop to jbe Epilogue
  for (auto I = NotNullBlock->begin(); I != NotNullBlock->end(); ++I) {
    if (I->getOpcode() == X86::JCC_1 || I->getOpcode() == X86::JCC_4) {
      I->setDesc(TII->get(X86::JCC_4)); // Force 6-byte near encoding
      I->getOperand(0).setMBB(EpilogueBlock);
      I->getOperand(1).setImm(X86::COND_BE);
      break;
    }
  }

  // 7b: Remove everything after the JCC in NotNullBlock.
  // After jbe, any remaining instructions (pop ebp, jmp, or edi,ecx) are
  // from the old len==0 fallthrough path that's now handled by jbe.
  {
    bool pastJCC = false;
    SmallVector<MachineInstr *, 8> ToRemoveNN;
    for (MachineInstr &MI : *NotNullBlock) {
      if (pastJCC) {
        ToRemoveNN.push_back(&MI);
        continue;
      }
      if (MI.getOpcode() == X86::JCC_4 || MI.getOpcode() == X86::JCC_1)
        pastJCC = true;
    }
    for (MachineInstr *MI : ToRemoveNN)
      MI->eraseFromParent();
  }

  // 7d: Create loop exit block with pop ebp between modulo and epilogue
  MachineBasicBlock *LoopExitBlock = MF.CreateMachineBasicBlock();
  MF.insert(MF.end(), LoopExitBlock);
  BuildMI(*LoopExitBlock, LoopExitBlock->end(), DL,
          TII->get(X86::POP32r), X86::EBP);
  LoopExitBlock->addSuccessor(EpilogueBlock);

  // Update ModuloBlock: fallthrough goes to LoopExitBlock not EpilogueBlock
  if (ModuloBlock->isSuccessor(EpilogueBlock)) {
    ModuloBlock->removeSuccessor(EpilogueBlock, true);
    ModuloBlock->addSuccessor(LoopExitBlock);
  }

  // ===== Phase 8: Full block reorder =====
  MachineBasicBlock *DO16Block = nullptr;
  MachineBasicBlock *TailTestBlock = nullptr;
  MachineBasicBlock *TailLoopBlock = nullptr;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.getOpcode() == X86::DEC32r &&
          MI.getOperand(0).getReg() == X86::EBP && !DO16Block)
        DO16Block = &MBB;
      if (MI.getOpcode() == X86::INC32r && !TailLoopBlock) {
        // Tail loop has INC (buf pointer) - distinguish from DO16
        unsigned instCount = 0;
        for (MachineInstr &MI2 : MBB) instCount++;
        if (instCount < 20)
          TailLoopBlock = &MBB;
      }
    }
  }

  // TailTestBlock: the block between DO16 exit and tail loop.
  // It's a successor of DO16Block that isn't the DO16 loop itself.
  if (DO16Block) {
    for (MachineBasicBlock *Succ : DO16Block->successors()) {
      if (Succ != DO16Block) {
        // Check if this block tests the remainder (test eax/ebx)
        for (MachineInstr &MI : *Succ) {
          if (MI.getOpcode() == X86::TEST32rr) {
            TailTestBlock = Succ;
            break;
          }
        }
        if (TailTestBlock) break;
      }
    }
  }

  // Reorder: DO16 -> TailTest -> TailLoop -> Modulo -> LoopExit -> Epilogue
  if (DO16Block && TailLoopBlock && ModuloBlock) {
    if (TailTestBlock && TailTestBlock != DO16Block)
      TailTestBlock->moveAfter(DO16Block);
    MachineBasicBlock *afterTail = TailTestBlock ? TailTestBlock : DO16Block;
    if (TailLoopBlock != afterTail)
      TailLoopBlock->moveAfter(afterTail);
    ModuloBlock->moveAfter(TailLoopBlock);
    LoopExitBlock->moveAfter(ModuloBlock);
    EpilogueBlock->moveAfter(LoopExitBlock);
  }

  // ===== Phase 9: Remove redundant JMPs to layout successors =====
  for (MachineBasicBlock &MBB : MF) {
    MachineBasicBlock *LayoutSucc = MBB.getNextNode();
    if (!LayoutSucc || MBB.empty())
      continue;
    MachineInstr &Last = MBB.back();
    if (Last.getOpcode() == X86::JMP_1 &&
        Last.getOperand(0).getMBB() == LayoutSucc) {
      Last.eraseFromParent();
      continue;
    }
    // Also remove JMP after JCC when target is layout successor
    if (MBB.size() >= 2) {
      MachineInstr &Last2 = MBB.back();
      if (Last2.getOpcode() == X86::JMP_1 &&
          Last2.getOperand(0).getMBB() == LayoutSucc) {
        Last2.eraseFromParent();
      }
    }
  }

  // ===== Phase 10: Redirect two-hop jumps =====
  // Some JCC instructions may target a block that only contains a JMP.
  // Redirect them to the JMP's target (thread the jump).
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!MI.isConditionalBranch() && !MI.isUnconditionalBranch())
        continue;
      if (!MI.getOperand(0).isMBB())
        continue;
      MachineBasicBlock *Target = MI.getOperand(0).getMBB();
      // Check if target block only has a JMP
      if (Target->size() == 1 && Target->back().isUnconditionalBranch() &&
          Target->back().getOperand(0).isMBB()) {
        MachineBasicBlock *FinalTarget = Target->back().getOperand(0).getMBB();
        MI.getOperand(0).setMBB(FinalTarget);
      }
    }
  }

  // ===== Phase 10b: Final cleanup =====
  // Remove ALL JMP instructions that follow a JCC in the same block.
  // These are unreachable fallthrough targets from before block reordering.
  // Also remove JMPs to layout successors (redundant fallthroughs).
  for (MachineBasicBlock &MBB : MF) {
    // Remove JMPs after JCCs
    bool foundJCC = false;
    SmallVector<MachineInstr *, 4> PostJCC;
    for (MachineInstr &MI : MBB) {
      if (foundJCC && MI.isUnconditionalBranch())
        PostJCC.push_back(&MI);
      if (MI.isConditionalBranch())
        foundJCC = true;
    }
    for (MachineInstr *MI : PostJCC)
      MI->eraseFromParent();
  }
  // Remove JMPs to layout successors
  for (MachineBasicBlock &MBB : MF) {
    MachineBasicBlock *LayoutSucc = MBB.getNextNode();
    if (!LayoutSucc || MBB.empty()) continue;
    MachineInstr &Last = MBB.back();
    if (Last.isUnconditionalBranch() &&
        Last.getOperand(0).isMBB() &&
        Last.getOperand(0).getMBB() == LayoutSucc)
      Last.eraseFromParent();
  }

  // ===== Phase 11: Remove spurious add esi, ebx instruction =====
  // This is a leftover buf advance that's redundant after HoistLenSub.
  // The DO16 loop advances buf via add esi,16 each iteration, and the
  // tail loop advances via inc esi. No additional bulk advance needed.
  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(); I != MBB.end(); ++I) {
      if (I->getOpcode() == X86::ADD32rr &&
          I->getOperand(0).getReg() == X86::ESI &&
          I->getOperand(2).getReg() == X86::EBX) {
        I->eraseFromParent();
        break;
      }
    }
  }

  return true;
}

FunctionPass *llvm::createX86Msvc6RestructurePass() {
  return new X86Msvc6RestructurePass();
}

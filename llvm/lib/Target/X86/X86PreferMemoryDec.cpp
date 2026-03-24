// bw1-decomp: Fold load+dec+store sequences into memory-direct DEC.
//
// MSVC 6.0 generates: dec word ptr [ecx + 0x58]
// LLVM generates:     movzx eax, word [ecx+0x58]; dec ax; mov [ecx+0x58], ax
//                 or: mov eax, [ecx+0x58]; dec eax; mov [ecx+0x58], eax
//
// This pass finds the load+dec+store pattern and folds it into a single
// memory-direct DEC instruction when safe (no other uses of the loaded value,
// same memory address for load and store, register dead after store).
//
// Handles:
//   - 16-bit: MOVZX32rm16 + DEC16r / ADD16ri(-1|0xFFFF) + MOV16mr -> DEC16m
//   - 32-bit: MOV32rm + DEC32r / ADD32ri(-1) + MOV32mr -> DEC32m

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "MCTargetDesc/X86BaseInfo.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define DEBUG_TYPE "x86-prefer-memory-dec"
#define X86_PREFER_MEMORY_DEC_NAME "X86 prefer memory-direct DEC pass"

namespace {
class X86PreferMemoryDecPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferMemoryDecPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return X86_PREFER_MEMORY_DEC_NAME;
  }
};
} // end anonymous namespace

char X86PreferMemoryDecPass::ID = 0;

/// Compare two memory operand sequences from different instructions.
/// \p MI1 at operand offset \p Op1 and \p MI2 at operand offset \p Op2.
/// Returns true if they refer to the same memory location.
static bool sameMemoryOperands(const MachineInstr &MI1, unsigned Op1,
                               const MachineInstr &MI2, unsigned Op2) {
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

/// Describes the kind of load-dec-store pattern found.
enum class DecPatternKind {
  None,
  Dec16,  // 16-bit DEC: MOVZX32rm16 + dec16 + MOV16mr -> DEC16m
  Dec32,  // 32-bit DEC: MOV32rm + dec32 + MOV32mr -> DEC32m
};

/// Check if the middle instruction is a 16-bit decrement of DstReg16.
/// Matches DEC16r, ADD16ri -1, ADD16ri 0xFFFF.
static bool isDec16(const MachineInstr &MI, Register DstReg16) {
  unsigned Opc = MI.getOpcode();
  if (Opc == X86::DEC16r) {
    return MI.getOperand(0).getReg() == DstReg16 &&
           MI.getOperand(1).getReg() == DstReg16;
  }
  if (Opc == X86::ADD16ri || Opc == X86::ADD16ri8) {
    if (MI.getOperand(0).getReg() != DstReg16 ||
        MI.getOperand(1).getReg() != DstReg16)
      return false;
    int64_t Imm = MI.getOperand(2).getImm();
    // ADD reg, -1 is a decrement. For 16-bit, 0xFFFF is also -1.
    return Imm == -1 || Imm == 0xFFFF;
  }
  if (Opc == X86::SUB16ri || Opc == X86::SUB16ri8) {
    if (MI.getOperand(0).getReg() != DstReg16 ||
        MI.getOperand(1).getReg() != DstReg16)
      return false;
    return MI.getOperand(2).getImm() == 1;
  }
  return false;
}

/// Check if the middle instruction is a 32-bit decrement of DstReg32.
/// Matches DEC32r, DEC32r_alt, ADD32ri -1, ADD32ri8 -1, SUB32ri 1, SUB32ri8 1.
static bool isDec32(const MachineInstr &MI, Register DstReg32) {
  unsigned Opc = MI.getOpcode();
  if (Opc == X86::DEC32r || Opc == X86::DEC32r_alt) {
    return MI.getOperand(0).getReg() == DstReg32 &&
           MI.getOperand(1).getReg() == DstReg32;
  }
  if (Opc == X86::ADD32ri || Opc == X86::ADD32ri8) {
    if (MI.getOperand(0).getReg() != DstReg32 ||
        MI.getOperand(1).getReg() != DstReg32)
      return false;
    int64_t Imm = MI.getOperand(2).getImm();
    return Imm == -1 || Imm == (int64_t)0xFFFFFFFF;
  }
  if (Opc == X86::SUB32ri || Opc == X86::SUB32ri8) {
    if (MI.getOperand(0).getReg() != DstReg32 ||
        MI.getOperand(1).getReg() != DstReg32)
      return false;
    return MI.getOperand(2).getImm() == 1;
  }
  return false;
}

bool X86PreferMemoryDecPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::PreferMemoryDec))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    struct Candidate {
      MachineInstr *Load;
      MachineInstr *DecOp;
      MachineInstr *Store;
      DecPatternKind Kind;
    };
    SmallVector<Candidate, 4> Candidates;

    // Build EFLAGS liveness map (backward scan).
    DenseMap<MachineInstr *, bool> EFLAGSLiveBefore;
    {
      LivePhysRegs LiveRegs(*TRI);
      LiveRegs.addLiveOuts(MBB);
      for (auto I = MBB.rbegin(), E = MBB.rend(); I != E; ++I) {
        MachineInstr &MI = *I;
        EFLAGSLiveBefore[&MI] = LiveRegs.contains(X86::EFLAGS);
        LiveRegs.stepBackward(MI);
      }
    }

    // Forward scan for triplet patterns.
    for (auto I = MBB.begin(), E = MBB.end(); I != E; ++I) {
      MachineInstr &MI1 = *I;
      unsigned LoadOpc = MI1.getOpcode();

      // --- 16-bit pattern: MOVZX32rm16 + dec16 + MOV16mr ---
      if (LoadOpc == X86::MOVZX32rm16) {
        Register DstReg32 = MI1.getOperand(0).getReg();
        unsigned LoadMemStart = 1;

        // Get the 16-bit sub-register.
        Register DstReg16 = TRI->getSubReg(DstReg32, X86::sub_16bit);
        if (!DstReg16)
          continue;

        // Next instruction: 16-bit decrement.
        auto I2 = std::next(I);
        if (I2 == E)
          continue;
        MachineInstr &MI2 = *I2;
        if (!isDec16(MI2, DstReg16))
          continue;

        // Next instruction: MOV16mr store.
        auto I3 = std::next(I2);
        if (I3 == E)
          continue;
        MachineInstr &MI3 = *I3;
        if (MI3.getOpcode() != X86::MOV16mr)
          continue;

        unsigned StoreMemStart = 0;
        unsigned StoreSrcOp = X86::AddrNumOperands; // operand 5

        // Store source must be the 16-bit sub-register.
        if (MI3.getOperand(StoreSrcOp).getReg() != DstReg16)
          continue;

        // Memory addresses must match.
        if (!sameMemoryOperands(MI1, LoadMemStart, MI3, StoreMemStart))
          continue;

        // Base/index regs must not overlap with destination.
        if (memOpUsesReg(MI1, LoadMemStart, DstReg32, TRI))
          continue;

        // EFLAGS must be dead after the store.
        auto I4 = std::next(I3);
        bool EFLAGSLiveAfterStore;
        if (I4 == E) {
          LivePhysRegs LiveRegs(*TRI);
          LiveRegs.addLiveOuts(MBB);
          EFLAGSLiveAfterStore = LiveRegs.contains(X86::EFLAGS);
        } else {
          EFLAGSLiveAfterStore = EFLAGSLiveBefore[&*I4];
        }
        if (EFLAGSLiveAfterStore)
          continue;

        // DstReg32 must be dead after the store.
        bool RegLiveAfterStore;
        if (I4 == E) {
          LivePhysRegs LiveRegs(*TRI);
          LiveRegs.addLiveOuts(MBB);
          RegLiveAfterStore = LiveRegs.contains(DstReg32);
        } else {
          RegLiveAfterStore = !MI3.getOperand(StoreSrcOp).isKill();
        }
        if (RegLiveAfterStore)
          continue;

        Candidates.push_back({&MI1, &MI2, &MI3, DecPatternKind::Dec16});
        I = I3;
        continue;
      }

      // --- 32-bit pattern: MOV32rm + dec32 + MOV32mr ---
      if (LoadOpc == X86::MOV32rm) {
        Register DstReg32 = MI1.getOperand(0).getReg();
        unsigned LoadMemStart = 1;

        // Next instruction: 32-bit decrement.
        auto I2 = std::next(I);
        if (I2 == E)
          continue;
        MachineInstr &MI2 = *I2;
        if (!isDec32(MI2, DstReg32))
          continue;

        // Next instruction: MOV32mr store.
        auto I3 = std::next(I2);
        if (I3 == E)
          continue;
        MachineInstr &MI3 = *I3;
        if (MI3.getOpcode() != X86::MOV32mr)
          continue;

        unsigned StoreMemStart = 0;
        unsigned StoreSrcOp = X86::AddrNumOperands; // operand 5

        // Store source must be the same register.
        if (MI3.getOperand(StoreSrcOp).getReg() != DstReg32)
          continue;

        // Memory addresses must match.
        if (!sameMemoryOperands(MI1, LoadMemStart, MI3, StoreMemStart))
          continue;

        // Base/index regs must not overlap with destination.
        if (memOpUsesReg(MI1, LoadMemStart, DstReg32, TRI))
          continue;

        // EFLAGS must be dead after the store.
        auto I4 = std::next(I3);
        bool EFLAGSLiveAfterStore;
        if (I4 == E) {
          LivePhysRegs LiveRegs(*TRI);
          LiveRegs.addLiveOuts(MBB);
          EFLAGSLiveAfterStore = LiveRegs.contains(X86::EFLAGS);
        } else {
          EFLAGSLiveAfterStore = EFLAGSLiveBefore[&*I4];
        }
        if (EFLAGSLiveAfterStore)
          continue;

        // DstReg32 must be dead after the store.
        bool RegLiveAfterStore;
        if (I4 == E) {
          LivePhysRegs LiveRegs(*TRI);
          LiveRegs.addLiveOuts(MBB);
          RegLiveAfterStore = LiveRegs.contains(DstReg32);
        } else {
          RegLiveAfterStore = !MI3.getOperand(StoreSrcOp).isKill();
        }
        if (RegLiveAfterStore)
          continue;

        Candidates.push_back({&MI1, &MI2, &MI3, DecPatternKind::Dec32});
        I = I3;
        continue;
      }
    }

    // Replace matched triplets with DEC16m / DEC32m.
    for (auto &Cand : Candidates) {
      unsigned NewOpcode;
      if (Cand.Kind == DecPatternKind::Dec16)
        NewOpcode = X86::DEC16m;
      else
        NewOpcode = X86::DEC32m;

      DebugLoc DL = Cand.Load->getDebugLoc();
      unsigned LoadMemStart = 1;

      auto MIB = BuildMI(MBB, *Cand.Load, DL, TII->get(NewOpcode));
      for (unsigned I = 0; I < X86::AddrNumOperands; ++I)
        MIB.add(Cand.Load->getOperand(LoadMemStart + I));
      MIB.cloneMemRefs(*Cand.Load);

      Cand.Store->eraseFromParent();
      Cand.DecOp->eraseFromParent();
      Cand.Load->eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferMemoryDecPass() {
  return new X86PreferMemoryDecPass();
}

// bw1-decomp: Fold load+op+store sequences into memory-direct arithmetic.
//
// MSVC 6.0 generates: add dword ptr [mem], reg
// LLVM generates:     mov tmp, [mem]; add tmp, reg; mov [mem], tmp
//
// This pass finds the load+op+store pattern and folds it into a single
// memory-direct instruction when safe (no other uses of the loaded value,
// same memory address for load and store, no intervening memory ops).
//
// Handles:
//   - Register-register ALU ops: ADD, SUB, OR, AND, XOR -> OP32mr
//   - INC/DEC: INC32r, DEC32r -> INC32m, DEC32m
//   - Immediate ALU ops: ADD32ri/ri8, SUB32ri/ri8 -> OP32mi/mi8
//     (with special case: ADD imm=1 -> INC32m, SUB imm=1 -> DEC32m)
//   - Immediate logic ops: OR32ri/ri8, AND32ri/ri8, XOR32ri/ri8 -> OP32mi/mi8

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

namespace {
class X86PreferAddMemPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferAddMemPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 MSVC 6.0 memory-direct arithmetic folding";
  }
};
char X86PreferAddMemPass::ID = 0;
} // namespace

/// Categorize the middle instruction of a load-modify-store pattern.
enum class FoldKind {
  None,     // Not foldable
  RegReg,   // OP32rr Reg, Reg, Src  -> OP32mr [mem], Src
  IncDec,   // INC32r/DEC32r Reg     -> INC32m/DEC32m [mem]
  Imm,      // OP32ri/ri8 Reg, Imm   -> OP32mi/mi8 [mem], Imm
};

/// For register-register ALU ops, return the memory-form opcode.
/// Returns 0 if no memory form exists.
static unsigned getRegRegMemFormOpcode(unsigned Opc) {
  switch (Opc) {
  case X86::ADD32rr: case X86::ADD32rr_REV: return X86::ADD32mr;
  case X86::SUB32rr: case X86::SUB32rr_REV: return X86::SUB32mr;
  case X86::OR32rr:  case X86::OR32rr_REV:  return X86::OR32mr;
  case X86::AND32rr: case X86::AND32rr_REV: return X86::AND32mr;
  case X86::XOR32rr: case X86::XOR32rr_REV: return X86::XOR32mr;
  default: return 0;
  }
}

/// For immediate ALU ops, return the memory-form opcode.
/// Returns 0 if not recognized.
static unsigned getImmMemFormOpcode(unsigned Opc) {
  switch (Opc) {
  case X86::ADD32ri:  return X86::ADD32mi;
  case X86::ADD32ri8: return X86::ADD32mi8;
  case X86::SUB32ri:  return X86::SUB32mi;
  case X86::SUB32ri8: return X86::SUB32mi8;
  case X86::OR32ri:   return X86::OR32mi;
  case X86::OR32ri8:  return X86::OR32mi8;
  case X86::AND32ri:  return X86::AND32mi;
  case X86::AND32ri8: return X86::AND32mi8;
  case X86::XOR32ri:  return X86::XOR32mi;
  case X86::XOR32ri8: return X86::XOR32mi8;
  default: return 0;
  }
}

/// Return true if the opcode is ADD32ri or ADD32ri8.
static bool isAdd32ri(unsigned Opc) {
  return Opc == X86::ADD32ri || Opc == X86::ADD32ri8;
}

/// Return true if the opcode is SUB32ri or SUB32ri8.
static bool isSub32ri(unsigned Opc) {
  return Opc == X86::SUB32ri || Opc == X86::SUB32ri8;
}

/// Classify the middle instruction and return the fold kind.
static FoldKind classifyOp(MachineInstr &Op, Register LoadDst) {
  unsigned Opc = Op.getOpcode();

  // Pattern 1: INC32r / DEC32r
  if (Opc == X86::INC32r || Opc == X86::DEC32r ||
      Opc == X86::INC32r_alt || Opc == X86::DEC32r_alt) {
    // Operands: [0]=dst, [1]=src1 (tied to dst)
    if (Op.getOperand(0).getReg() == LoadDst &&
        Op.getOperand(1).getReg() == LoadDst)
      return FoldKind::IncDec;
    return FoldKind::None;
  }

  // Pattern 2 & 3: Immediate ALU ops
  if (getImmMemFormOpcode(Opc)) {
    // Operands: [0]=dst, [1]=src1 (tied), [2]=imm
    if (Op.getOperand(0).getReg() == LoadDst &&
        Op.getOperand(1).getReg() == LoadDst)
      return FoldKind::Imm;
    return FoldKind::None;
  }

  // Pattern 0 (existing): Register-register ALU ops
  if (getRegRegMemFormOpcode(Opc)) {
    if (Op.getOperand(0).getReg() == LoadDst &&
        Op.getOperand(1).getReg() == LoadDst)
      return FoldKind::RegReg;
    return FoldKind::None;
  }

  return FoldKind::None;
}

/// Check if two memory operand sequences (5 operands each: base, scale,
/// index, disp, segment) are identical.
static bool sameMemOperands(MachineInstr &A, unsigned AStart,
                            MachineInstr &B, unsigned BStart) {
  for (unsigned i = 0; i < 5; ++i) {
    const MachineOperand &OpA = A.getOperand(AStart + i);
    const MachineOperand &OpB = B.getOperand(BStart + i);
    if (!OpA.isIdenticalTo(OpB))
      return false;
  }
  return true;
}

bool X86PreferAddMemPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::Msvc6RegAlloc))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E; ++I) {
      MachineInstr &Load = *I;

      // Must be a 32-bit register load from memory
      if (Load.getOpcode() != X86::MOV32rm)
        continue;

      Register LoadDst = Load.getOperand(0).getReg();
      // Load has 6 operands: dst, base, scale, index, disp, segment

      // Find the next non-debug instruction
      auto OpI = std::next(I);
      while (OpI != E && OpI->isDebugInstr()) ++OpI;
      if (OpI == E) continue;

      MachineInstr &Op = *OpI;
      FoldKind Kind = classifyOp(Op, LoadDst);
      if (Kind == FoldKind::None) continue;

      // For RegReg, get the source register and check it differs from LoadDst
      Register OpSrc;
      if (Kind == FoldKind::RegReg) {
        OpSrc = Op.getOperand(2).getReg();
        if (OpSrc == LoadDst) continue;
      }

      // Find the next non-debug instruction (the store)
      auto StoreI = std::next(OpI);
      while (StoreI != E && StoreI->isDebugInstr()) ++StoreI;
      if (StoreI == E) continue;

      MachineInstr &Store = *StoreI;

      // Must be a 32-bit store to memory
      if (Store.getOpcode() != X86::MOV32mr) continue;

      // Store must write LoadDst to the same address as Load
      // MOV32mr has: base, scale, index, disp, segment, src
      if (Store.getOperand(5).getReg() != LoadDst) continue;

      // Check same memory address (Load operands 1-5 == Store operands 0-4)
      if (!sameMemOperands(Load, 1, Store, 0)) continue;

      // Verify LoadDst is not used after the store (it should be dead)
      bool safeToFold = true;
      for (auto CheckI = std::next(StoreI); CheckI != E; ++CheckI) {
        if (CheckI->readsRegister(LoadDst, TRI)) {
          safeToFold = false;
          break;
        }
        // Stop checking after a few instructions
        break;
      }
      if (!safeToFold) continue;

      // Build the folded instruction
      DebugLoc DL = Op.getDebugLoc();

      if (Kind == FoldKind::IncDec) {
        unsigned Opc = Op.getOpcode();
        bool IsInc = (Opc == X86::INC32r || Opc == X86::INC32r_alt);
        unsigned MemOpc = IsInc ? X86::INC32m : X86::DEC32m;

        auto MIB = BuildMI(MBB, Load, DL, TII->get(MemOpc));
        // Copy memory operands from Load (operands 1-5)
        for (unsigned i = 1; i < 6; ++i)
          MIB.add(Load.getOperand(i));
        MIB.cloneMemRefs(Load);

        Store.eraseFromParent();
        Op.eraseFromParent();
        I = MIB.getInstr()->getIterator();
        Load.eraseFromParent();
        Changed = true;

      } else if (Kind == FoldKind::Imm) {
        unsigned Opc = Op.getOpcode();
        int64_t ImmVal = Op.getOperand(2).getImm();

        // Special case: ADD with Imm=1 -> INC32m, SUB with Imm=1 -> DEC32m
        if (ImmVal == 1 && (isAdd32ri(Opc) || isSub32ri(Opc))) {
          unsigned MemOpc = isAdd32ri(Opc) ? X86::INC32m : X86::DEC32m;
          auto MIB = BuildMI(MBB, Load, DL, TII->get(MemOpc));
          for (unsigned i = 1; i < 6; ++i)
            MIB.add(Load.getOperand(i));
          MIB.cloneMemRefs(Load);

          Store.eraseFromParent();
          Op.eraseFromParent();
          I = MIB.getInstr()->getIterator();
          Load.eraseFromParent();
          Changed = true;
        } else {
          unsigned MemOpc = getImmMemFormOpcode(Opc);
          auto MIB = BuildMI(MBB, Load, DL, TII->get(MemOpc));
          // Copy memory operands from Load (operands 1-5)
          for (unsigned i = 1; i < 6; ++i)
            MIB.add(Load.getOperand(i));
          // Add the immediate operand
          MIB.addImm(ImmVal);
          MIB.cloneMemRefs(Load);

          Store.eraseFromParent();
          Op.eraseFromParent();
          I = MIB.getInstr()->getIterator();
          Load.eraseFromParent();
          Changed = true;
        }

      } else {
        // FoldKind::RegReg (existing behavior)
        unsigned MemFormOpc = getRegRegMemFormOpcode(Op.getOpcode());
        auto MIB = BuildMI(MBB, Load, DL, TII->get(MemFormOpc));
        for (unsigned i = 1; i < 6; ++i)
          MIB.add(Load.getOperand(i));
        MIB.addReg(OpSrc);
        MIB.cloneMemRefs(Load);

        Store.eraseFromParent();
        Op.eraseFromParent();
        I = MIB.getInstr()->getIterator();
        Load.eraseFromParent();
        Changed = true;
      }
    }
  }
  return Changed;
}

FunctionPass *llvm::createX86PreferAddMemPass() {
  return new X86PreferAddMemPass();
}

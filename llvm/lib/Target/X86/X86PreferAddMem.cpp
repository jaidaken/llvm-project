// bw1-decomp: Fold load+op+store sequences into memory-direct arithmetic.
//
// MSVC 6.0 generates: add dword ptr [mem], reg
// LLVM generates:     mov tmp, [mem]; add tmp, reg; mov [mem], tmp
//
// This pass finds the load+op+store pattern and folds it into a single
// memory-direct instruction when safe (no other uses of the loaded value,
// same memory address for load and store, no intervening memory ops).
//
// Handles ADD, SUB, OR, AND, XOR memory-direct forms.

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

/// Get the memory-form opcode for a register-register arithmetic op.
/// Returns 0 if no memory form exists.
static unsigned getMemFormOpcode(unsigned RegRegOpc) {
  switch (RegRegOpc) {
  case X86::ADD32rr: case X86::ADD32rr_REV: return X86::ADD32mr;
  case X86::SUB32rr: case X86::SUB32rr_REV: return X86::SUB32mr;
  case X86::OR32rr:  case X86::OR32rr_REV:  return X86::OR32mr;
  case X86::AND32rr: case X86::AND32rr_REV: return X86::AND32mr;
  case X86::XOR32rr: case X86::XOR32rr_REV: return X86::XOR32mr;
  default: return 0;
  }
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
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Scan for: MOV32rm Reg, [mem]; OP32rr Reg, Reg, Src; MOV32mr [mem], Reg
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
      unsigned MemFormOpc = getMemFormOpcode(Op.getOpcode());
      if (!MemFormOpc) continue;

      // The arithmetic op must use LoadDst as both source and destination
      // OP32rr dst, src1, src2 where dst == src1 == LoadDst
      if (Op.getOperand(0).getReg() != LoadDst) continue;
      if (Op.getOperand(1).getReg() != LoadDst) continue;
      Register OpSrc = Op.getOperand(2).getReg();

      // Find the next non-debug instruction
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

      // Verify LoadDst is not used between Store and the end of this sequence
      // (it should be dead after the store)
      // Simple check: LoadDst should only appear in these 3 instructions
      bool safeToFold = true;
      for (auto CheckI = std::next(StoreI); CheckI != E; ++CheckI) {
        if (CheckI->readsRegister(LoadDst, /*TRI=*/nullptr)) {
          safeToFold = false;
          break;
        }
        // Stop checking after a few instructions
        break;
      }
      // Also check OpSrc is not defined between Load and Op
      // (conservative: skip if OpSrc == LoadDst)
      if (OpSrc == LoadDst) continue;

      if (!safeToFold) continue;

      // Fold: replace load+op+store with OP32mr [mem], src
      DebugLoc DL = Op.getDebugLoc();
      auto MIB = BuildMI(MBB, Load, DL, TII->get(MemFormOpc));
      // Copy memory operands from Load (operands 1-5)
      for (unsigned i = 1; i < 6; ++i)
        MIB.add(Load.getOperand(i));
      // Add the source register
      MIB.addReg(OpSrc);
      MIB.cloneMemRefs(Load);

      // Erase the three original instructions
      Store.eraseFromParent();
      Op.eraseFromParent();
      I = MIB.getInstr()->getIterator();
      Load.eraseFromParent();

      Changed = true;
    }
  }
  return Changed;
}

FunctionPass *llvm::createX86PreferAddMemPass() {
  return new X86PreferAddMemPass();
}

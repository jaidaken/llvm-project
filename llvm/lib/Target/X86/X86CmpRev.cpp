// bw1-decomp: Convert CMP32mr (opcode 0x39) to CMP32rm (opcode 0x3B) for
// functions with the "cmp_rev" attribute.
//
// MSVC 6.0 uses CMP32rm (cmp reg, [mem]) while Clang generates CMP32mr
// (cmp [mem], reg). The comparison operands are swapped, so the condition
// code on any following JCC/SETcc/CMOVcc must be flipped.
//
// CMP32mr operands: [base, scale, index, disp, segment, src_reg]
// CMP32rm operands: [src_reg, base, scale, index, disp, segment]

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define DEBUG_TYPE "x86-cmp-rev"

namespace {

/// Swap condition code for reversed CMP operands.
/// When CMP operands are swapped (cmp [mem],reg -> cmp reg,[mem]),
/// the comparison direction reverses:
///   E/NE stay the same (equality is symmetric)
///   L <-> G, LE <-> GE (signed)
///   B <-> A, BE <-> AE (unsigned)
static X86::CondCode swapConditionCode(X86::CondCode CC) {
  switch (CC) {
  case X86::COND_E:  return X86::COND_E;
  case X86::COND_NE: return X86::COND_NE;
  case X86::COND_L:  return X86::COND_G;
  case X86::COND_LE: return X86::COND_GE;
  case X86::COND_G:  return X86::COND_L;
  case X86::COND_GE: return X86::COND_LE;
  case X86::COND_B:  return X86::COND_A;
  case X86::COND_BE: return X86::COND_AE;
  case X86::COND_A:  return X86::COND_B;
  case X86::COND_AE: return X86::COND_BE;
  // S, NS, P, NP, O, NO don't depend on operand order for CMP
  case X86::COND_S:  return X86::COND_S;
  case X86::COND_NS: return X86::COND_NS;
  case X86::COND_P:  return X86::COND_P;
  case X86::COND_NP: return X86::COND_NP;
  case X86::COND_O:  return X86::COND_O;
  case X86::COND_NO: return X86::COND_NO;
  default: return X86::COND_INVALID;
  }
}

/// Map CMP mr opcodes to their rm equivalents.
static unsigned getCmpRmForCmpMr(unsigned Opc) {
  switch (Opc) {
  case X86::CMP8mr:  return X86::CMP8rm;
  case X86::CMP16mr: return X86::CMP16rm;
  case X86::CMP32mr: return X86::CMP32rm;
  case X86::CMP64mr: return X86::CMP64rm;
  default: return 0;
  }
}

/// Check if an instruction reads EFLAGS with a condition code that needs
/// swapping. Returns the operand index of the condition code, or -1.
static int getCondOperandIdx(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  if (Opc == X86::JCC_1 || Opc == X86::JCC_4)
    return 1; // operand[0]=MBB, operand[1]=CC
  if (Opc == X86::SETCCr || Opc == X86::SETCCm)
    return MI.getDesc().getNumDefs(); // CC is first use operand after defs
  // CMOVcc: condition is the last operand
  if (Opc == X86::CMOV32rr || Opc == X86::CMOV16rr || Opc == X86::CMOV64rr)
    return MI.getNumOperands() - 1;
  return -1;
}

/// Check if an instruction defines EFLAGS (writes to flags).
static bool definesEFLAGS(const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isReg() && MO.isDef() && MO.getReg() == X86::EFLAGS)
      return true;
  }
  return false;
}

class X86CmpRevPass : public MachineFunctionPass {
public:
  static char ID;
  X86CmpRevPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 CMP mr->rm reversal for MSVC 6.0";
  }
};
} // end anonymous namespace

char X86CmpRevPass::ID = 0;

bool X86CmpRevPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("cmp_rev"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E; ) {
      MachineInstr &MI = *I;
      unsigned RmOpc = getCmpRmForCmpMr(MI.getOpcode());
      if (!RmOpc) {
        ++I;
        continue;
      }

      // CMP mr operands: [base(0), scale(1), index(2), disp(3), segment(4), src_reg(5)]
      // We need at least 6 operands.
      if (MI.getNumOperands() < 6) {
        ++I;
        continue;
      }

      // Extract the register operand (last operand of CMP mr).
      Register SrcReg = MI.getOperand(5).getReg();

      // Build the new CMP rm instruction:
      // CMP rm operands: [src_reg, base, scale, index, disp, segment]
      DebugLoc DL = MI.getDebugLoc();
      auto MIB = BuildMI(MBB, MI, DL, TII->get(RmOpc))
          .addReg(SrcReg);
      // Copy the 5 memory operands from the original CMP mr.
      for (unsigned i = 0; i < 5; ++i)
        MIB.add(MI.getOperand(i));
      MIB.cloneMemRefs(MI);

      // Flip condition codes on all following instructions that read EFLAGS,
      // until we hit an instruction that defines EFLAGS (a new comparison).
      auto Next = std::next(I);
      while (Next != E) {
        MachineInstr &NextMI = *Next;
        if (NextMI.isDebugInstr()) {
          ++Next;
          continue;
        }
        // Stop if this instruction defines EFLAGS (new comparison/test).
        if (definesEFLAGS(NextMI))
          break;
        int CondIdx = getCondOperandIdx(NextMI);
        if (CondIdx >= 0) {
          auto OldCC = static_cast<X86::CondCode>(
              NextMI.getOperand(CondIdx).getImm());
          X86::CondCode NewCC = swapConditionCode(OldCC);
          if (NewCC != X86::COND_INVALID)
            NextMI.getOperand(CondIdx).setImm(NewCC);
        }
        ++Next;
      }

      // Remove original CMP mr and advance iterator.
      auto NextI = std::next(I);
      MI.eraseFromParent();
      I = NextI;
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86CmpRevPass() {
  return new X86CmpRevPass();
}

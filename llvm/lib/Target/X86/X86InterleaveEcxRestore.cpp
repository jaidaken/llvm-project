// bw1-decomp: Hoist MOV ECX, ESI from block starts to before the Jcc in the
// predecessor block, for functions with the "interleave_ecx_restore" attribute.
//
// MSVC 6.0 interleaves non-flag-affecting instructions between test/cmp and
// the conditional branch as a pipeline fill optimization:
//
//   test eax, eax
//   mov ecx, esi        ; interleaved between test and branch
//   je LAB_skip
//
// Clang keeps the MOV inside the branch target:
//
//   test eax, eax
//   je LAB_skip
//   mov ecx, esi        ; inside the fall-through block
//
// This pass finds MOV32rr ECX, ESI at the start of a basic block whose single
// predecessor ends with a flags-setting instruction followed by a conditional
// branch (Jcc), and hoists the MOV to just before the Jcc. This is safe
// because MOV32rr does not affect EFLAGS.

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define DEBUG_TYPE "x86-interleave-ecx-restore"

namespace {

class X86InterleaveEcxRestorePass : public MachineFunctionPass {
public:
  static char ID;
  X86InterleaveEcxRestorePass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 interleave ECX restore before Jcc (MSVC 6.0)";
  }
};

} // end anonymous namespace

char X86InterleaveEcxRestorePass::ID = 0;

/// Check if MI is MOV32rr ECX, ESI.
static bool isMovEcxEsi(const MachineInstr &MI) {
  if (MI.getOpcode() != X86::MOV32rr)
    return false;
  if (MI.getNumOperands() < 2)
    return false;
  return MI.getOperand(0).isReg() && MI.getOperand(0).getReg() == X86::ECX &&
         MI.getOperand(1).isReg() && MI.getOperand(1).getReg() == X86::ESI;
}

/// Check if MI defines EFLAGS.
static bool definesEFLAGS(const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isReg() && MO.isDef() && MO.getReg() == X86::EFLAGS)
      return true;
  }
  return false;
}

bool X86InterleaveEcxRestorePass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("interleave_ecx_restore"))
    return false;

  const X86InstrInfo *TII =
      MF.getSubtarget<X86Subtarget>().getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Find the first non-PHI, non-debug instruction.
    MachineBasicBlock::iterator FirstReal = MBB.begin();
    while (FirstReal != MBB.end() &&
           (FirstReal->isPHI() || FirstReal->isDebugInstr()))
      ++FirstReal;

    if (FirstReal == MBB.end())
      continue;

    // Must be MOV32rr ECX, ESI.
    if (!isMovEcxEsi(*FirstReal))
      continue;

    // Must have exactly one predecessor.
    if (MBB.pred_size() != 1)
      continue;

    MachineBasicBlock *Pred = *MBB.pred_begin();

    // Find the terminator (Jcc) in the predecessor by scanning backward.
    MachineBasicBlock::iterator TermIt = Pred->end();
    for (auto I = Pred->end(), B = Pred->begin(); I != B;) {
      --I;
      if (I->isDebugInstr())
        continue;
      if (I->getOpcode() == X86::JCC_1) {
        TermIt = I;
        break;
      }
      // First real instruction from end is not Jcc - stop.
      break;
    }

    if (TermIt == Pred->end())
      continue;

    // The instruction before the Jcc must set EFLAGS (TEST, CMP, etc.).
    MachineBasicBlock::iterator FlagSetter = TermIt;
    bool FoundFlagSetter = false;
    for (auto I = TermIt, B = Pred->begin(); I != B;) {
      --I;
      if (I->isDebugInstr())
        continue;
      if (definesEFLAGS(*I)) {
        FlagSetter = I;
        FoundFlagSetter = true;
      }
      break;
    }

    if (!FoundFlagSetter)
      continue;

    // Hoist the MOV from this block to just before the Jcc in the predecessor.
    DebugLoc DL = FirstReal->getDebugLoc();
    BuildMI(*Pred, TermIt, DL, TII->get(X86::MOV32rr), X86::ECX)
        .addReg(X86::ESI);

    // Remove the original MOV from this block.
    FirstReal->eraseFromParent();
    Changed = true;
  }

  return Changed;
}

FunctionPass *llvm::createX86InterleaveEcxRestorePass() {
  return new X86InterleaveEcxRestorePass();
}

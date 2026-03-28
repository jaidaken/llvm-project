// bw1-decomp: Move callee-save PUSH from before TEST/CMP to between TEST/CMP
// and the following Jcc, for functions with "interleave_push_before_branch".
//
// MSVC 6.0 fills pipeline slots between a flag-setting instruction and a
// conditional branch with callee-save pushes:
//
//   test ecx, ecx
//   push edi               ; interleaved between test and branch
//   je skip
//
// Clang puts the push before the test:
//
//   push edi               ; before the test (wrong position)
//   test ecx, ecx
//   je skip
//
// This pass finds PUSH32r; TEST/CMP; JCC_1 sequences within a basic block
// and reorders them to TEST/CMP; PUSH32r; JCC_1. This is safe because
// PUSH32r does not affect EFLAGS.

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "x86-interleave-push-before-branch"

namespace {

class X86InterleavePushBeforeBranchPass : public MachineFunctionPass {
public:
  static char ID;
  X86InterleavePushBeforeBranchPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 interleave PUSH before branch (MSVC 6.0)";
  }

private:
  const X86InstrInfo *TII = nullptr;

  bool processBlock(MachineBasicBlock &MBB);
};

} // end anonymous namespace

char X86InterleavePushBeforeBranchPass::ID = 0;

/// Check if MI is a TEST or CMP instruction that sets EFLAGS.
static bool isTestOrCmp(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case X86::TEST8rr:
  case X86::TEST16rr:
  case X86::TEST32rr:
  case X86::TEST8ri:
  case X86::TEST16ri:
  case X86::TEST32ri:
  case X86::TEST8mi:
  case X86::TEST16mi:
  case X86::TEST32mi:
  case X86::CMP8rr:
  case X86::CMP16rr:
  case X86::CMP32rr:
  case X86::CMP8ri:
  case X86::CMP16ri:
  case X86::CMP32ri:
  case X86::CMP8ri8:
  case X86::CMP16ri8:
  case X86::CMP32ri8:
  case X86::CMP8mi:
  case X86::CMP16mi:
  case X86::CMP32mi:
  case X86::CMP8mi8:
  case X86::CMP16mi8:
  case X86::CMP32mi8:
  case X86::CMP8rm:
  case X86::CMP16rm:
  case X86::CMP32rm:
  case X86::CMP8mr:
  case X86::CMP16mr:
  case X86::CMP32mr:
    return true;
  default:
    return false;
  }
}

/// Check if the PUSH register conflicts with any operand of the TEST/CMP.
/// We must not move a PUSH of a register that is used by the TEST/CMP,
/// because the PUSH modifies ESP and the pushed register is read - but
/// PUSH32r only reads the register, so the only conflict is if the TEST/CMP
/// uses ESP (unlikely but safe to check).
static bool pushConflictsWithTestCmp(const MachineInstr &Push,
                                     const MachineInstr &TestCmp,
                                     const TargetRegisterInfo *TRI) {
  Register PushReg = Push.getOperand(0).getReg();
  // PUSH reads PushReg and writes ESP. Check that TEST/CMP does not
  // read or define ESP or the pushed register in a way that conflicts.
  // Since we are moving PUSH after TEST/CMP:
  // - TEST/CMP must not read ESP (PUSH modifies ESP)
  // - TEST/CMP must not define PushReg (PUSH reads it)
  for (const MachineOperand &MO : TestCmp.operands()) {
    if (!MO.isReg() || MO.getReg() == X86::NoRegister)
      continue;
    // If TEST/CMP reads ESP, moving PUSH (which modifies ESP) before it
    // would be wrong.  But we are moving PUSH *after* TEST/CMP, so the
    // concern is the opposite: if TEST/CMP reads ESP in memory operands,
    // the original code had PUSH before TEST/CMP so ESP was already modified.
    // Moving PUSH after means ESP would be different.  Bail out.
    if (TRI->regsOverlap(MO.getReg(), X86::ESP))
      return true;
    // If TEST/CMP defines PushReg, moving PUSH after would read the wrong
    // value.  TEST/CMP normally only defines EFLAGS, but be safe.
    if (MO.isDef() && TRI->regsOverlap(MO.getReg(), PushReg))
      return true;
  }
  return false;
}

bool X86InterleavePushBeforeBranchPass::processBlock(MachineBasicBlock &MBB) {
  bool Changed = false;

  for (auto I = MBB.begin(), E = MBB.end(); I != E;) {
    MachineInstr &MI = *I;
    ++I; // advance early since we may splice

    // Look for a JCC_1 instruction.
    if (MI.getOpcode() != X86::JCC_1)
      continue;

    // Walk backward past debug instructions to find the TEST/CMP.
    auto TestIt = MachineBasicBlock::iterator(MI);
    if (TestIt == MBB.begin())
      continue;
    --TestIt;
    while (TestIt != MBB.begin() &&
           (TestIt->isPseudo() || TestIt->isDebugInstr()))
      --TestIt;

    if (!isTestOrCmp(*TestIt))
      continue;

    // Walk backward past debug instructions to find the PUSH.
    auto PushIt = TestIt;
    if (PushIt == MBB.begin())
      continue;
    --PushIt;
    while (PushIt != MBB.begin() &&
           (PushIt->isPseudo() || PushIt->isDebugInstr()))
      --PushIt;

    if (PushIt->getOpcode() != X86::PUSH32r)
      continue;

    // Check for conflicts.
    const TargetRegisterInfo *TRI = &TII->getRegisterInfo();
    if (pushConflictsWithTestCmp(*PushIt, *TestIt, TRI))
      continue;

    LLVM_DEBUG(dbgs() << "InterleavePushBeforeBranch: moving PUSH "
                      << printReg(PushIt->getOperand(0).getReg(), TRI)
                      << " after " << TII->getName(TestIt->getOpcode())
                      << " in " << MBB.getParent()->getName() << "\n");

    // Move the PUSH from before TEST/CMP to between TEST/CMP and JCC.
    // splice(insertBefore, fromMBB, fromIt, toIt)
    MBB.splice(MachineBasicBlock::iterator(MI), &MBB,
               PushIt, std::next(PushIt));
    Changed = true;
  }

  return Changed;
}

bool X86InterleavePushBeforeBranchPass::runOnMachineFunction(
    MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("interleave_push_before_branch"))
    return false;

  TII = MF.getSubtarget<X86Subtarget>().getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF)
    Changed |= processBlock(MBB);

  return Changed;
}

FunctionPass *llvm::createX86InterleavePushBeforeBranchPass() {
  return new X86InterleavePushBeforeBranchPass();
}

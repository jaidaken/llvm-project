// bw1-decomp: Sink MOV32ri EAX, imm past a TEST/CMP + Jcc sequence when
// the MOV result is only used after the branch.
//
// Clang preloads return values before the branch test:
//   mov eax, 1           ; preload return value
//   test ecx, ecx        ; test condition
//   je LAB_return_0
//
// MSVC 6.0 tests first, then loads the return value:
//   test eax, eax        ; test condition
//   je LAB_return_0
//   mov eax, 1           ; load return value only in the taken path
//
// This pass, gated on the "defer_return_value" string attribute, moves the
// MOV32ri EAX instruction from before the TEST/CMP to the start of the
// fall-through block (after the Jcc). Only applies when the TEST/CMP does
// not read or write EAX.

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define DEBUG_TYPE "x86-defer-return-value"
#define PASS_NAME "X86 defer return value past branch"

namespace {
class X86DeferReturnValuePass : public MachineFunctionPass {
public:
  static char ID;
  X86DeferReturnValuePass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return PASS_NAME; }

private:
  /// Check if an instruction reads or defines EAX (or any sub/super register).
  static bool usesOrDefsEAX(const MachineInstr &MI,
                            const TargetRegisterInfo *TRI);

  /// Check if an instruction is a TEST or CMP that does not use EAX.
  static bool isTestOrCmpNotUsingEAX(const MachineInstr &MI,
                                     const TargetRegisterInfo *TRI);

  /// Check if an instruction is a conditional jump (JCC_1 or JCC_4).
  static bool isJcc(const MachineInstr &MI);
};
} // end anonymous namespace

char X86DeferReturnValuePass::ID = 0;

bool X86DeferReturnValuePass::usesOrDefsEAX(const MachineInstr &MI,
                                            const TargetRegisterInfo *TRI) {
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || MO.getReg() == 0)
      continue;
    if (TRI->regsOverlap(MO.getReg(), X86::EAX))
      return true;
  }
  return false;
}

bool X86DeferReturnValuePass::isTestOrCmpNotUsingEAX(
    const MachineInstr &MI, const TargetRegisterInfo *TRI) {
  unsigned Opc = MI.getOpcode();
  switch (Opc) {
  case X86::TEST8rr:
  case X86::TEST16rr:
  case X86::TEST32rr:
  case X86::TEST64rr:
  case X86::TEST8ri:
  case X86::TEST16ri:
  case X86::TEST32ri:
  case X86::TEST64ri32:
  case X86::TEST8mi:
  case X86::TEST16mi:
  case X86::TEST32mi:
  case X86::TEST64mi32:
  case X86::CMP8rr:
  case X86::CMP16rr:
  case X86::CMP32rr:
  case X86::CMP64rr:
  case X86::CMP8ri:
  case X86::CMP16ri:
  case X86::CMP32ri:
  case X86::CMP32ri8:
  case X86::CMP64ri8:
  case X86::CMP64ri32:
  case X86::CMP8rm:
  case X86::CMP16rm:
  case X86::CMP32rm:
  case X86::CMP64rm:
  case X86::CMP8mr:
  case X86::CMP16mr:
  case X86::CMP32mr:
  case X86::CMP64mr:
  case X86::CMP8mi:
  case X86::CMP16mi:
  case X86::CMP32mi:
  case X86::CMP32mi8:
  case X86::CMP64mi8:
  case X86::CMP64mi32:
    break;
  default:
    return false;
  }
  return !usesOrDefsEAX(MI, TRI);
}

bool X86DeferReturnValuePass::isJcc(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  return Opc == X86::JCC_1 || Opc == X86::JCC_4;
}

bool X86DeferReturnValuePass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("defer_return_value"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E;) {
      MachineInstr &MI = *I;

      // Look for MOV32ri EAX, imm.
      if (MI.getOpcode() != X86::MOV32ri ||
          MI.getOperand(0).getReg() != X86::EAX) {
        ++I;
        continue;
      }

      // Next non-debug instruction must be a TEST/CMP that doesn't use EAX.
      auto TestIt = std::next(I);
      while (TestIt != E && TestIt->isDebugInstr())
        ++TestIt;
      if (TestIt == E) {
        ++I;
        continue;
      }

      if (!isTestOrCmpNotUsingEAX(*TestIt, TRI)) {
        ++I;
        continue;
      }

      // Next non-debug instruction after TEST/CMP must be a Jcc.
      auto JccIt = std::next(TestIt);
      while (JccIt != E && JccIt->isDebugInstr())
        ++JccIt;
      if (JccIt == E) {
        ++I;
        continue;
      }

      if (!isJcc(*JccIt)) {
        ++I;
        continue;
      }

      // The Jcc should be the last instruction in the block (terminators
      // are at the end). The fall-through block is the layout successor.
      MachineBasicBlock *FallThrough = MBB.getFallThrough();
      if (!FallThrough) {
        ++I;
        continue;
      }

      // Verify EAX is not live-in to the branch target. The Jcc target is
      // operand 0 of the JCC instruction.
      MachineBasicBlock *BranchTarget = JccIt->getOperand(0).getMBB();
      if (BranchTarget) {
        // Check if EAX (or any overlapping reg) is live-in to the branch
        // target. If so, the branch target expects EAX to hold the value we
        // set, so we cannot sink past the branch.
        bool EAXLiveInTarget = false;
        for (const auto &LI : BranchTarget->liveins()) {
          if (TRI->regsOverlap(LI.PhysReg, X86::EAX)) {
            EAXLiveInTarget = true;
            break;
          }
        }
        if (EAXLiveInTarget) {
          ++I;
          continue;
        }
      }

      // All checks passed. Move the MOV32ri from before TEST to the
      // beginning of the fall-through block.
      int64_t ImmVal = MI.getOperand(1).getImm();
      DebugLoc DL = MI.getDebugLoc();

      // Insert at the beginning of the fall-through block.
      BuildMI(*FallThrough, FallThrough->begin(), DL,
              TII->get(X86::MOV32ri), X86::EAX)
          .addImm(ImmVal);

      // Remove the original MOV and advance the iterator.
      auto NextI = std::next(I);
      MI.eraseFromParent();
      I = NextI;
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86DeferReturnValuePass() {
  return new X86DeferReturnValuePass();
}

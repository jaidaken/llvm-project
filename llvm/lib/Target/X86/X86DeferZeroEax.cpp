// bw1-decomp: Sink XOR32rr EAX, EAX from early in the function to just before
// the first use of EAX (typically a SETCCr or a branch returning 0).
//
// Clang pre-zeroes EAX early to prepare for a later sete/setcc:
//   xor eax, eax         ; pre-zero early
//   ... other instructions ...
//   test ecx, ecx
//   sete al              ; uses the pre-zeroed high bytes
//
// MSVC 6.0 places the xor eax, eax immediately before where it's needed:
//   ... other instructions ...
//   xor eax, eax         ; zero right before use
//   test ecx, ecx
//   sete al
//
// This pass, gated on the "defer_zero_eax" string attribute, moves the
// XOR32rr EAX, EAX instruction from its early position to just before the
// first instruction that reads EAX. The XOR sets EFLAGS, so it cannot be
// moved past instructions that read EFLAGS, or past EFLAGS-setting
// instructions whose flags are consumed later.

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define DEBUG_TYPE "x86-defer-zero-eax"
#define PASS_NAME "X86 defer zero EAX to before first use"

namespace {
class X86DeferZeroEaxPass : public MachineFunctionPass {
public:
  static char ID;
  X86DeferZeroEaxPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return PASS_NAME; }

private:
  /// Return true if MI is a XOR32rr or XOR32rr_REV self-xor zeroing EAX.
  static bool isXorZeroEax(const MachineInstr &MI);

  /// Return true if MI reads EAX (or any sub/super register of EAX).
  static bool readsEAX(const MachineInstr &MI, const TargetRegisterInfo *TRI);

  /// Return true if MI reads EFLAGS.
  static bool readsEFLAGS(const MachineInstr &MI);

  /// Return true if MI defines EFLAGS.
  static bool defsEFLAGS(const MachineInstr &MI);
};
} // end anonymous namespace

char X86DeferZeroEaxPass::ID = 0;

bool X86DeferZeroEaxPass::isXorZeroEax(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  if (Opc != X86::XOR32rr && Opc != X86::XOR32rr_REV)
    return false;
  // Self-xor: both source operands are the same register.
  if (MI.getOperand(1).getReg() != MI.getOperand(2).getReg())
    return false;
  // Destination must be EAX.
  return MI.getOperand(0).getReg() == X86::EAX;
}

bool X86DeferZeroEaxPass::readsEAX(const MachineInstr &MI,
                                    const TargetRegisterInfo *TRI) {
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || MO.getReg() == 0)
      continue;
    if (!MO.isUse())
      continue;
    if (TRI->regsOverlap(MO.getReg(), X86::EAX))
      return true;
  }
  return false;
}

bool X86DeferZeroEaxPass::readsEFLAGS(const MachineInstr &MI) {
  return MI.readsRegister(X86::EFLAGS, /*TRI=*/nullptr);
}

bool X86DeferZeroEaxPass::defsEFLAGS(const MachineInstr &MI) {
  return MI.definesRegister(X86::EFLAGS, /*TRI=*/nullptr);
}

bool X86DeferZeroEaxPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("defer_zero_eax"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E;) {
      MachineInstr &XorMI = *I;

      if (!isXorZeroEax(XorMI)) {
        ++I;
        continue;
      }

      // Scan forward from the XOR to find the first instruction that reads EAX.
      // We need to track EFLAGS liveness: the XOR sets EFLAGS, so we cannot
      // move it past an instruction that reads EFLAGS (it would change the
      // flags that instruction sees). We also need to handle EFLAGS-defining
      // instructions carefully: if an instruction defines EFLAGS and a later
      // instruction reads EFLAGS, we cannot insert the XOR between them.
      //
      // Strategy: scan forward and find the first EAX reader. Verify that
      // we can safely place the XOR just before it by checking that no
      // instruction between the proposed insertion point and the XOR reads
      // EFLAGS (since the XOR would clobber them).
      auto ScanIt = std::next(I);
      bool CanSink = true;
      bool FoundUse = false;
      MachineBasicBlock::iterator InsertPoint;

      // Track whether the EFLAGS from the XOR are still "live" - if any
      // instruction between the XOR and our insertion point reads EFLAGS,
      // we would break things by moving the XOR later. But actually, the
      // XOR's EFLAGS are generally dead (nobody reads them intentionally).
      // The concern is the opposite: if we place the XOR just before the
      // use point, the XOR will clobber EFLAGS. If any instruction between
      // the new position and the next EFLAGS-definer reads EFLAGS, we'd
      // break that chain.
      //
      // So we need: at the insertion point, EFLAGS must either be dead or
      // be about to be defined before next use. The simplest safe rule:
      // the instruction immediately after the insertion point must not
      // read EFLAGS, OR the instruction before the insertion point must
      // define EFLAGS.
      //
      // Actually, the simplest correct approach: find the first EAX reader.
      // Then check that inserting the XOR just before it is safe by verifying
      // that no instruction after the insertion point (up to the next EFLAGS
      // definer) reads the old EFLAGS.
      //
      // Simpler formulation: the XOR can be placed at position P if:
      // 1. P is before the first EAX reader
      // 2. At position P, EFLAGS are "dead" - either no instruction from P
      //    onward reads EFLAGS before the next EFLAGS definer, OR P is right
      //    after an EFLAGS definer.

      // First pass: find the first EAX reader and check for blockers.
      bool EflagsLiveFromXor = false;
      for (auto J = ScanIt; J != E; ++J) {
        if (J->isDebugInstr())
          continue;

        // If this instruction reads EFLAGS, the XOR's EFLAGS output may
        // be consumed. Mark that EFLAGS from the XOR were used.
        if (readsEFLAGS(*J))
          EflagsLiveFromXor = true;

        if (readsEAX(*J, TRI)) {
          FoundUse = true;
          InsertPoint = J;
          break;
        }

        // If this instruction defines EAX (but doesn't read it), the XOR
        // is dead - EAX gets overwritten before any read.
        for (const MachineOperand &MO : J->operands()) {
          if (MO.isReg() && MO.isDef() && MO.getReg() != 0 &&
              TRI->regsOverlap(MO.getReg(), X86::EAX)) {
            // EAX is overwritten before any read - the XOR is dead, but
            // we don't remove it (that could change semantics if EFLAGS
            // from the XOR were read). Just bail out.
            CanSink = false;
            break;
          }
        }
        if (!CanSink)
          break;
      }

      if (!FoundUse || !CanSink) {
        ++I;
        continue;
      }

      // If the XOR is already immediately before the use (no instructions
      // between them except debug instrs), no need to move.
      auto CheckAdj = std::next(I);
      while (CheckAdj != E && CheckAdj->isDebugInstr())
        ++CheckAdj;
      if (CheckAdj == InsertPoint) {
        ++I;
        continue;
      }

      // Now verify that we can insert the XOR just before InsertPoint
      // without breaking EFLAGS liveness. Walk backward from InsertPoint
      // to find the state of EFLAGS at the insertion point.
      //
      // If the instruction just before InsertPoint defines EFLAGS, then
      // placing the XOR there would clobber those flags before InsertPoint
      // can read them. So check: does InsertPoint read EFLAGS? If so, we
      // need to place the XOR *before* the EFLAGS-defining chain.
      //
      // The typical pattern we want to handle:
      //   xor eax, eax       ; early
      //   ... stuff ...
      //   test ecx, ecx      ; defines EFLAGS
      //   sete al            ; reads EFLAGS, reads EAX (implicit via AL)
      //
      // We want to produce:
      //   ... stuff ...
      //   xor eax, eax       ; moved here
      //   test ecx, ecx      ; defines EFLAGS
      //   sete al            ; reads EFLAGS, reads EAX

      // If InsertPoint reads EFLAGS, we need to find the EFLAGS-defining
      // chain and insert before it.
      if (readsEFLAGS(*InsertPoint)) {
        // Walk backward from InsertPoint to find the start of the
        // EFLAGS-defining chain. The chain is a sequence of instructions
        // where the last one before InsertPoint defines EFLAGS.
        auto ChainStart = InsertPoint;
        auto Prev = InsertPoint;
        while (Prev != ScanIt) {
          --Prev;
          if (Prev->isDebugInstr())
            continue;
          if (defsEFLAGS(*Prev)) {
            ChainStart = Prev;
            // Keep walking back - there might be more in the chain.
            // But stop if this instruction also reads EFLAGS (it's the
            // start of a new chain).
            if (readsEFLAGS(*Prev)) {
              // This instruction both reads and defines EFLAGS. Need to
              // find who defines the flags it reads.
              continue;
            }
            // This instruction only defines EFLAGS. Check if the
            // instruction before it reads EFLAGS (meaning this is part
            // of a chain). If not, this is the start.
            break;
          }
          // If this instruction reads EAX, we can't insert before it
          // either - bail out entirely.
          if (readsEAX(*Prev, TRI)) {
            CanSink = false;
            break;
          }
          // This instruction doesn't touch EFLAGS. The chain starts at
          // the next EFLAGS definer we found (or InsertPoint).
          break;
        }

        if (!CanSink) {
          ++I;
          continue;
        }

        InsertPoint = ChainStart;

        // Re-check adjacency after adjusting InsertPoint.
        CheckAdj = std::next(I);
        while (CheckAdj != E && CheckAdj->isDebugInstr())
          ++CheckAdj;
        if (CheckAdj == InsertPoint) {
          ++I;
          continue;
        }
      }

      // Final safety check: verify that at the insertion point, EFLAGS
      // are not live (i.e., no instruction from InsertPoint backward to
      // the XOR reads EFLAGS that would be clobbered). We already know
      // InsertPoint doesn't read EFLAGS (we adjusted above if it did).
      // But we need to ensure the XOR's EFLAGS clobber doesn't matter.
      // Check: does any instruction between the new InsertPoint and the
      // next EFLAGS definer (going forward) read EFLAGS? The XOR IS the
      // new EFLAGS definer, so we need to check if anything between the
      // insert position and the original InsertPoint reads EFLAGS.
      // But we've already adjusted InsertPoint to be before any EFLAGS
      // chain, so the instruction at InsertPoint should define EFLAGS
      // (in the test+sete case) or not read EFLAGS.
      //
      // Just verify InsertPoint itself doesn't read EFLAGS.
      if (readsEFLAGS(*InsertPoint)) {
        // This shouldn't happen after the adjustment above, but be safe.
        ++I;
        continue;
      }

      // All checks passed. Build a new XOR at the insertion point and
      // remove the original.
      unsigned XorOpcode = XorMI.getOpcode();
      DebugLoc DL = XorMI.getDebugLoc();

      BuildMI(MBB, InsertPoint, DL, TII->get(XorOpcode), X86::EAX)
          .addReg(X86::EAX, RegState::Undef)
          .addReg(X86::EAX, RegState::Undef);

      auto NextI = std::next(I);
      XorMI.eraseFromParent();
      I = NextI;
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86DeferZeroEaxPass() {
  return new X86DeferZeroEaxPass();
}

// bw1-decomp: Fold SHL + LEA [reg+global] into LEA [reg*scale+global].
//
// MSVC 6.0 folds multiply-by-constant and global address displacement into a
// two-LEA chain using scaled-index addressing:
//
//   lea ecx, [eax+eax*2]          ; ecx = eax * 3
//   lea edx, [ecx*8+0x00cc63e0]   ; edx = eax * 24 + global_addr
//
// Clang emits SHL + base-register LEA instead:
//
//   lea ecx, [eax+eax*2]          ; ecx = eax * 3
//   shl ecx, 3                    ; ecx = eax * 24
//   lea edx, [ecx+0x00cc63e0]     ; edx = eax * 24 + global_addr
//
// The encoding difference: MSVC uses [index*scale+disp32] (no base register,
// scaled index with global displacement) while Clang uses [base+disp32].
// These produce different ModR/M and SIB bytes.
//
// This pass finds SHL reg, N (N=1,2,3) followed by LEA dst, [reg+global]
// and converts to LEA dst, [reg*scale+global] where scale = 2^N. The SHL
// is then deleted.
//
// Gate: function attribute "lea_chain_global_fold".

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "x86-lea-chain-global-fold"

namespace {
class X86LeaChainGlobalFoldPass : public MachineFunctionPass {
public:
  static char ID;
  X86LeaChainGlobalFoldPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 fold SHL + LEA [reg+global] into LEA [reg*scale+global]";
  }
};
} // end anonymous namespace

char X86LeaChainGlobalFoldPass::ID = 0;

bool X86LeaChainGlobalFoldPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("lea_chain_global_fold"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E;) {
      MachineInstr &ShlMI = *I;

      // Match SHL32ri with shift amount 1, 2, or 3.
      if (ShlMI.getOpcode() != X86::SHL32ri) {
        ++I;
        continue;
      }

      unsigned ShiftAmt = ShlMI.getOperand(2).getImm();
      if (ShiftAmt < 1 || ShiftAmt > 3) {
        ++I;
        continue;
      }

      Register ShlDst = ShlMI.getOperand(0).getReg();
      Register ShlSrc = ShlMI.getOperand(1).getReg();

      // SHL32ri is in-place: dst must equal src.
      if (ShlDst != ShlSrc) {
        ++I;
        continue;
      }

      // Find the next non-debug instruction.
      auto Next = std::next(I);
      while (Next != E && Next->isDebugInstr())
        ++Next;
      if (Next == E) {
        ++I;
        continue;
      }

      MachineInstr &LeaMI = *Next;

      // Match LEA32r where the base register is the SHL destination,
      // scale is 1, index register is 0 (no index), and displacement
      // is a global address.
      if (LeaMI.getOpcode() != X86::LEA32r) {
        ++I;
        continue;
      }

      // LEA32r operands: [0]=dst, [1]=base, [2]=scale, [3]=index, [4]=disp, [5]=segment
      Register LeaDst = LeaMI.getOperand(0).getReg();
      Register LeaBase = LeaMI.getOperand(1).getReg();
      unsigned LeaScale = LeaMI.getOperand(2).getImm();
      Register LeaIndex = LeaMI.getOperand(3).getReg();
      const MachineOperand &LeaDisp = LeaMI.getOperand(4);
      Register LeaSeg = LeaMI.getOperand(5).getReg();

      // The LEA must use ShlDst as base, have scale=1, no index register,
      // and a global address as displacement.
      if (LeaBase != ShlDst || LeaScale != 1 || LeaIndex != X86::NoRegister ||
          !LeaDisp.isGlobal()) {
        ++I;
        continue;
      }

      // Build the replacement: LEA dst, [0 + ShlDst * (1 << ShiftAmt) + global]
      // This is: base=0, scale=2^ShiftAmt, index=ShlDst, disp=global, seg=0
      unsigned NewScale = 1u << ShiftAmt;
      DebugLoc DL = LeaMI.getDebugLoc();

      BuildMI(MBB, LeaMI, DL, TII->get(X86::LEA32r), LeaDst)
          .addReg(0)                                              // base = none
          .addImm(NewScale)                                       // scale
          .addReg(ShlDst)                                         // index
          .addGlobalAddress(LeaDisp.getGlobal(),
                            LeaDisp.getOffset(),
                            LeaDisp.getTargetFlags())             // disp
          .addReg(LeaSeg);                                        // segment

      // Remove the original LEA and SHL.
      auto NextNext = std::next(Next);
      LeaMI.eraseFromParent();
      ShlMI.eraseFromParent();
      I = NextNext;
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86LeaChainGlobalFoldPass() {
  return new X86LeaChainGlobalFoldPass();
}

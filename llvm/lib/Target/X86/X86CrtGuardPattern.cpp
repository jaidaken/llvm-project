// bw1-decomp: Convert TEST8ri to MOV8ri + TEST8rr for CRT guard patterns.
//
// MSVC 6.0 CRT static initializer guards use a specific instruction sequence:
//   mov cl, byte ptr [guard]
//   mov al, 0x01
//   test al, cl
//   jne skip
//   or.s cl, al
//   mov byte ptr [guard], cl
//
// The compiler produces TEST8ri (test reg, 1) which encodes the immediate
// directly. This pass expands it into the two-instruction sequence:
//   MOV8ri AL, 1     (materialize constant into AL)
//   TEST8rr AL, reg  (register-register test)
//
// The existing TestRev pass then swaps operands to match the MSVC 6.0 ModR/M
// encoding order, and the ReversedOps pass handles the OR8rr_REV encoding.
//
// Gated by the crt_guard_pattern function attribute.

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-crt-guard-pattern"

namespace {
class X86CrtGuardPatternPass : public MachineFunctionPass {
public:
  static char ID;
  X86CrtGuardPatternPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 CRT guard pattern (TEST8ri -> MOV8ri + TEST8rr)";
  }
};
} // end anonymous namespace

char X86CrtGuardPatternPass::ID = 0;

bool X86CrtGuardPatternPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::CrtGuardPattern))
    return false;

  const X86Subtarget &ST = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = ST.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E;) {
      MachineInstr &MI = *I;
      ++I;

      if (MI.getOpcode() != X86::TEST8ri)
        continue;

      // TEST8ri operands: reg(0), imm(1)
      // Only convert when the immediate is 1 (the guard bit check).
      if (!MI.getOperand(1).isImm() || MI.getOperand(1).getImm() != 1)
        continue;

      Register TestReg = MI.getOperand(0).getReg();
      DebugLoc DL = MI.getDebugLoc();

      // Insert: MOV8ri AL, 1
      BuildMI(MBB, MI, DL, TII->get(X86::MOV8ri), X86::AL)
          .addImm(1);

      // Insert: TEST8rr AL, TestReg
      BuildMI(MBB, MI, DL, TII->get(X86::TEST8rr))
          .addReg(X86::AL)
          .addReg(TestReg);

      // Remove original TEST8ri
      MI.eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86CrtGuardPatternPass() {
  return new X86CrtGuardPatternPass();
}

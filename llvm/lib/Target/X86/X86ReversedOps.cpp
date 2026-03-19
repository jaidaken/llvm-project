// bw1-decomp: Convert reg-reg arithmetic operations to their reversed
// encoding variants when the msvc6_regalloc attribute is set.
//
// MSVC 6.0 consistently uses the reversed encoding for reg-reg operations:
//   add.s eax, ecx  -> opcode 03 (MRMSrcReg) instead of 01 (MRMDestReg)
//   or.s  eax, ecx  -> opcode 0B instead of 09
//   sub.s eax, ecx  -> opcode 2B instead of 29
//   cmp.s eax, ecx  -> opcode 3B instead of 39
//   and.s eax, ecx  -> opcode 23 instead of 21
//   sbb.s eax, ecx  -> opcode 1B instead of 19
//   adc.s eax, ecx  -> opcode 13 instead of 11
//
// This does not change semantics -- both encodings produce the same result.
// The difference is purely in the byte encoding of the ModR/M field.

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

namespace {
class X86ReversedOpsPass : public MachineFunctionPass {
public:
  static char ID;
  X86ReversedOpsPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (!MF.getFunction().hasFnAttribute(Attribute::Msvc6RegAlloc))
      return false;

    bool Changed = false;
    for (auto &MBB : MF) {
      for (auto &MI : MBB) {
        unsigned NewOpc = 0;
        switch (MI.getOpcode()) {
        case X86::ADD32rr: NewOpc = X86::ADD32rr_REV; break;
        case X86::OR32rr:  NewOpc = X86::OR32rr_REV;  break;
        case X86::SUB32rr: NewOpc = X86::SUB32rr_REV; break;
        case X86::CMP32rr: NewOpc = X86::CMP32rr_REV; break;
        case X86::AND32rr: NewOpc = X86::AND32rr_REV; break;
        case X86::XOR32rr: NewOpc = X86::XOR32rr_REV; break;
        case X86::SBB32rr: NewOpc = X86::SBB32rr_REV; break;
        case X86::ADC32rr: NewOpc = X86::ADC32rr_REV; break;
        case X86::ADD8rr:  NewOpc = X86::ADD8rr_REV;  break;
        case X86::OR8rr:   NewOpc = X86::OR8rr_REV;   break;
        case X86::SUB8rr:  NewOpc = X86::SUB8rr_REV;  break;
        case X86::CMP8rr:  NewOpc = X86::CMP8rr_REV;  break;
        case X86::AND8rr:  NewOpc = X86::AND8rr_REV;  break;
        case X86::XOR8rr:  NewOpc = X86::XOR8rr_REV;  break;
        case X86::ADD16rr: NewOpc = X86::ADD16rr_REV; break;
        case X86::OR16rr:  NewOpc = X86::OR16rr_REV;  break;
        case X86::SUB16rr: NewOpc = X86::SUB16rr_REV; break;
        case X86::CMP16rr: NewOpc = X86::CMP16rr_REV; break;
        case X86::AND16rr: NewOpc = X86::AND16rr_REV; break;
        case X86::XOR16rr: NewOpc = X86::XOR16rr_REV; break;
        default: break;
        }
        if (NewOpc) {
          MI.setDesc(MF.getSubtarget().getInstrInfo()->get(NewOpc));
          Changed = true;
        }
      }
    }
    return Changed;
  }

  StringRef getPassName() const override {
    return "X86 MSVC 6.0 Reversed Register-Register Encoding";
  }
};
char X86ReversedOpsPass::ID = 0;
} // namespace

FunctionPass *llvm::createX86ReversedOpsPass() {
  return new X86ReversedOpsPass();
}

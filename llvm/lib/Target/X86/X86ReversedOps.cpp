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
    bool HasRegAlloc = MF.getFunction().hasFnAttribute(Attribute::Msvc6RegAlloc);
    bool HasMovRev = MF.getFunction().hasFnAttribute(Attribute::MOV32rr_REV);
    bool HasOrRev = MF.getFunction().hasFnAttribute(Attribute::OR32rr_REV);
    bool HasTestRev = MF.getFunction().hasFnAttribute(Attribute::TestRev);
    bool HasAdd32Rev = MF.getFunction().hasFnAttribute(Attribute::ADD32rr_REV);
    bool HasAdd8Rev = MF.getFunction().hasFnAttribute(Attribute::ADD8rr_REV);
    bool HasAdd16Rev = MF.getFunction().hasFnAttribute(Attribute::ADD16rr_REV);
    if (!HasRegAlloc && !HasMovRev && !HasOrRev && !HasTestRev &&
        !HasAdd32Rev && !HasAdd8Rev && !HasAdd16Rev)
      return false;

    bool Changed = false;
    for (auto &MBB : MF) {
      for (auto &MI : MBB) {
        unsigned NewOpc = 0;
        // Arithmetic reversal gated on msvc6_regalloc attribute.
        if (HasRegAlloc) {
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
        }
        // MOV32rr reversal gated on MOV32rr_REV attribute (not msvc6_regalloc)
        if (!NewOpc && MI.getOpcode() == X86::MOV32rr && HasMovRev)
          NewOpc = X86::MOV32rr_REV;
        // OR reversal gated on OR32rr_REV attribute (not msvc6_regalloc)
        if (!NewOpc && HasOrRev) {
          switch (MI.getOpcode()) {
          case X86::OR8rr:   NewOpc = X86::OR8rr_REV;   break;
          case X86::OR16rr:  NewOpc = X86::OR16rr_REV;  break;
          case X86::OR32rr:  NewOpc = X86::OR32rr_REV;  break;
          default: break;
          }
        }
        // ADD reversal gated on individual ADD*rr_REV attributes
        if (!NewOpc && HasAdd32Rev && MI.getOpcode() == X86::ADD32rr)
          NewOpc = X86::ADD32rr_REV;
        if (!NewOpc && HasAdd8Rev && MI.getOpcode() == X86::ADD8rr)
          NewOpc = X86::ADD8rr_REV;
        if (!NewOpc && HasAdd16Rev && MI.getOpcode() == X86::ADD16rr)
          NewOpc = X86::ADD16rr_REV;
        if (NewOpc) {
          MI.setDesc(MF.getSubtarget().getInstrInfo()->get(NewOpc));
          Changed = true;
        }
        // TEST operand swap gated on TestRev attribute.
        // TEST is commutative so swapping operands only changes the ModR/M
        // encoding, not the flags result.  MSVC 6.0 uses a different operand
        // order than Clang (e.g. 84 C8 vs 84 C1 for test al, cl).
        if (HasTestRev) {
          unsigned Opc = MI.getOpcode();
          if (Opc == X86::TEST8rr || Opc == X86::TEST16rr ||
              Opc == X86::TEST32rr) {
            Register Reg0 = MI.getOperand(0).getReg();
            Register Reg1 = MI.getOperand(1).getReg();
            if (Reg0 != Reg1) {
              MI.getOperand(0).setReg(Reg1);
              MI.getOperand(1).setReg(Reg0);
              Changed = true;
            }
          }
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

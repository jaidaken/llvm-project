// bw1-decomp: Convert MOVZX return patterns to partial-register MOV.
//
// MSVC 6.0 returns byte/word values in AL/AX without zero-extending to EAX.
// Modern compilers use MOVZX for safety. This pass converts:
//   movzx eax, byte ptr [mem]; ret  ->  mov al, byte ptr [mem]; ret
//   movzx eax, word ptr [mem]; ret  ->  mov ax, word ptr [mem]; ret
//   mov eax, small_imm; ret         ->  mov al, small_imm; ret (when imm fits in 8 bits)
//
// Gated by the Msvc6PartialReturn function attribute.

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-msvc6-partial-return"

namespace {
class X86Msvc6PartialReturnPass : public MachineFunctionPass {
public:
  static char ID;
  X86Msvc6PartialReturnPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 MSVC 6.0 partial register return";
  }
};
} // end anonymous namespace

char X86Msvc6PartialReturnPass::ID = 0;

bool X86Msvc6PartialReturnPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::Msvc6PartialReturn))
    return false;

  const X86Subtarget &ST = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = ST.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E; ++I) {
      MachineInstr &MI = *I;

      // Check for RET instruction
      if (MI.getOpcode() != X86::RET32 && MI.getOpcode() != X86::RETI32)
        continue;

      // Look at the instruction before RET
      if (I == MBB.begin())
        continue;
      auto Prev = std::prev(I);
      MachineInstr &PrevMI = *Prev;

      // Pattern 1: MOVZX32rm8 -> MOV8rm (movzx eax, byte [mem] -> mov al, [mem])
      if (PrevMI.getOpcode() == X86::MOVZX32rm8 &&
          PrevMI.getOperand(0).getReg() == X86::EAX) {
        MachineInstrBuilder NewMI = BuildMI(MBB, Prev, PrevMI.getDebugLoc(),
                                            TII->get(X86::MOV8rm), X86::AL);
        // Copy memory operands (base, scale, index, disp, segment)
        for (unsigned i = 1; i < PrevMI.getNumOperands(); ++i)
          NewMI.add(PrevMI.getOperand(i));
        NewMI.setMemRefs(PrevMI.memoperands());
        PrevMI.eraseFromParent();
        Changed = true;
        continue;
      }

      // Pattern 2: MOVZX32rm16 -> MOV16rm (movzx eax, word [mem] -> mov ax, [mem])
      if (PrevMI.getOpcode() == X86::MOVZX32rm16 &&
          PrevMI.getOperand(0).getReg() == X86::EAX) {
        MachineInstrBuilder NewMI = BuildMI(MBB, Prev, PrevMI.getDebugLoc(),
                                            TII->get(X86::MOV16rm), X86::AX);
        for (unsigned i = 1; i < PrevMI.getNumOperands(); ++i)
          NewMI.add(PrevMI.getOperand(i));
        NewMI.setMemRefs(PrevMI.memoperands());
        PrevMI.eraseFromParent();
        Changed = true;
        continue;
      }

      // Pattern 3: MOV32ri with small imm -> MOV8ri (mov eax, 5 -> mov al, 5)
      if (PrevMI.getOpcode() == X86::MOV32ri &&
          PrevMI.getOperand(0).getReg() == X86::EAX &&
          PrevMI.getOperand(1).isImm()) {
        int64_t Imm = PrevMI.getOperand(1).getImm();
        if (Imm >= 0 && Imm <= 255) {
          BuildMI(MBB, Prev, PrevMI.getDebugLoc(),
                  TII->get(X86::MOV8ri), X86::AL)
              .addImm(Imm);
          PrevMI.eraseFromParent();
          Changed = true;
          continue;
        }
      }
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86Msvc6PartialReturnPass() {
  return new X86Msvc6PartialReturnPass();
}

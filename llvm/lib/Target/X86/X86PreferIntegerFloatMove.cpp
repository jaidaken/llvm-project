// bw1-decomp: Rewrite fld+fstp pairs to integer mov+mov.
//
// MSVC 6.0 treats float parameters as raw 32-bit values:
//   mov eax, [esp+4]; mov [ecx+0x50], eax
//
// LLVM uses the FPU:
//   fld dword ptr [esp+4]; fstp dword ptr [ecx+0x50]
//
// This pass finds LD_F32m + ST_FP32m pairs (fld+fstp with no other FPU
// ops in between) and replaces them with MOV32rm + MOV32mr using EAX
// as the intermediate register.
//
// Must run AFTER the FP stackifier (which converts FP pseudos to real
// x87 instructions) and BEFORE other bw1-decomp passes.

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

namespace {
class X86PreferIntegerFloatMovePass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferIntegerFloatMovePass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 MSVC 6.0 integer float move (fld+fstp -> mov+mov)";
  }
};
char X86PreferIntegerFloatMovePass::ID = 0;
} // namespace

bool X86PreferIntegerFloatMovePass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::Msvc6RegAlloc))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Compute liveness to know which GPR is free for the intermediate mov.
    LivePhysRegs LiveRegs(*TRI);
    LiveRegs.addLiveOuts(MBB);

    // Walk backwards so liveness is tracked correctly.
    SmallVector<std::pair<MachineInstr*, MachineInstr*>, 4> ToReplace;

    for (auto I = MBB.rbegin(), E = MBB.rend(); I != E; ++I) {
      MachineInstr &MI = *I;
      LiveRegs.stepBackward(MI);

      // Look for ST_FP32m (fstp dword ptr [mem])
      if (MI.getOpcode() != X86::ST_FP32m)
        continue;

      // Check if the previous instruction is LD_F32m (fld dword ptr [mem])
      auto PrevIt = std::next(I); // reverse iterator: next = previous instruction
      if (PrevIt == E)
        continue;
      MachineInstr &Prev = *PrevIt;
      if (Prev.getOpcode() != X86::LD_F32m)
        continue;

      // Found fld+fstp pair. Check if a GPR is available.
      // Prefer EAX (MSVC 6.0 convention), fall back to EDX, ECX.
      Register IntermediateReg = 0;
      for (Register Candidate : {X86::EAX, X86::EDX, X86::ECX}) {
        if (!LiveRegs.contains(Candidate)) {
          IntermediateReg = Candidate;
          break;
        }
      }
      if (!IntermediateReg)
        continue; // No free GPR, can't transform

      // Also verify the fld source doesn't use our intermediate register
      // as a base/index (mov reg,[reg+N] when reg is the destination is fine
      // for MOV, unlike the expand_movzx overlap issue)
      ToReplace.push_back({&Prev, &MI});
    }

    // Apply replacements (forward order since we collected in reverse)
    for (auto &[Fld, Fstp] : ToReplace) {
      DebugLoc DL = Fld->getDebugLoc();

      // Build MOV32rm: load float as integer from fld's source address
      auto MovLoad = BuildMI(MBB, *Fld, DL, TII->get(X86::MOV32rm), X86::EAX);
      // Copy memory operands from LD_F32m (operands 0..4: base, scale, index, disp, seg)
      for (unsigned i = 0; i < Fld->getNumOperands(); ++i)
        MovLoad.add(Fld->getOperand(i));
      MovLoad.cloneMemRefs(*Fld);

      // Build MOV32mr: store integer to fstp's destination address
      auto MovStore = BuildMI(MBB, *Fstp, DL, TII->get(X86::MOV32mr));
      // Copy memory operands from ST_FP32m (operands 0..4: base, scale, index, disp, seg)
      for (unsigned i = 0; i < Fstp->getNumOperands(); ++i)
        MovStore.add(Fstp->getOperand(i));
      MovStore.addReg(X86::EAX);
      MovStore.cloneMemRefs(*Fstp);

      // Remove the fld and fstp
      Fld->eraseFromParent();
      Fstp->eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferIntegerFloatMovePass() {
  return new X86PreferIntegerFloatMovePass();
}

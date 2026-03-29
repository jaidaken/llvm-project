// bw1-decomp: Pre-regalloc analysis pass for defer_ret_eax_alloc.
//
// MSVC 6.0 sometimes computes a return value in a register other than EAX,
// then copies it to EAX only at the RET instruction. LLVM's register
// allocator normally assigns the return value virtual register directly
// to EAX (via the copy hint from the COPY $eax, %vreg before RET).
//
// This pass scans for COPY $eax, %vreg immediately before RET32/RETI32
// and records the source %vreg in X86MachineFunctionInfo::DeferredRetVRegs.
// The getRegAllocationHints function then steers those vregs away from EAX
// by providing EDX, ECX as hard hints. This forces the computation into
// a non-EAX register, producing the desired "test ah" instead of "test dh".
//
// This is a pure analysis pass - it does NOT modify any MachineIR.

#include "X86.h"
#include "X86MachineFunctionInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

using namespace llvm;

#define DEBUG_TYPE "x86-defer-ret-eax-alloc"
#define PASS_NAME "X86 defer ret eax alloc analysis"

namespace {
class X86DeferRetEaxAllocPass : public MachineFunctionPass {
public:
  static char ID;
  X86DeferRetEaxAllocPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return PASS_NAME; }
};
} // end anonymous namespace

char X86DeferRetEaxAllocPass::ID = 0;

bool X86DeferRetEaxAllocPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("defer_ret_eax_alloc"))
    return false;

  auto *MFI = MF.getInfo<X86MachineFunctionInfo>();

  for (auto &MBB : MF) {
    for (auto It = MBB.begin(), End = MBB.end(); It != End; ++It) {
      const MachineInstr &MI = *It;
      unsigned Opc = MI.getOpcode();

      // Look for RET32 or RETI32 (stdcall/thiscall ret with immediate).
      if (Opc != X86::RET32 && Opc != X86::RETI32)
        continue;

      // Walk backwards to find the immediately preceding COPY $eax, %vreg.
      if (It == MBB.begin())
        continue;

      auto Prev = std::prev(It);
      const MachineInstr &PrevMI = *Prev;
      if (!PrevMI.isCopy())
        continue;

      // Check: dest is $eax, source is a virtual register.
      const MachineOperand &Dst = PrevMI.getOperand(0);
      const MachineOperand &Src = PrevMI.getOperand(1);
      if (!Dst.isReg() || Dst.getReg() != X86::EAX)
        continue;
      if (!Src.isReg() || !Src.getReg().isVirtual())
        continue;

      Register VReg = Src.getReg();
      LLVM_DEBUG(dbgs() << "DeferRetEaxAlloc: recording vreg "
                        << printReg(VReg) << " in "
                        << MF.getName() << "\n");
      MFI->addDeferredRetVReg(VReg);
    }
  }

  return false; // Pure analysis, no IR modifications.
}

FunctionPass *llvm::createX86DeferRetEaxAllocPass() {
  return new X86DeferRetEaxAllocPass();
}

// bw1-decomp: Rewrite fucompp/fnstsw/sahf/setcc sequences to
// fcomp/fnstsw/test ah,N/jcc sequences matching MSVC 6.0 output.
//
// LLVM generates:
//   fld [a]; fld [b]; fucompp; fnstsw ax; sahf; seta/setae/setb/sete cl
//
// MSVC 6.0 generates:
//   fld [a]; fcomp [b]; fnstsw ax; test ah, N; je/jne target
//
// The pass:
// 1. Finds UCOM_FPPr (fucompp) preceded by LD_F32m/LD_F64m (fld [mem])
// 2. Replaces fld+fucompp with fcomp [mem] (folds load into compare)
// 3. Removes SAHF
// 4. Replaces SETCCr/JCC_1 condition with TEST8ri AH,mask + new JCC/SETcc
//
// Condition code mapping (SAHF sets CF=C0, PF=C2, ZF=C3):
//   COND_A  (CF=0,ZF=0) -> test ah, 0x41; COND_E  (both zero)
//   COND_AE (CF=0)       -> test ah, 0x01; COND_E  (C0 zero)
//   COND_B  (CF=1)       -> test ah, 0x01; COND_NE (C0 set)
//   COND_BE (CF=1|ZF=1)  -> test ah, 0x41; COND_NE (C0 or C3 set)
//   COND_E  (ZF=1)       -> test ah, 0x40; COND_NE (C3 set)
//   COND_NE (ZF=0)       -> test ah, 0x40; COND_E  (C3 zero)
//   COND_P  (PF=1)       -> test ah, 0x04; COND_NE (C2 set)
//   COND_NP (PF=0)       -> test ah, 0x04; COND_E  (C2 zero)

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

namespace {
class X86PreferFcompFnstswPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferFcompFnstswPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 MSVC 6.0 FPU comparison rewrite (fucompp -> fcomp/test ah)";
  }
};
char X86PreferFcompFnstswPass::ID = 0;

/// Map SAHF-based condition code to (test ah mask, replacement condition).
/// Returns false if the condition cannot be mapped.
static bool mapCondToTestAH(X86::CondCode CC, uint8_t &Mask,
                            X86::CondCode &NewCC) {
  switch (CC) {
  case X86::COND_A:  Mask = 0x41; NewCC = X86::COND_E;  return true;
  case X86::COND_AE: Mask = 0x01; NewCC = X86::COND_E;  return true;
  case X86::COND_B:  Mask = 0x01; NewCC = X86::COND_NE; return true;
  case X86::COND_BE: Mask = 0x41; NewCC = X86::COND_NE; return true;
  case X86::COND_E:  Mask = 0x40; NewCC = X86::COND_NE; return true;
  case X86::COND_NE: Mask = 0x40; NewCC = X86::COND_E;  return true;
  case X86::COND_P:  Mask = 0x04; NewCC = X86::COND_NE; return true;
  case X86::COND_NP: Mask = 0x04; NewCC = X86::COND_E;  return true;
  default: return false;
  }
}

} // namespace

bool X86PreferFcompFnstswPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::Msvc6RegAlloc))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E; /*below*/) {
      MachineInstr &MI = *I;

      // Look for UCOM_FPPr (fucompp)
      if (MI.getOpcode() != X86::UCOM_FPPr) {
        ++I;
        continue;
      }

      // Check if preceded by LD_F32m or LD_F64m (the second fld)
      if (I == MBB.begin()) { ++I; continue; }
      auto PrevI = std::prev(I);
      // Skip fxch if present (compiler sometimes inserts it)
      if (PrevI->getOpcode() == X86::XCH_F) {
        if (PrevI == MBB.begin()) { ++I; continue; }
        PrevI = std::prev(PrevI);
      }

      unsigned FldOpc = PrevI->getOpcode();
      unsigned FcompOpc = 0;
      if (FldOpc == X86::LD_F32m) FcompOpc = X86::FCOMP32m;
      else if (FldOpc == X86::LD_F64m) FcompOpc = X86::FCOMP64m;

      if (!FcompOpc) { ++I; continue; }

      // Find FNSTSW16r after fucompp
      auto FnstswI = std::next(I);
      if (FnstswI == E || FnstswI->getOpcode() != X86::FNSTSW16r) {
        ++I;
        continue;
      }

      // Find SAHF after FNSTSW16r. There may be intervening instructions
      // (COPY for AH extraction, xor for zero-init, kill markers, etc.)
      // Skip anything that doesn't clobber AH/AX or EFLAGS in a way that
      // would break the fnstsw->sahf chain.
      auto SahfI = std::next(FnstswI);
      bool FoundSahf = false;
      for (unsigned Skip = 0; SahfI != E && Skip < 8; ++SahfI, ++Skip) {
        if (SahfI->getOpcode() == X86::SAHF) {
          FoundSahf = true;
          break;
        }
        // Stop if something clobbers AX/AH (besides the expected COPY)
        if (SahfI->definesRegister(X86::AX, /*TRI=*/nullptr) &&
            !SahfI->isCopy())
          break;
      }

      if (!FoundSahf) {
        ++I;
        continue;
      }

      // Find the SETCCr or JCC_1 that uses the flags set by SAHF.
      // May have intervening instructions (copy, kill, xor for zero-init).
      auto CondI = std::next(SahfI);
      while (CondI != E && CondI->getOpcode() != X86::SETCCr &&
             CondI->getOpcode() != X86::SETCCm &&
             CondI->getOpcode() != X86::JCC_1) {
        // Stop if something clobbers EFLAGS
        if (CondI->definesRegister(X86::EFLAGS, /*TRI=*/nullptr) &&
            !CondI->isCopy())
          break;
        ++CondI;
      }

      if (CondI == E) { ++I; continue; }

      X86::CondCode OldCC = X86::COND_INVALID;
      bool IsSetCC = false;
      bool IsJCC = false;

      if (CondI->getOpcode() == X86::SETCCr) {
        OldCC = X86::getCondFromSETCC(*CondI);
        IsSetCC = true;
      } else if (CondI->getOpcode() == X86::JCC_1) {
        OldCC = X86::getCondFromBranch(*CondI);
        IsJCC = true;
      }

      if (OldCC == X86::COND_INVALID) { ++I; continue; }

      // Map the condition to test ah mask
      uint8_t Mask;
      X86::CondCode NewCC;
      if (!mapCondToTestAH(OldCC, Mask, NewCC)) { ++I; continue; }

      DebugLoc DL = MI.getDebugLoc();

      // Step 1: Replace fld [mem] + fucompp with fcomp [mem]
      // The fcomp instruction loads from memory and compares with ST(0),
      // then pops ST(0). The preceding fld loaded the first operand.
      auto FcompMI = BuildMI(MBB, *PrevI, PrevI->getDebugLoc(),
                              TII->get(FcompOpc));
      // Copy memory operands from the fld
      for (unsigned i = 0; i < PrevI->getNumOperands(); ++i)
        FcompMI.add(PrevI->getOperand(i));
      FcompMI.cloneMemRefs(*PrevI);

      // Remove fxch if it was between fld and fucompp
      auto CheckFxch = std::next(MachineBasicBlock::iterator(FcompMI.getInstr()));
      if (CheckFxch != E && CheckFxch->getOpcode() == X86::XCH_F)
        CheckFxch->eraseFromParent();

      // Step 2: Keep FNSTSW16r (fnstsw ax) as-is

      // Step 3: Insert TEST8ri AH, Mask right after FNSTSW16r
      // (replacing all the COPY/xor/SAHF stuff between fnstsw and setcc)
      auto TestInsertPt = std::next(MachineBasicBlock::iterator(FnstswI));
      BuildMI(MBB, TestInsertPt, DL, TII->get(X86::TEST8ri))
          .addReg(X86::AH)
          .addImm(Mask);

      // Step 4: Replace SETCCr/JCC_1 with new condition code
      // Must build a new instruction because the condition is baked into the
      // opcode encoding (0x90+CC for SETcc, 0x80+CC for Jcc).
      if (IsSetCC) {
        Register DstReg = CondI->getOperand(0).getReg();
        auto NewSet = BuildMI(MBB, *CondI, CondI->getDebugLoc(),
                              TII->get(X86::SETCCr), DstReg)
                          .addImm(NewCC);
        CondI->eraseFromParent();
        CondI = NewSet.getInstr()->getIterator();
      } else if (IsJCC) {
        MachineBasicBlock *Target = CondI->getOperand(0).getMBB();
        auto NewJcc = BuildMI(MBB, *CondI, CondI->getDebugLoc(),
                              TII->get(X86::JCC_1))
                          .addMBB(Target)
                          .addImm(NewCC);
        CondI->eraseFromParent();
        CondI = NewJcc.getInstr()->getIterator();
      }

      // Step 5: Remove only SAHF and COPY/kill instructions related to AH
      // extraction. Keep other instructions (like xor ecx,ecx for zero-init).
      {
        auto CleanI = std::next(MachineBasicBlock::iterator(FnstswI));
        while (CleanI != E && &*CleanI != &*CondI) {
          auto NextClean = std::next(CleanI);
          // Remove: SAHF, COPY involving AH/AX, kill markers
          if (CleanI->getOpcode() == X86::SAHF ||
              CleanI->isKill() ||
              (CleanI->isCopy() &&
               (CleanI->getOperand(0).getReg() == X86::AH ||
                CleanI->getOperand(0).getReg() == X86::AX ||
                (CleanI->getNumOperands() > 1 &&
                 CleanI->getOperand(1).isReg() &&
                 (CleanI->getOperand(1).getReg() == X86::AH ||
                  CleanI->getOperand(1).getReg() == X86::AX))))) {
            CleanI->eraseFromParent();
          }
          // Keep everything else (xor zeroing, etc.)
          CleanI = NextClean;
        }
      }

      // Step 6: Remove the old fld and fucompp
      PrevI->eraseFromParent();
      auto NextI = std::next(I);
      MI.eraseFromParent(); // fucompp
      I = NextI;

      Changed = true;
      continue;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferFcompFnstswPass() {
  return new X86PreferFcompFnstswPass();
}

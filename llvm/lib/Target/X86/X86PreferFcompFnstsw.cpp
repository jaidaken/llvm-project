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
      // Debug: print bail points for UCOM_FPPr matching
      if (MI.getOpcode() != X86::UCOM_FPPr) {
        ++I;
        continue;
      }

      // Check if preceded by LD_F32m or LD_F64m (the second fld)
      if (I == MBB.begin()) { ++I; continue; }
      auto PrevI = std::prev(I);
      // Skip and track fxch if present (compiler inserts it for gt/ge)
      MachineInstr *FxchToRemove = nullptr;
      if (PrevI->getOpcode() == X86::XCH_F) {
        FxchToRemove = &*PrevI;
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
        // Stop if something clobbers AX/AH (besides expected COPY/KILL)
        if (SahfI->definesRegister(X86::AX, /*TRI=*/nullptr) &&
            !SahfI->isCopy() && !SahfI->isKill())
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
        // Stop if something clobbers EFLAGS (but not COPY/KILL pseudo-ops)
        if (CondI->definesRegister(X86::EFLAGS, /*TRI=*/nullptr) &&
            !CondI->isCopy() && !CondI->isKill())
          break;
        ++CondI;
      }

      if (CondI == E) { ++I; continue; }

      X86::CondCode OldCC = X86::COND_INVALID;
      bool IsSetCC = false;
      bool IsJCC = false;
      bool CondAlreadyReplaced = false;
      uint8_t Mask = 0;
      X86::CondCode NewCC = X86::COND_INVALID;

      if (CondI->getOpcode() == X86::SETCCr) {
        OldCC = X86::getCondFromSETCC(*CondI);
        IsSetCC = true;

        // Handle the eq/ne pattern: setnp+sete+and (OEQ) or setp+setne+or (UNE).
        // LLVM uses two setccs for float equality. Collapse into test ah, 0x40.
        if (OldCC == X86::COND_NP || OldCC == X86::COND_P) {
          auto NextCond = std::next(CondI);
          if (NextCond != E && NextCond->getOpcode() == X86::SETCCr) {
            X86::CondCode SecondCC = X86::getCondFromSETCC(*NextCond);
            bool IsOEQ = (OldCC == X86::COND_NP && SecondCC == X86::COND_E);
            bool IsUNE = (OldCC == X86::COND_P && SecondCC == X86::COND_NE);
            if (IsOEQ || IsUNE) {
              // Find the combining AND8rr (OEQ) or OR8rr (UNE)
              auto CombI = std::next(NextCond);
              while (CombI != E && CombI->getOpcode() != X86::AND8rr &&
                     CombI->getOpcode() != X86::OR8rr)
                ++CombI;
              if (CombI != E) {
                Register CombDstReg = CombI->getOperand(0).getReg();
                X86::CondCode ResultCC = IsOEQ ? X86::COND_NE : X86::COND_E;
                // Replace with single setcc using the combine result register
                auto NewSet = BuildMI(MBB, *CondI, CondI->getDebugLoc(),
                                      TII->get(X86::SETCCr), CombDstReg)
                                  .addImm(ResultCC);
                // Remove first setcc, second setcc, and combine instruction
                CombI->eraseFromParent();
                NextCond->eraseFromParent();
                CondI->eraseFromParent();
                CondI = NewSet.getInstr()->getIterator();
                Mask = 0x40;
                NewCC = ResultCC;
                IsSetCC = true;
                CondAlreadyReplaced = true;
              }
            }
            if (!CondAlreadyReplaced) {
              // Could not match full eq/ne pattern, bail out
              ++I;
              continue;
            }
          }
        }
      } else if (CondI->getOpcode() == X86::JCC_1) {
        OldCC = X86::getCondFromBranch(*CondI);
        IsJCC = true;
      }

      if (OldCC == X86::COND_INVALID) { ++I; continue; }

      // Map the condition to test ah mask (unless eq/ne already set it)
      if (!CondAlreadyReplaced) {
        if (!mapCondToTestAH(OldCC, Mask, NewCC)) { ++I; continue; }
      }

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

      // Remove fxch that was between fld and fucompp (tracked earlier)
      if (FxchToRemove)
        FxchToRemove->eraseFromParent();

      // Step 2: Keep FNSTSW16r (fnstsw ax) as-is

      // Step 3: Insert TEST8ri AH, Mask right after FNSTSW16r
      // (replacing all the COPY/xor/SAHF stuff between fnstsw and setcc)
      auto TestInsertPt = std::next(MachineBasicBlock::iterator(FnstswI));
      BuildMI(MBB, TestInsertPt, DL, TII->get(X86::TEST8ri))
          .addReg(X86::AH)
          .addImm(Mask);

      // Step 4: Replace SETCCr/JCC_1 with new condition code
      // (skip if eq/ne pattern already handled this above)
      if (CondAlreadyReplaced) {
        // eq/ne setcc pattern already replaced CondI above
      } else if (IsSetCC) {
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
        // Check for companion JCC in eq/ne two-JCC pattern:
        // JCC_1 target, COND_NE + JCC_1 target, COND_P (OEQ)
        // JCC_1 target, COND_NE + JCC_1 target, COND_P (UNE)
        auto NextJcc = std::next(MachineBasicBlock::iterator(CondI));
        CondI->eraseFromParent();
        CondI = NewJcc.getInstr()->getIterator();
        if (NextJcc != E && NextJcc->getOpcode() == X86::JCC_1) {
          MachineBasicBlock *NextTarget = NextJcc->getOperand(0).getMBB();
          X86::CondCode NextCC = X86::getCondFromBranch(*NextJcc);
          if (NextCC == X86::COND_P || NextCC == X86::COND_NP) {
            if (NextTarget == Target) {
              // OEQ: both JCCs target same block - just remove companion
              NextJcc->eraseFromParent();
            } else {
              // UNE: companion JCC targets different block (the "equal" path).
              // Replace conditional jnp/jp with unconditional jmp to handle
              // the equal case without NaN checking (matches MSVC 6.0).
              BuildMI(MBB, *NextJcc, NextJcc->getDebugLoc(),
                      TII->get(X86::JMP_1))
                  .addMBB(NextTarget);
              NextJcc->eraseFromParent();
            }
          }
        }
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

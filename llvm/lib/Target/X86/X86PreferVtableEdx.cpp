//===--- X86PreferVtableEdx.cpp - Fix vtable call register conflict --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: After PreferMovPush unfolds PUSH32rmm to MOV32rm EAX + PUSH EAX,
// the register allocator may assign both the vtable pointer load and the
// parameter load to EAX, creating a conflict:
//
//   mov eax, [ecx]       ; vtable load
//   mov eax, [esp+4]     ; param load - clobbers vtable!
//   push eax             ; push param
//   call [eax+0x914]     ; WRONG: eax has param, not vtable
//
// MSVC 6.0 uses EDX for parameter loads in this pattern:
//
//   mov edx, [esp+4]     ; param load into EDX
//   mov eax, [ecx]       ; vtable load into EAX
//   push edx             ; push param from EDX
//   call [eax+0x914]     ; CORRECT: eax has vtable
//
// This pass detects the conflict and rewrites param loads to EDX.
//
// Supported patterns:
// - Params from any register base ([ESP+N], [ESI+N], [EDI+N], etc.)
// - N params (1, 2, 3, ...) with conflict detection per param
// - LEA/ADD param computation (LEA EAX, [REG+N] or ADD EAX, imm)
// - Vtable from subobject ([REG+N] where N != 0)
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-vtable-edx"

namespace {

/// Returns true if MI is a MOV32rm that defines DestReg.
static bool isMov32rmDef(const MachineInstr &MI, Register DestReg) {
  return MI.getOpcode() == X86::MOV32rm &&
         MI.getOperand(0).getReg() == DestReg;
}

/// Returns true if MI is a LEA32r that defines DestReg.
static bool isLea32rDef(const MachineInstr &MI, Register DestReg) {
  return MI.getOpcode() == X86::LEA32r &&
         MI.getOperand(0).getReg() == DestReg;
}

/// Returns true if MI is an ADD32ri or ADD32ri8 that defines DestReg.
static bool isAdd32riDef(const MachineInstr &MI, Register DestReg) {
  unsigned Opc = MI.getOpcode();
  return (Opc == X86::ADD32ri || Opc == X86::ADD32ri8) &&
         MI.getOperand(0).getReg() == DestReg;
}

/// Returns true if MI is a param-producing instruction (MOV32rm, LEA32r,
/// ADD32ri, ADD32ri8) that defines DestReg.
static bool isParamProducer(const MachineInstr &MI, Register DestReg) {
  return isMov32rmDef(MI, DestReg) ||
         isLea32rDef(MI, DestReg) ||
         isAdd32riDef(MI, DestReg);
}

/// Returns true if MI is a MOV32rm that looks like a vtable load:
/// defines CallBase and loads from a register base (not ESP).
/// Accepts any displacement (including non-zero for subobject vtables).
static bool isVtableLoad(const MachineInstr &MI, Register CallBase) {
  if (MI.getOpcode() != X86::MOV32rm)
    return false;
  if (MI.getOperand(0).getReg() != CallBase)
    return false;
  if (!MI.getOperand(1).isReg())
    return false;
  Register Base = MI.getOperand(1).getReg();
  return Base != X86::NoRegister && Base != X86::ESP;
}

/// Returns true if a memory-format instruction (MOV32rm or LEA32r) uses ESP
/// as its base register. Operand 1 is the base for these instructions.
static bool memUsesESP(const MachineInstr &MI) {
  return MI.getOperand(1).isReg() &&
         MI.getOperand(1).getReg() == X86::ESP;
}

/// Build a replacement param load into EDX, inserted before InsertBefore.
/// Clones the source addressing from OrigMI but targets EDX.
/// If OrigMI uses ESP as base, adds EspAdjust to the displacement.
static void buildParamLoadEdx(MachineBasicBlock &MBB,
                              MachineInstr &InsertBefore,
                              MachineInstr &OrigMI,
                              const X86InstrInfo *TII,
                              int64_t EspAdjust) {
  DebugLoc DL = OrigMI.getDebugLoc();
  unsigned Opc = OrigMI.getOpcode();

  if (Opc == X86::MOV32rm) {
    // MOV32rm: def, base, scale, index, disp, segment
    int64_t Disp = OrigMI.getOperand(4).getImm();
    if (memUsesESP(OrigMI))
      Disp += EspAdjust;
    BuildMI(MBB, InsertBefore, DL, TII->get(X86::MOV32rm), X86::EDX)
        .addReg(OrigMI.getOperand(1).getReg())
        .addImm(OrigMI.getOperand(2).getImm())
        .addReg(OrigMI.getOperand(3).getReg())
        .addImm(Disp)
        .addReg(OrigMI.getOperand(5).getReg());
  } else if (Opc == X86::LEA32r) {
    // LEA32r: def, base, scale, index, disp, segment
    int64_t Disp = OrigMI.getOperand(4).getImm();
    if (memUsesESP(OrigMI))
      Disp += EspAdjust;
    BuildMI(MBB, InsertBefore, DL, TII->get(X86::LEA32r), X86::EDX)
        .addReg(OrigMI.getOperand(1).getReg())
        .addImm(OrigMI.getOperand(2).getImm())
        .addReg(OrigMI.getOperand(3).getReg())
        .addImm(Disp)
        .addReg(OrigMI.getOperand(5).getReg());
  } else if (Opc == X86::ADD32ri || Opc == X86::ADD32ri8) {
    // ADD32ri: def, tied-src, imm
    // The source register of the ADD is in operand 1. We need to emit:
    //   MOV32rr EDX, src
    //   ADD EDX, imm
    // But if the source is CallBase (which it typically is since the ADD
    // overwrites it), we need the value of CallBase before the ADD.
    // Since the vtable load will overwrite CallBase anyway, we redirect
    // the ADD itself to EDX.
    Register SrcReg = OrigMI.getOperand(1).getReg();
    int64_t Imm = OrigMI.getOperand(2).getImm();

    // Copy the source into EDX first.
    BuildMI(MBB, InsertBefore, DL, TII->get(X86::MOV32rr), X86::EDX)
        .addReg(SrcReg);
    // Then add the immediate to EDX.
    BuildMI(MBB, InsertBefore, DL, TII->get(Opc), X86::EDX)
        .addReg(X86::EDX)
        .addImm(Imm);
  }
}

class X86PreferVtableEdxPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferVtableEdxPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 prefer vtable EDX param pass";
  }
};
} // end anonymous namespace

char X86PreferVtableEdxPass::ID = 0;

bool X86PreferVtableEdxPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::NoCalleeSaves))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E; ++I) {
      MachineInstr &CallMI = *I;

      // Step 1: Find CALL32m with a register base (indirect vtable call).
      // Also check FARCALL32m in case the compiler uses that variant.
      unsigned CallOpc = CallMI.getOpcode();
      if (CallOpc != X86::CALL32m && CallOpc != X86::FARCALL32m)
        continue;

      // CALL32m operands: base, scale, index, disp, segment
      if (!CallMI.getOperand(0).isReg())
        continue;
      Register CallBase = CallMI.getOperand(0).getReg();
      if (CallBase == X86::NoRegister || CallBase == X86::ESP)
        continue;

      // Collect preceding non-pseudo instructions (up to 20 to handle
      // many-param cases: N pushes + N loads + 1 vtable = 2N+1).
      SmallVector<MachineInstr*, 20> Prev;
      {
        auto WalkIt = MachineBasicBlock::iterator(CallMI);
        while (Prev.size() < 20 && WalkIt != MBB.begin()) {
          --WalkIt;
          if (!WalkIt->isPseudo())
            Prev.push_back(&*WalkIt);
        }
      }
      // Prev[0] = instruction right before CALL, Prev[1] = one before that...

      // ================================================================
      // FAST PATH: 2-PARAM PATTERN (ESP-only, original logic)
      // ================================================================
      // Prev[0]: PUSH32r RegB (param_1, e.g. EDX)
      // Prev[1]: PUSH32r CallBase (param_2, e.g. EAX)
      // Prev[2]: MOV32rm RegB, [ESP+N1] (param_1 load)
      // Prev[3]: MOV32rm CallBase, [ESP+N2] (param_2 load, clobbers vtable!)
      // Prev[4]: MOV32rm CallBase, [non-ESP] (vtable load)
      if (Prev.size() >= 5 &&
          Prev[0]->getOpcode() == X86::PUSH32r &&
          Prev[1]->getOpcode() == X86::PUSH32r &&
          Prev[1]->getOperand(0).getReg() == CallBase &&
          Prev[2]->getOpcode() == X86::MOV32rm &&
          Prev[2]->getOperand(1).isReg() &&
          Prev[2]->getOperand(1).getReg() == X86::ESP &&
          Prev[3]->getOpcode() == X86::MOV32rm &&
          Prev[3]->getOperand(0).getReg() == CallBase &&
          Prev[3]->getOperand(1).isReg() &&
          Prev[3]->getOperand(1).getReg() == X86::ESP &&
          Prev[4]->getOpcode() == X86::MOV32rm &&
          Prev[4]->getOperand(0).getReg() == CallBase &&
          Prev[4]->getOperand(1).isReg() &&
          Prev[4]->getOperand(1).getReg() != X86::ESP) {

        DebugLoc DL = Prev[3]->getDebugLoc();
        int64_t Param2Disp = Prev[3]->getOperand(4).getImm();
        int64_t Param1Disp = Prev[2]->getOperand(4).getImm();
        MachineInstr &VtableMI = *Prev[4];

        // Build: MOV32rm EDX, [ESP+Param2Disp] before vtable
        BuildMI(MBB, VtableMI, DL, TII->get(X86::MOV32rm), X86::EDX)
            .addReg(X86::ESP)
            .addImm(1).addReg(X86::NoRegister)
            .addImm(Param2Disp)
            .addReg(X86::NoRegister);

        // VtableMI stays (now after new param2 load)
        // Change push of param_2 from CallBase to EDX
        Prev[1]->getOperand(0).setReg(X86::EDX);

        // Build: MOV32rm EDX, [ESP+Param1Disp+4] between the two pushes
        // +4 because first push shifted ESP
        auto AfterPush2 = std::next(MachineBasicBlock::iterator(*Prev[1]));
        BuildMI(MBB, *AfterPush2, DL, TII->get(X86::MOV32rm), X86::EDX)
            .addReg(X86::ESP)
            .addImm(1).addReg(X86::NoRegister)
            .addImm(Param1Disp + 4)
            .addReg(X86::NoRegister);

        // Change push of param_1 to EDX
        Prev[0]->getOperand(0).setReg(X86::EDX);

        // Remove old param loads
        Prev[3]->eraseFromParent();
        Prev[2]->eraseFromParent();

        Changed = true;
        continue;
      }

      // ================================================================
      // FAST PATH: 1-PARAM PATTERN (ESP-only, original logic)
      // ================================================================
      // Prev[0]: PUSH32r CallBase
      // Prev[1]: MOV32rm CallBase, [ESP+N] (param load, clobbers vtable!)
      // Prev[2]: MOV32rm CallBase, [non-ESP] (vtable load)
      if (Prev.size() >= 3 &&
          Prev[0]->getOpcode() == X86::PUSH32r &&
          Prev[0]->getOperand(0).getReg() == CallBase &&
          Prev[1]->getOpcode() == X86::MOV32rm &&
          Prev[1]->getOperand(0).getReg() == CallBase &&
          Prev[1]->getOperand(1).isReg() &&
          Prev[1]->getOperand(1).getReg() == X86::ESP &&
          Prev[2]->getOpcode() == X86::MOV32rm &&
          Prev[2]->getOperand(0).getReg() == CallBase &&
          Prev[2]->getOperand(1).isReg() &&
          Prev[2]->getOperand(1).getReg() != X86::ESP) {

        // 1-param fix: change param load to EDX, reorder before vtable.
        MachineInstr &PushMI = *Prev[0];
        MachineInstr &ParamMI = *Prev[1];
        MachineInstr &VtableMI = *Prev[2];

        DebugLoc DL = ParamMI.getDebugLoc();

        // Build new param load: MOV32rm EDX, [ESP+offset]
        int64_t ParamDisp = ParamMI.getOperand(4).getImm();
        BuildMI(MBB, VtableMI, DL, TII->get(X86::MOV32rm), X86::EDX)
            .addReg(X86::ESP)
            .addImm(ParamMI.getOperand(2).getImm())  // scale
            .addReg(ParamMI.getOperand(3).getReg())   // index
            .addImm(ParamDisp)                         // disp
            .addReg(ParamMI.getOperand(5).getReg());   // segment

        // VtableMI stays in place (now after new param load).
        // Change push to use EDX.
        PushMI.getOperand(0).setReg(X86::EDX);

        // Remove old param load.
        ParamMI.eraseFromParent();

        Changed = true;
        continue;
      }

      // ================================================================
      // GENERALIZED N-PARAM PATTERN
      // ================================================================
      // Handles:
      //   - Params from any register base (not just ESP)
      //   - LEA32r and ADD32ri/ADD32ri8 param computation
      //   - N params (1, 2, 3, ...)
      //   - Vtable from subobject ([REG+N] where N != 0)
      //
      // Expected layout (walking backwards from CALL):
      //   [N pushes] [N param loads] [vtable load]
      //
      // A param load "conflicts" if it writes to CallBase, clobbering the
      // vtable pointer. Conflicting loads are redirected to EDX.
      //
      // Rewrite: all loads that define CallBase or EDX are relocated right
      // before their corresponding push, with ESP displacement adjustments
      // for the number of pushes that have already executed. Conflicting
      // loads are redirected to EDX. The first relocated load (in execution
      // order) is placed before VtableMI since no pushes have happened yet.

      if (Prev.size() < 3)
        continue;

      // Phase 1: Collect consecutive PUSH32r instructions walking backwards.
      unsigned NumPushes = 0;
      while (NumPushes < Prev.size() &&
             Prev[NumPushes]->getOpcode() == X86::PUSH32r)
        ++NumPushes;

      if (NumPushes == 0)
        continue;

      // Phase 2: Collect param load instructions (one per push).
      // Each param load should define the register used by the push at the
      // same position. The loads appear in Prev after all pushes, in the
      // same order (Prev[NumPushes + k] is the load for Prev[k]'s push).
      unsigned ExpectedEnd = NumPushes * 2 + 1; // N pushes + N loads + vtable
      if (Prev.size() < ExpectedEnd)
        continue;

      SmallVector<MachineInstr*, 8> Pushes;
      SmallVector<MachineInstr*, 8> Loads;
      bool ValidStructure = true;

      for (unsigned k = 0; k < NumPushes; ++k) {
        Pushes.push_back(Prev[k]);
        MachineInstr *LoadMI = Prev[NumPushes + k];
        Register PushReg = Prev[k]->getOperand(0).getReg();

        // The load must define the register that the push uses.
        if (!isParamProducer(*LoadMI, PushReg)) {
          ValidStructure = false;
          break;
        }
        Loads.push_back(LoadMI);
      }

      if (!ValidStructure)
        continue;

      // Phase 3: The instruction after all loads should be the vtable load.
      MachineInstr *VtableMI = Prev[NumPushes * 2];
      if (!isVtableLoad(*VtableMI, CallBase))
        continue;

      // Phase 4: Identify which params conflict (write to CallBase).
      SmallVector<unsigned, 8> ConflictIndices;
      for (unsigned k = 0; k < NumPushes; ++k) {
        if (Loads[k]->getOperand(0).getReg() == CallBase)
          ConflictIndices.push_back(k);
      }

      // No conflicts means no rewrite needed.
      if (ConflictIndices.empty())
        continue;

      // Cannot redirect to EDX if CallBase is already EDX.
      if (CallBase == X86::EDX)
        continue;

      // Phase 5: Identify params that need relocation.
      // "Relocate" means the load must be moved right before its push.
      // Two reasons to relocate:
      //   (a) Load defines CallBase (conflict) - redirect to EDX.
      //   (b) Load defines EDX (EDX hazard) - our new EDX loads would
      //       clobber it, so move it right before its push to stay safe.
      //
      // Params defining a register other than CallBase or EDX are safe
      // and stay in their original position.
      SmallVector<unsigned, 8> RelocateIndices; // sorted ascending (low = close to CALL)
      bool IsConflict[20] = {};  // true if load defines CallBase
      bool Feasible = true;

      for (unsigned k = 0; k < NumPushes; ++k) {
        Register DefReg = Loads[k]->getOperand(0).getReg();
        if (DefReg == CallBase) {
          RelocateIndices.push_back(k);
          IsConflict[k] = true;
        } else if (DefReg == X86::EDX) {
          // EDX hazard: must relocate to avoid clobbering.
          // ADD32ri with tied EDX source can't be safely relocated because
          // EDX may have been clobbered by a prior conflict's EDX load.
          if (isAdd32riDef(*Loads[k], X86::EDX)) {
            Feasible = false;
            break;
          }
          // If the load reads CallBase (e.g., MOV32rm EDX, [EAX+N] where
          // EAX=CallBase), relocating after vtable load would read the
          // vtable pointer instead of the intended value.
          unsigned LOpc = Loads[k]->getOpcode();
          if (LOpc == X86::MOV32rm || LOpc == X86::LEA32r) {
            if (Loads[k]->getOperand(1).getReg() == CallBase ||
                Loads[k]->getOperand(3).getReg() == CallBase) {
              Feasible = false;
              break;
            }
          }
          RelocateIndices.push_back(k);
          IsConflict[k] = false;
        }
      }

      // Safety: check that no non-relocated load uses EDX as a source.
      // For MOV32rm/LEA32r, check the base register (op1) and index (op3).
      // For ADD32ri, the source (op1) is being overwritten, so if src is
      // EDX we can't safely redirect to EDX.
      for (unsigned k = 0; k < NumPushes; ++k) {
        bool NeedsRelocate = false;
        for (unsigned ri : RelocateIndices) {
          if (ri == k) { NeedsRelocate = true; break; }
        }
        if (NeedsRelocate)
          continue;

        // Non-relocated load: verify it doesn't use EDX in any operand
        // that we might clobber.
        MachineInstr &LMI = *Loads[k];
        unsigned Opc = LMI.getOpcode();
        if (Opc == X86::MOV32rm || Opc == X86::LEA32r) {
          if (LMI.getOperand(1).getReg() == X86::EDX ||
              LMI.getOperand(3).getReg() == X86::EDX) {
            Feasible = false;
            break;
          }
        }
      }

      // Vtable load base must not be EDX.
      if (VtableMI->getOperand(1).getReg() == X86::EDX)
        Feasible = false;

      // Check if the first relocated load (highest index, first to execute)
      // reads CallBase in its source addressing. If so, it depends on the
      // vtable load having executed first, so it cannot be placed before
      // VtableMI. We mark this so Phase 6 places it after VtableMI instead.
      bool FirstReadsCallBase = false;
      if (Feasible && !RelocateIndices.empty()) {
        unsigned firstCi = RelocateIndices.back(); // highest index
        MachineInstr &FirstLoad = *Loads[firstCi];
        unsigned FLOpc = FirstLoad.getOpcode();
        if (FLOpc == X86::MOV32rm || FLOpc == X86::LEA32r) {
          if (FirstLoad.getOperand(1).getReg() == CallBase ||
              FirstLoad.getOperand(3).getReg() == CallBase)
            FirstReadsCallBase = true;
        }
        // ADD32ri with CallBase source (tied): the ADD reads the pre-vtable
        // value. Moving before vtable is correct - it restores the intended
        // computation that the register allocator conflict broke.
      }

      if (!Feasible)
        continue;

      LLVM_DEBUG(dbgs() << "VtableEdx: matched " << NumPushes
                        << "-param pattern with " << ConflictIndices.size()
                        << " conflicts, " << RelocateIndices.size()
                        << " relocations\n");

      // Phase 6: Rewrite.
      //
      // Process relocated params in execution order: highest index first
      // (farthest from CALL = first to execute after VtableMI).
      //
      // Pushes are ordered: Pushes[NumPushes-1] executes first (farthest
      // from CALL), Pushes[0] executes last (closest to CALL).
      //
      // For each relocated param at index ci:
      //   - If it's the first in execution order: place load before VtableMI
      //     (no ESP adjustment, no pushes have executed yet).
      //   - Otherwise: place load right before Pushes[ci]. The ESP adjustment
      //     is 4 * (number of pushes that execute before Pushes[ci]).
      //     Pushes executing before ci are those with index > ci.
      //
      // For conflict loads (CallBase): redirect to EDX.
      // For EDX-hazard loads: keep as EDX, just relocate with ESP adjust.

      // Process in execution order: highest index first (farthest from CALL).
      // RelocateIndices is sorted ascending, so iterate in reverse.
      for (unsigned ei = 0; ei < RelocateIndices.size(); ++ei) {
        unsigned ci = RelocateIndices[RelocateIndices.size() - 1 - ei];
        unsigned PushesBefore = NumPushes - 1 - ci;
        int64_t EspAdj = 4 * (int64_t)PushesBefore;

        // Determine insertion point:
        // First in execution order goes before VtableMI (no pushes yet),
        // UNLESS it reads CallBase in source (needs vtable loaded first).
        bool BeforeVtable = (ei == 0) && !FirstReadsCallBase;

        if (IsConflict[ci]) {
          // Conflict: redirect to EDX.
          if (BeforeVtable) {
            buildParamLoadEdx(MBB, *VtableMI, *Loads[ci], TII,
                              /*EspAdjust=*/0);
          } else {
            buildParamLoadEdx(MBB, *Pushes[ci], *Loads[ci], TII, EspAdj);
          }
          Pushes[ci]->getOperand(0).setReg(X86::EDX);
        } else {
          // EDX hazard: keep as EDX load, just relocate.
          if (BeforeVtable) {
            buildParamLoadEdx(MBB, *VtableMI, *Loads[ci], TII,
                              /*EspAdjust=*/0);
          } else {
            buildParamLoadEdx(MBB, *Pushes[ci], *Loads[ci], TII, EspAdj);
          }
          // Push already uses EDX, no change needed.
        }
      }

      // Remove all old relocated param loads.
      for (unsigned ci : RelocateIndices) {
        Loads[ci]->eraseFromParent();
      }

      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferVtableEdxPass() {
  return new X86PreferVtableEdxPass();
}

//===--- X86ZeroViaXor.cpp - Convert imm-zero stores to XOR+store ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: MSVC 6.0 zeros memory by first zeroing a register with XOR,
// then storing through it:
//
//   xor.s ecx, ecx        ; 33 c9 (reversed XOR encoding, 2 bytes)
//   mov [eax], ecx         ; 89 08 (2 bytes)
//   mov [eax+4], ecx       ; 89 48 04 (3 bytes)
//   mov [eax+8], ecx       ; 89 48 08 (3 bytes)
//
// LLVM uses immediate-zero stores:
//
//   mov dword ptr [eax], 0     ; c7 00 00 00 00 00 (6 bytes)
//   mov dword ptr [eax+4], 0   ; c7 40 04 00 00 00 00 (7 bytes)
//   mov dword ptr [eax+8], 0   ; c7 40 08 00 00 00 00 (7 bytes)
//
// The MSVC pattern is shorter and produces different encodings. This pass
// finds consecutive MOV32mi instructions storing 0 to addresses based on
// the same register, inserts a XOR32rr self-xor of a scratch register
// before the first store, and replaces each MOV32mi with MOV32mr using
// the scratch register.
//
// Gate: function attribute "zero_via_xor".
//
// The X86ReversedOpsPass will later convert XOR32rr to XOR32rr_REV if
// msvc6_regalloc is set.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-zero-via-xor"

namespace {

class X86ZeroViaXorPass : public MachineFunctionPass {
public:
  static char ID;
  X86ZeroViaXorPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 convert immediate-zero stores to XOR+store";
  }
};

} // end anonymous namespace

char X86ZeroViaXorPass::ID = 0;

/// Check whether MI is a MOV32mi that stores immediate 0.
static bool isZeroStore32(const MachineInstr &MI) {
  if (MI.getOpcode() != X86::MOV32mi)
    return false;
  // MOV32mi operands: [0]=base, [1]=scale, [2]=index, [3]=disp, [4]=segment,
  //                   [5]=immediate
  if (MI.getNumOperands() < 6)
    return false;
  const MachineOperand &ImmOp = MI.getOperand(5);
  return ImmOp.isImm() && ImmOp.getImm() == 0;
}

/// Return the base register of a MOV32mi instruction (operand 0).
static Register getBaseReg(const MachineInstr &MI) {
  return MI.getOperand(0).getReg();
}

bool X86ZeroViaXorPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("zero_via_xor"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // We process the block by scanning forward, collecting groups of
    // consecutive MOV32mi-imm0 instructions that share the same base register.
    // For each group, we check scratch register availability and EFLAGS
    // liveness at the point of the first store.
    //
    // We use a two-pass approach: first scan backward with LivePhysRegs to
    // record liveness at each instruction, then scan forward to find groups.

    // Pass 1: Build a map from MachineInstr* to a snapshot of which
    // relevant registers are live just before that instruction.
    // We only need EFLAGS + the scratch candidates (EAX, ECX, EDX).
    struct LiveInfo {
      bool EflagsLive = false;
      bool EaxLive = false;
      bool EcxLive = false;
      bool EdxLive = false;
      bool contains(Register R) const {
        if (R == X86::EFLAGS) return EflagsLive;
        if (R == X86::EAX) return EaxLive;
        if (R == X86::ECX) return EcxLive;
        if (R == X86::EDX) return EdxLive;
        return false;
      }
    };
    DenseMap<MachineInstr *, LiveInfo> LivenessMap;
    {
      LivePhysRegs LiveRegs(*TRI);
      LiveRegs.addLiveOuts(MBB);

      for (auto I = MBB.rbegin(), E = MBB.rend(); I != E; ++I) {
        MachineInstr &MI = *I;
        LiveRegs.stepBackward(MI);
        if (isZeroStore32(MI)) {
          LiveInfo LI;
          LI.EflagsLive = LiveRegs.contains(X86::EFLAGS);
          LI.EaxLive = LiveRegs.contains(X86::EAX);
          LI.EcxLive = LiveRegs.contains(X86::ECX);
          LI.EdxLive = LiveRegs.contains(X86::EDX);
          LivenessMap[&MI] = LI;
        }
      }
    }

    // Pass 2: Scan forward to find groups of consecutive zero stores.
    auto I = MBB.begin();
    while (I != MBB.end()) {
      MachineInstr &FirstMI = *I;
      if (!isZeroStore32(FirstMI)) {
        ++I;
        continue;
      }

      // Found a zero store. Collect consecutive zero stores with the same
      // base register.
      Register BaseReg = getBaseReg(FirstMI);
      SmallVector<MachineInstr *, 8> Group;
      Group.push_back(&FirstMI);

      auto J = std::next(I);
      while (J != MBB.end() && isZeroStore32(*J) &&
             getBaseReg(*J) == BaseReg) {
        Group.push_back(&*J);
        ++J;
      }

      // Need at least 2 consecutive zero stores to be worth transforming.
      // (A single store wouldn't save bytes: XOR is 2 bytes, MOV32mr is
      // same or larger than MOV32mi for a single store with no displacement.)
      if (Group.size() < 2) {
        I = J;
        continue;
      }

      // Check EFLAGS liveness before the first store.
      auto It = LivenessMap.find(Group[0]);
      if (It == LivenessMap.end()) {
        I = J;
        continue;
      }
      const LiveInfo &LiveAtFirst = It->second;
      if (LiveAtFirst.EflagsLive) {
        // XOR clobbers EFLAGS; can't insert here.
        I = J;
        continue;
      }

      // Select a scratch register. Prefer ECX (MSVC pattern for __fastcall
      // where ECX held this_ptr and was moved to EAX, freeing ECX).
      // Fall back to EDX, then EAX.
      static const Register ScratchCandidates[] = {X86::ECX, X86::EDX,
                                                   X86::EAX};
      Register ScratchReg;

      for (Register Cand : ScratchCandidates) {
        if (LiveAtFirst.contains(Cand))
          continue;

        // The scratch register must not overlap with the base register used
        // in the stores (we'd zero the base before storing through it).
        if (TRI->regsOverlap(Cand, BaseReg))
          continue;

        // Check that the scratch register doesn't overlap with index
        // registers used in any of the group's stores.
        bool OverlapsGroupMem = false;
        for (MachineInstr *MI : Group) {
          // Operand 2 is the index register.
          const MachineOperand &IndexOp = MI->getOperand(2);
          if (IndexOp.isReg() && IndexOp.getReg() != 0 &&
              TRI->regsOverlap(Cand, IndexOp.getReg())) {
            OverlapsGroupMem = true;
            break;
          }
        }
        if (OverlapsGroupMem)
          continue;

        ScratchReg = Cand;
        break;
      }

      if (!ScratchReg) {
        // No available scratch register; skip this group.
        I = J;
        continue;
      }

      // Insert XOR32rr scratch, scratch before the first store.
      DebugLoc DL = Group[0]->getDebugLoc();
      BuildMI(MBB, *Group[0], DL, TII->get(X86::XOR32rr), ScratchReg)
          .addReg(ScratchReg, RegState::Undef)
          .addReg(ScratchReg, RegState::Undef);

      // Replace each MOV32mi with MOV32mr using the scratch register.
      for (MachineInstr *MI : Group) {
        DebugLoc StoreDL = MI->getDebugLoc();

        // Build MOV32mr with same memory operands: base, scale, index, disp,
        // segment, then the source register.
        auto MIB =
            BuildMI(MBB, *MI, StoreDL, TII->get(X86::MOV32mr));
        // Copy memory operands (operands 0..4).
        for (unsigned i = 0; i < 5; ++i)
          MIB.add(MI->getOperand(i));
        // Add the scratch register as the source.
        MIB.addReg(ScratchReg);
        // Copy memory references for alias analysis.
        MIB.cloneMemRefs(*MI);

        MI->eraseFromParent();
      }

      Changed = true;
      I = J;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86ZeroViaXorPass() {
  return new X86ZeroViaXorPass();
}

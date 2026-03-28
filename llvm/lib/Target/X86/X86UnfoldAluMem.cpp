//===--- X86UnfoldAluMem.cpp - Unfold memory-source ALU to MOV+ALU --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: MSVC 6.0 generates separate load + register-register ALU:
//   mov edx, [ecx + 0x0c]    ; load from memory
//   add.s eax, edx            ; register-register add (reversed encoding)
//
// LLVM folds these into a single memory-source ALU:
//   add eax, [ecx + 0x0c]    ; memory-source add
//
// Both are semantically identical but produce different bytes. This pass
// unfolds memory-source ALU instructions back into MOV + register-register
// ALU to match the MSVC 6.0 output. The reversed ops pass will later
// convert the register-register ALU to the reversed encoding.
//
// Gate: function attribute "unfold_alu_mem" or "prevent_add_load_fold".
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-unfold-alu-mem"

namespace {

/// Describes an ALU-rm instruction and its unfolded equivalents.
struct AluRmEntry {
  unsigned RmOpc;    // Memory-source ALU opcode (e.g. ADD32rm)
  unsigned RrOpc;    // Register-register ALU opcode (e.g. ADD32rr)
  unsigned MovOpc;   // MOV from memory opcode (e.g. MOV32rm)
  bool HasDef;       // true for ADD/SUB/OR/AND/XOR, false for CMP
  // Scratch register candidates, in preference order.
  // For 32-bit: EDX, ECX, EAX; for 16-bit: DX, CX, AX; for 8-bit: DL, CL, AL.
  Register ScratchCandidates[3];
};

// 32-bit entries
static const AluRmEntry AluRm32Entries[] = {
  { X86::ADD32rm, X86::ADD32rr, X86::MOV32rm, true,  { X86::EDX, X86::ECX, X86::EAX } },
  { X86::SUB32rm, X86::SUB32rr, X86::MOV32rm, true,  { X86::EDX, X86::ECX, X86::EAX } },
  { X86::OR32rm,  X86::OR32rr,  X86::MOV32rm, true,  { X86::EDX, X86::ECX, X86::EAX } },
  { X86::AND32rm, X86::AND32rr, X86::MOV32rm, true,  { X86::EDX, X86::ECX, X86::EAX } },
  { X86::XOR32rm, X86::XOR32rr, X86::MOV32rm, true,  { X86::EDX, X86::ECX, X86::EAX } },
  { X86::CMP32rm, X86::CMP32rr, X86::MOV32rm, false, { X86::EDX, X86::ECX, X86::EAX } },
  { X86::SBB32rm, X86::SBB32rr, X86::MOV32rm, true,  { X86::EDX, X86::ECX, X86::EAX } },
  { X86::ADC32rm, X86::ADC32rr, X86::MOV32rm, true,  { X86::EDX, X86::ECX, X86::EAX } },
};

// 16-bit entries
static const AluRmEntry AluRm16Entries[] = {
  { X86::ADD16rm, X86::ADD16rr, X86::MOV16rm, true,  { X86::DX, X86::CX, X86::AX } },
  { X86::SUB16rm, X86::SUB16rr, X86::MOV16rm, true,  { X86::DX, X86::CX, X86::AX } },
  { X86::OR16rm,  X86::OR16rr,  X86::MOV16rm, true,  { X86::DX, X86::CX, X86::AX } },
  { X86::AND16rm, X86::AND16rr, X86::MOV16rm, true,  { X86::DX, X86::CX, X86::AX } },
  { X86::XOR16rm, X86::XOR16rr, X86::MOV16rm, true,  { X86::DX, X86::CX, X86::AX } },
  { X86::CMP16rm, X86::CMP16rr, X86::MOV16rm, false, { X86::DX, X86::CX, X86::AX } },
};

// 8-bit entries
static const AluRmEntry AluRm8Entries[] = {
  { X86::ADD8rm, X86::ADD8rr, X86::MOV8rm, true,  { X86::DL, X86::CL, X86::AL } },
  { X86::SUB8rm, X86::SUB8rr, X86::MOV8rm, true,  { X86::DL, X86::CL, X86::AL } },
  { X86::OR8rm,  X86::OR8rr,  X86::MOV8rm, true,  { X86::DL, X86::CL, X86::AL } },
  { X86::AND8rm, X86::AND8rr, X86::MOV8rm, true,  { X86::DL, X86::CL, X86::AL } },
  { X86::XOR8rm, X86::XOR8rr, X86::MOV8rm, true,  { X86::DL, X86::CL, X86::AL } },
  { X86::CMP8rm, X86::CMP8rr, X86::MOV8rm, false, { X86::DL, X86::CL, X86::AL } },
};

/// Look up the table entry for a given opcode. Returns nullptr if not found.
static const AluRmEntry *findEntry(unsigned Opc) {
  for (const auto &E : AluRm32Entries)
    if (E.RmOpc == Opc) return &E;
  for (const auto &E : AluRm16Entries)
    if (E.RmOpc == Opc) return &E;
  for (const auto &E : AluRm8Entries)
    if (E.RmOpc == Opc) return &E;
  return nullptr;
}

class X86UnfoldAluMemPass : public MachineFunctionPass {
public:
  static char ID;
  X86UnfoldAluMemPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 unfold memory-source ALU to MOV+ALU";
  }
};
} // end anonymous namespace

char X86UnfoldAluMemPass::ID = 0;

bool X86UnfoldAluMemPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("unfold_alu_mem") &&
      !MF.getFunction().hasFnAttribute("prevent_add_load_fold"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Two-pass approach to avoid iterator invalidation.
    // Pass 1: Walk backwards to track liveness and collect instructions.
    struct UnfoldCandidate {
      MachineInstr *MI;
      const AluRmEntry *Entry;
      Register ScratchReg;
    };
    SmallVector<UnfoldCandidate, 8> Candidates;

    {
      LivePhysRegs LiveRegs(*TRI);
      LiveRegs.addLiveOuts(MBB);

      for (auto I = MBB.rbegin(), E = MBB.rend(); I != E; ++I) {
        MachineInstr &MI = *I;

        const AluRmEntry *Entry = findEntry(MI.getOpcode());
        if (Entry) {
          // Determine the operand layout.
          // For HasDef instructions (ADD/SUB/OR/AND/XOR/SBB/ADC):
          //   [0] = dst (def, tied to src1)
          //   [1] = src1 (tied to dst)
          //   [2..6] = memory operands (base, scale, index, disp, segment)
          //
          // For CMP (no def):
          //   [0] = src1 (reg)
          //   [1..5] = memory operands (base, scale, index, disp, segment)
          unsigned MemStart = Entry->HasDef ? 2 : 1;
          Register Src1Reg = Entry->HasDef ? MI.getOperand(1).getReg()
                                           : MI.getOperand(0).getReg();

          // Find a scratch register that is:
          // 1. Not live at this point
          // 2. Does not overlap with src1 (the ALU destination/source register)
          // 3. Does not overlap with any register used in the memory operand
          Register ScratchReg;
          for (unsigned c = 0; c < 3; ++c) {
            Register Cand = Entry->ScratchCandidates[c];
            if (LiveRegs.contains(Cand))
              continue;
            if (TRI->regsOverlap(Cand, Src1Reg))
              continue;

            // Check memory operand registers (base and index).
            bool OverlapsMem = false;
            for (unsigned i = MemStart; i < MemStart + 5; ++i) {
              if (i >= MI.getNumOperands())
                break;
              const MachineOperand &MO = MI.getOperand(i);
              if (MO.isReg() && MO.getReg() != 0 &&
                  TRI->regsOverlap(Cand, MO.getReg())) {
                OverlapsMem = true;
                break;
              }
            }
            if (OverlapsMem)
              continue;

            ScratchReg = Cand;
            break;
          }

          if (ScratchReg)
            Candidates.push_back({&MI, Entry, ScratchReg});
        }

        LiveRegs.stepBackward(MI);
      }
    }

    // Pass 2: Unfold collected instructions.
    for (auto &Cand : Candidates) {
      MachineInstr *MI = Cand.MI;
      const AluRmEntry *Entry = Cand.Entry;
      Register ScratchReg = Cand.ScratchReg;
      DebugLoc DL = MI->getDebugLoc();

      unsigned MemStart = Entry->HasDef ? 2 : 1;

      // 1. Insert MOV scratch, [mem] before the ALU instruction.
      auto MovMIB = BuildMI(MBB, *MI, DL, TII->get(Entry->MovOpc), ScratchReg);
      for (unsigned i = MemStart; i < MemStart + 5; ++i)
        MovMIB.add(MI->getOperand(i));
      MovMIB.cloneMemRefs(*MI);

      // 2. Replace the memory-source ALU with register-register ALU.
      if (Entry->HasDef) {
        // ADD32rm: [0]=dst, [1]=src1, [2..6]=mem
        // -> ADD32rr: [0]=dst, [1]=src1, [2]=scratch
        Register DstReg = MI->getOperand(0).getReg();
        Register Src1Reg = MI->getOperand(1).getReg();
        BuildMI(MBB, *MI, DL, TII->get(Entry->RrOpc), DstReg)
            .addReg(Src1Reg)
            .addReg(ScratchReg);
      } else {
        // CMP32rm: [0]=src1, [1..5]=mem
        // -> CMP32rr: [0]=src1, [1]=scratch
        Register Src1Reg = MI->getOperand(0).getReg();
        BuildMI(MBB, *MI, DL, TII->get(Entry->RrOpc))
            .addReg(Src1Reg)
            .addReg(ScratchReg);
      }

      MI->eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86UnfoldAluMemPass() {
  return new X86UnfoldAluMemPass();
}

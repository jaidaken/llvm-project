//===--- X86PreferByteParamLoad.cpp - Load stack params as bytes -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: MSVC 6.0 loads stack parameters as bytes even when the type is
// int, using the pattern:
//   xor eax, eax
//   mov al, [esp+0x04]
//
// Clang generates the full 32-bit load:
//   mov eax, [esp+0x04]
//
// This pass converts MOV32rm reg, [ESP+offset] to XOR32rr_REV reg, reg +
// MOV8rm low(reg), [ESP+offset] for stack parameter offsets specified in the
// prefer_byte_param_load attribute.
//
// Attribute format: prefer_byte_param_load("4") or prefer_byte_param_load("4,8")
// where each number is the ESP offset of a param to load as a byte.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-byte-param-load"

namespace {
class X86PreferByteParamLoadPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferByteParamLoadPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 prefer byte param load (MSVC 6.0)";
  }

private:
  static bool parseOffsets(StringRef AttrVal,
                           SmallVectorImpl<int64_t> &Offsets);
};
} // end anonymous namespace

char X86PreferByteParamLoadPass::ID = 0;

/// Map a 32-bit register to its 8-bit low sub-register.
static Register get8BitSubReg(Register Reg32) {
  switch (Reg32) {
  case X86::EAX: return X86::AL;
  case X86::ECX: return X86::CL;
  case X86::EDX: return X86::DL;
  case X86::EBX: return X86::BL;
  default: return X86::NoRegister;
  }
}

bool X86PreferByteParamLoadPass::parseOffsets(
    StringRef AttrVal, SmallVectorImpl<int64_t> &Offsets) {
  SmallVector<StringRef, 4> Parts;
  AttrVal.split(Parts, ',');

  for (StringRef Part : Parts) {
    Part = Part.trim();
    if (Part.empty())
      continue;

    int64_t Val;
    StringRef NumStr = Part;
    if (NumStr.starts_with("0x") || NumStr.starts_with("0X")) {
      NumStr = NumStr.drop_front(2);
      if (NumStr.getAsInteger(16, Val))
        return false;
    } else {
      if (NumStr.getAsInteger(10, Val))
        return false;
    }
    Offsets.push_back(Val);
  }
  return !Offsets.empty();
}

bool X86PreferByteParamLoadPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("prefer_byte_param_load"))
    return false;

  StringRef AttrVal =
      MF.getFunction()
          .getFnAttribute("prefer_byte_param_load")
          .getValueAsString();
  if (AttrVal.empty())
    return false;

  SmallVector<int64_t, 4> TargetOffsets;
  if (!parseOffsets(AttrVal, TargetOffsets))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Use LivePhysRegs to check EFLAGS liveness (XOR clobbers it).
    LivePhysRegs LiveRegs(*TRI);
    LiveRegs.addLiveOuts(MBB);

    // Collect candidates in reverse order (for liveness tracking).
    SmallVector<MachineInstr *, 4> ToConvert;

    for (auto I = MBB.rbegin(), E = MBB.rend(); I != E; ++I) {
      MachineInstr &MI = *I;

      if (MI.getOpcode() == X86::MOV32rm) {
        // Check if this is a load from ESP + one of the target offsets.
        // MOV32rm operands: def reg, base, scale, index, disp, segment
        // Operand layout: 0=dst, 1=base, 2=scale, 3=index, 4=disp, 5=segment
        if (MI.getNumOperands() >= 6 &&
            MI.getOperand(1).isReg() &&
            MI.getOperand(1).getReg() == X86::ESP &&
            MI.getOperand(3).isReg() &&
            MI.getOperand(3).getReg() == X86::NoRegister &&
            MI.getOperand(4).isImm()) {
          int64_t Disp = MI.getOperand(4).getImm();

          bool IsTarget = false;
          for (int64_t Off : TargetOffsets) {
            if (Disp == Off) {
              IsTarget = true;
              break;
            }
          }

          if (IsTarget) {
            Register DstReg = MI.getOperand(0).getReg();
            Register SubReg = get8BitSubReg(DstReg);
            if (SubReg != X86::NoRegister &&
                !LiveRegs.contains(X86::EFLAGS)) {
              ToConvert.push_back(&MI);
            }
          }
        }
      }

      LiveRegs.stepBackward(MI);
    }

    // Convert collected instructions.
    for (MachineInstr *MI : ToConvert) {
      Register DstReg = MI->getOperand(0).getReg();
      Register DstSubReg = get8BitSubReg(DstReg);
      DebugLoc DL = MI->getDebugLoc();

      // Insert XOR32rr_REV to clear the full 32-bit register.
      BuildMI(MBB, *MI, DL, TII->get(X86::XOR32rr_REV), DstReg)
          .addReg(DstReg, RegState::Undef)
          .addReg(DstReg, RegState::Undef);

      // Insert MOV8rm to load the low byte from the same stack location.
      auto MIB = BuildMI(MBB, *MI, DL, TII->get(X86::MOV8rm), DstSubReg);
      // Copy memory operands: base, scale, index, disp, segment
      for (unsigned i = 1; i < MI->getNumOperands(); ++i)
        MIB.add(MI->getOperand(i));
      MIB.cloneMemRefs(*MI);

      LLVM_DEBUG(dbgs() << "PreferByteParamLoad: converted MOV32rm to "
                        << "XOR+MOV8rm at ESP+"
                        << MI->getOperand(4).getImm()
                        << " in " << MF.getName() << "\n");

      MI->eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferByteParamLoadPass() {
  return new X86PreferByteParamLoadPass();
}

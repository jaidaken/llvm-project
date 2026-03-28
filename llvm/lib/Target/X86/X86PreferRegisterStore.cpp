//===--- X86PreferRegisterStore.cpp - Materialize imm in reg then store ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: MSVC 6.0 materializes repeated immediate values in a register
// (typically EAX) before storing, rather than using immediate stores.
//
// For zero stores:
//   xor eax, eax           ; 2 bytes (or 2 bytes for reversed encoding)
//   mov [ecx+0x2c], eax    ; 3-6 bytes per store
//   mov [ecx+0x28], eax    ; ...
//
// For non-zero repeated stores (e.g., 0x3f800000 = 1.0f):
//   mov eax, 0x3f800000    ; 5 bytes
//   mov [ecx+0x20], eax    ; 3-6 bytes per store
//   mov [ecx+0x10], eax    ; ...
//
// LLVM uses immediate stores:
//   mov dword ptr [ecx+0x2c], 0         ; 7 bytes
//   mov dword ptr [ecx+0x20], 3f800000h ; 7 bytes
//
// This pass finds groups of 2+ MOV32mi instructions with the same immediate
// value, inserts a register materialization (XOR for zero, MOV32ri for
// non-zero) before the first store, and replaces all MOV32mi in the group
// with MOV32mr using EAX.
//
// Gate: function attribute "prefer_register_store".
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-register-store"

namespace {

class X86PreferRegisterStorePass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferRegisterStorePass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 prefer register store for repeated immediates";
  }
};

} // end anonymous namespace

char X86PreferRegisterStorePass::ID = 0;

/// Check whether MI is a MOV32mi (32-bit immediate store to memory).
static bool isImmStore32(const MachineInstr &MI) {
  if (MI.getOpcode() != X86::MOV32mi)
    return false;
  if (MI.getNumOperands() < 6)
    return false;
  return MI.getOperand(5).isImm();
}

/// Return the immediate value from a MOV32mi instruction (operand 5).
static int64_t getStoreImm(const MachineInstr &MI) {
  return MI.getOperand(5).getImm();
}

/// Return the base register of a MOV32mi instruction (operand 0).
static Register getBaseReg(const MachineInstr &MI) {
  return MI.getOperand(0).getReg();
}

bool X86PreferRegisterStorePass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("prefer_register_store"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  bool UseXorRev =
      MF.getFunction().hasFnAttribute(Attribute::AttrKind::XOR32rr_REV);
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    auto I = MBB.begin();
    while (I != MBB.end()) {
      MachineInstr &FirstMI = *I;
      if (!isImmStore32(FirstMI)) {
        ++I;
        continue;
      }

      // Found an immediate store. Collect consecutive MOV32mi instructions
      // with the same immediate value and same base register.
      Register BaseReg = getBaseReg(FirstMI);
      int64_t ImmVal = getStoreImm(FirstMI);
      SmallVector<MachineInstr *, 16> Group;
      Group.push_back(&FirstMI);

      auto J = std::next(I);
      while (J != MBB.end() && isImmStore32(*J) &&
             getBaseReg(*J) == BaseReg && getStoreImm(*J) == ImmVal) {
        Group.push_back(&*J);
        ++J;
      }

      // Need at least 2 stores to justify register materialization.
      if (Group.size() < 2) {
        I = J;
        continue;
      }

      // Verify EAX does not overlap with the base register or any index
      // register used in the group.
      bool EaxSafe = !TRI->regsOverlap(X86::EAX, BaseReg);
      if (EaxSafe) {
        for (MachineInstr *MI : Group) {
          const MachineOperand &IndexOp = MI->getOperand(2);
          if (IndexOp.isReg() && IndexOp.getReg() != 0 &&
              TRI->regsOverlap(X86::EAX, IndexOp.getReg())) {
            EaxSafe = false;
            break;
          }
        }
      }

      if (!EaxSafe) {
        I = J;
        continue;
      }

      // Insert the materialization instruction before the first store.
      DebugLoc DL = Group[0]->getDebugLoc();
      if (ImmVal == 0) {
        // XOR32rr EAX, EAX (clobbers EFLAGS). Use reversed encoding if
        // the XOR32rr_REV attribute is set.
        unsigned XorOpc = UseXorRev ? X86::XOR32rr_REV : X86::XOR32rr;
        BuildMI(MBB, *Group[0], DL, TII->get(XorOpc), X86::EAX)
            .addReg(X86::EAX, RegState::Undef)
            .addReg(X86::EAX, RegState::Undef);
      } else {
        // MOV32ri EAX, imm.
        BuildMI(MBB, *Group[0], DL, TII->get(X86::MOV32ri), X86::EAX)
            .addImm(ImmVal);
      }

      // Replace each MOV32mi with MOV32mr using EAX.
      for (MachineInstr *MI : Group) {
        DebugLoc StoreDL = MI->getDebugLoc();
        auto MIB = BuildMI(MBB, *MI, StoreDL, TII->get(X86::MOV32mr));
        // Copy memory operands (operands 0..4: base, scale, index, disp, seg).
        for (unsigned i = 0; i < 5; ++i)
          MIB.add(MI->getOperand(i));
        // Add EAX as the source register.
        MIB.addReg(X86::EAX);
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

FunctionPass *llvm::createX86PreferRegisterStorePass() {
  return new X86PreferRegisterStorePass();
}

//===--- X86SplitWordStores.cpp - Split merged 32-bit word stores ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: The compiler sometimes merges two adjacent 16-bit stores into a
// single 32-bit store. For example:
//
//   mov word [ecx+0x118], 0x14
//   mov word [ecx+0x11a], 0x24
//
// becomes:
//
//   mov dword [ecx+0x118], 0x00240014
//
// This pass reverses that optimization by splitting MOV32mi instructions
// (store immediate to memory) back into two MOV16mi instructions when the
// 32-bit immediate has distinct low and high 16-bit halves.
//
// Gate: function attribute "split_word_stores".
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-split-word-stores"

namespace {

class X86SplitWordStoresPass : public MachineFunctionPass {
public:
  static char ID;
  X86SplitWordStoresPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 split merged 32-bit word stores";
  }
};

} // end anonymous namespace

char X86SplitWordStoresPass::ID = 0;

bool X86SplitWordStoresPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("split_word_stores"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E; /* updated in body */) {
      MachineInstr &MI = *I;

      // Only match MOV32mi (store 32-bit immediate to memory).
      if (MI.getOpcode() != X86::MOV32mi) {
        ++I;
        continue;
      }

      // MOV32mi operands: [0]=base, [1]=scale, [2]=index, [3]=disp,
      //                   [4]=segment, [5]=immediate
      if (MI.getNumOperands() < 6 || !MI.getOperand(5).isImm()) {
        ++I;
        continue;
      }

      int64_t Imm = MI.getOperand(5).getImm();
      uint32_t UImm = static_cast<uint32_t>(Imm);
      uint16_t Lo16 = static_cast<uint16_t>(UImm & 0xFFFF);
      uint16_t Hi16 = static_cast<uint16_t>((UImm >> 16) & 0xFFFF);

      // Only split when the low and high halves are distinct (indicating a
      // merged pair of different 16-bit stores). If both halves are the same,
      // the original was likely a genuine 32-bit store, not a merged pair.
      if (Lo16 == Hi16) {
        ++I;
        continue;
      }

      // Also skip if either half is zero and the other is not - this could
      // be a legitimate 32-bit immediate. Only split when BOTH halves are
      // nonzero, strongly suggesting two separate 16-bit stores were merged.
      if (Lo16 == 0 || Hi16 == 0) {
        ++I;
        continue;
      }

      // Extract memory operands from the original instruction.
      const MachineOperand &BaseOp = MI.getOperand(0);
      const MachineOperand &ScaleOp = MI.getOperand(1);
      const MachineOperand &IndexOp = MI.getOperand(2);
      const MachineOperand &DispOp = MI.getOperand(3);
      const MachineOperand &SegOp = MI.getOperand(4);

      // The displacement must be an immediate (not a global/symbol) for us
      // to safely add +2 for the high-half store.
      if (!DispOp.isImm()) {
        ++I;
        continue;
      }

      int64_t Disp = DispOp.getImm();
      DebugLoc DL = MI.getDebugLoc();

      // Build: MOV16mi [base+scale*index+disp], Lo16
      auto LoMI = BuildMI(MBB, MI, DL, TII->get(X86::MOV16mi));
      LoMI.add(BaseOp);
      LoMI.add(ScaleOp);
      LoMI.add(IndexOp);
      LoMI.addImm(Disp);
      LoMI.add(SegOp);
      LoMI.addImm(static_cast<int64_t>(Lo16));
      LoMI.cloneMemRefs(MI);

      // Build: MOV16mi [base+scale*index+disp+2], Hi16
      auto HiMI = BuildMI(MBB, MI, DL, TII->get(X86::MOV16mi));
      HiMI.add(BaseOp);
      HiMI.add(ScaleOp);
      HiMI.add(IndexOp);
      HiMI.addImm(Disp + 2);
      HiMI.add(SegOp);
      HiMI.addImm(static_cast<int64_t>(Hi16));
      HiMI.cloneMemRefs(MI);

      // Remove the original MOV32mi.
      auto Next = std::next(I);
      MI.eraseFromParent();
      I = Next;
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86SplitWordStoresPass() {
  return new X86SplitWordStoresPass();
}

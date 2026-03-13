//===------- X86SuppressFPImm.cpp - Convert FLDZ/FLD1 to const pool -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bw1-decomp: This pass converts LD_Fp0xx (fldz) and LD_Fp1xx (fld1) pseudo
// instructions into constant pool memory loads for functions with the
// SuppressFPImm attribute. This matches MSVC 6.0's code generation pattern of
// "fld dword ptr [_rdata_float0p0]" instead of LLVM's "fldz" instruction.
//
// Must run before X86FloatingPointStackifierPass which lowers LD_Fp0xx to LD_F0.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrBuilder.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Type.h"
using namespace llvm;

#define DEBUG_TYPE "x86-suppress-fp-imm"
#define X86_SUPPRESS_FP_IMM_NAME "X86 FP immediate suppression pass"

namespace {
class X86SuppressFPImmPass : public MachineFunctionPass {
public:
  static char ID;
  X86SuppressFPImmPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return X86_SUPPRESS_FP_IMM_NAME; }
};
} // end anonymous namespace

char X86SuppressFPImmPass::ID = 0;

/// Get the float value and type for an FP immediate pseudo instruction.
/// Returns true if the instruction is an FP immediate we handle.
static bool getFPImmInfo(unsigned Opcode, double &Val, unsigned &LoadOpcode,
                         Type *&Ty, LLVMContext &Ctx) {
  switch (Opcode) {
  case X86::LD_Fp032:
    Val = 0.0;
    LoadOpcode = X86::LD_Fp32m;
    Ty = Type::getFloatTy(Ctx);
    return true;
  case X86::LD_Fp132:
    Val = 1.0;
    LoadOpcode = X86::LD_Fp32m;
    Ty = Type::getFloatTy(Ctx);
    return true;
  case X86::LD_Fp064:
    Val = 0.0;
    LoadOpcode = X86::LD_Fp64m;
    Ty = Type::getDoubleTy(Ctx);
    return true;
  case X86::LD_Fp164:
    Val = 1.0;
    LoadOpcode = X86::LD_Fp64m;
    Ty = Type::getDoubleTy(Ctx);
    return true;
  case X86::LD_Fp080:
    Val = 0.0;
    LoadOpcode = X86::LD_Fp80m;
    Ty = Type::getX86_FP80Ty(Ctx);
    return true;
  case X86::LD_Fp180:
    Val = 1.0;
    LoadOpcode = X86::LD_Fp80m;
    Ty = Type::getX86_FP80Ty(Ctx);
    return true;
  default:
    return false;
  }
}

bool X86SuppressFPImmPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::SuppressFPImm))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  MachineConstantPool *MCP = MF.getConstantPool();
  LLVMContext &Ctx = MF.getFunction().getContext();
  const DataLayout &DL = MF.getDataLayout();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    SmallVector<MachineInstr *, 4> ToReplace;

    for (MachineInstr &MI : MBB) {
      double Val;
      unsigned LoadOpcode;
      Type *Ty;
      if (getFPImmInfo(MI.getOpcode(), Val, LoadOpcode, Ty, Ctx))
        ToReplace.push_back(&MI);
    }

    for (MachineInstr *MI : ToReplace) {
      double Val;
      unsigned LoadOpcode;
      Type *Ty;
      getFPImmInfo(MI->getOpcode(), Val, LoadOpcode, Ty, Ctx);

      // Create constant pool entry for the float value.
      const Constant *C = ConstantFP::get(Ty, Val);
      Align Alignment = DL.getPrefTypeAlign(Ty);
      unsigned CPI = MCP->getConstantPoolIndex(C, Alignment);

      // Get the destination register from the original instruction.
      Register DstReg = MI->getOperand(0).getReg();
      DebugLoc DL = MI->getDebugLoc();

      // Build the memory load instruction.
      // For 32-bit non-PIC: base=0, scale=1, index=0, disp=CPI, segment=0
      BuildMI(MBB, *MI, DL, TII->get(LoadOpcode), DstReg)
          .addReg(0)               // base
          .addImm(1)               // scale
          .addReg(0)               // index
          .addConstantPoolIndex(CPI) // displacement
          .addReg(0);              // segment

      MI->eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86SuppressFPImmPass() {
  return new X86SuppressFPImmPass();
}

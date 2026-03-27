//===--- X86Msvc6EvalOrder.cpp - Swap param loads for RHS-first eval ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// deusex-decomp: MSVC 6.0 evaluates assignment RHS before LHS. For copy
// functions like *(int*)dst = *(int*)src, MSVC loads the source pointer
// first, then the destination pointer:
//
//   mov eax, [esp+8]     ; src (second param, loaded first)
//   mov edx, [esp+4]     ; dst (first param, loaded second)
//   mov ecx, [eax]       ; deref src
//   mov [edx], ecx       ; store to dst
//
// Clang loads parameters in declaration order:
//
//   mov eax, [esp+4]     ; dst (first param, loaded first)
//   mov ecx, [esp+8]     ; src (second param)
//   mov ecx, [ecx]       ; deref src
//   mov [eax], ecx       ; store to dst
//
// This pass detects two consecutive ESP-relative loads at function entry
// where the first load's register is used as a store base and the second
// load's register is used as a load base (or vice versa), and swaps them.
// When swapping, it also swaps all subsequent uses of the two registers
// throughout the function.
//
// Gate: function attribute "msvc6_eval_order".
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-msvc6-eval-order"

namespace {
class X86Msvc6EvalOrderPass : public MachineFunctionPass {
public:
  static char ID;
  X86Msvc6EvalOrderPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 MSVC 6.0 RHS-first evaluation order";
  }
};
} // end anonymous namespace

char X86Msvc6EvalOrderPass::ID = 0;

static bool isEspLoad(const MachineInstr &MI) {
  if (MI.getOpcode() != X86::MOV32rm)
    return false;
  // MOV32rm: dst(0), base(1), scale(2), index(3), disp(4), seg(5)
  return MI.getOperand(1).isReg() && MI.getOperand(1).getReg() == X86::ESP;
}

/// Swap two physical registers throughout all instructions in the function,
/// skipping the two parameter load instructions themselves.
static unsigned swapRegs(unsigned Reg, unsigned RegA, unsigned RegB) {
  if (Reg == RegA) return RegB;
  if (Reg == RegB) return RegA;
  // Handle sub-registers for common pairs
  // EAX<->EDX
  if ((RegA == X86::EAX && RegB == X86::EDX) ||
      (RegA == X86::EDX && RegB == X86::EAX)) {
    if (Reg == X86::EAX) return X86::EDX;
    if (Reg == X86::EDX) return X86::EAX;
    if (Reg == X86::AX)  return X86::DX;
    if (Reg == X86::DX)  return X86::AX;
    if (Reg == X86::AL)  return X86::DL;
    if (Reg == X86::DL)  return X86::AL;
    if (Reg == X86::AH)  return X86::DH;
    if (Reg == X86::DH)  return X86::AH;
  }
  // EAX<->ECX
  if ((RegA == X86::EAX && RegB == X86::ECX) ||
      (RegA == X86::ECX && RegB == X86::EAX)) {
    if (Reg == X86::EAX) return X86::ECX;
    if (Reg == X86::ECX) return X86::EAX;
    if (Reg == X86::AX)  return X86::CX;
    if (Reg == X86::CX)  return X86::AX;
    if (Reg == X86::AL)  return X86::CL;
    if (Reg == X86::CL)  return X86::AL;
    if (Reg == X86::AH)  return X86::CH;
    if (Reg == X86::CH)  return X86::AH;
  }
  // ECX<->EDX
  if ((RegA == X86::ECX && RegB == X86::EDX) ||
      (RegA == X86::EDX && RegB == X86::ECX)) {
    if (Reg == X86::ECX) return X86::EDX;
    if (Reg == X86::EDX) return X86::ECX;
    if (Reg == X86::CX)  return X86::DX;
    if (Reg == X86::DX)  return X86::CX;
    if (Reg == X86::CL)  return X86::DL;
    if (Reg == X86::DL)  return X86::CL;
    if (Reg == X86::CH)  return X86::DH;
    if (Reg == X86::DH)  return X86::CH;
  }
  return Reg;
}

bool X86Msvc6EvalOrderPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("msvc6_eval_order"))
    return false;

  MachineBasicBlock &EntryMBB = MF.front();

  // Find the first two instructions: both must be ESP-relative loads.
  auto I = EntryMBB.begin();
  auto E = EntryMBB.end();

  // Skip pseudo/debug instructions
  while (I != E && (I->isPseudo() || I->isDebugInstr()))
    ++I;
  if (I == E || !isEspLoad(*I))
    return false;
  MachineInstr *Load1 = &*I;
  ++I;

  while (I != E && (I->isPseudo() || I->isDebugInstr()))
    ++I;
  if (I == E || !isEspLoad(*I))
    return false;
  MachineInstr *Load2 = &*I;

  // Get the ESP displacements
  int64_t Disp1 = Load1->getOperand(4).getImm();
  int64_t Disp2 = Load2->getOperand(4).getImm();

  // MSVC loads the higher-offset parameter first (RHS = second param = higher ESP offset).
  // If Load1 already has the higher offset, no swap needed.
  if (Disp1 >= Disp2)
    return false;

  // Load1 has lower offset (first param = dst), Load2 has higher (second param = src).
  //
  // MSVC 6.0 pattern:
  //   mov eax, [esp+8]   ; src in EAX (higher offset loaded first)
  //   mov edx, [esp+4]   ; dst in EDX
  //
  // Compiler pattern:
  //   mov REG1, [esp+4]  ; dst in REG1
  //   mov REG2, [esp+8]  ; src in REG2
  //
  // Transform: rewrite Load2 to use EAX, Load1 to use EDX, swap order,
  // and rename all REG1->EDX, REG2->EAX in subsequent code.
  Register Reg1 = Load1->getOperand(0).getReg();  // compiler's dst reg
  Register Reg2 = Load2->getOperand(0).getReg();  // compiler's src reg

  // Swap the two loads' positions
  EntryMBB.splice(MachineBasicBlock::iterator(Load1), &EntryMBB,
                  MachineBasicBlock::iterator(Load2));

  // Rewrite Load2 (src, now first) dest to EAX
  Load2->getOperand(0).setReg(X86::EAX);
  // Rewrite Load1 (dst, now second) dest to EDX
  Load1->getOperand(0).setReg(X86::EDX);

  // Rename throughout the rest of the function:
  // Old Reg1 (was dst) -> EDX
  // Old Reg2 (was src) -> EAX
  // We need a 2-step rename to avoid conflicts: Reg1->EDX and Reg2->EAX.
  // If Reg1==EAX and Reg2==ECX: EAX->EDX, ECX->EAX (no conflict)
  // If Reg1==EAX and Reg2==EDX: EAX->EDX, EDX->EAX (swap, handled)
  bool Changed = true;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (&MI == Load1 || &MI == Load2)
        continue;
      for (MachineOperand &MO : MI.operands()) {
        if (!MO.isReg())
          continue;
        unsigned R = MO.getReg();
        // Map old Reg1 (and sub-regs) to EDX
        unsigned NewR = R;
        if (Reg1 == X86::EAX) {
          if (R == X86::EAX) NewR = X86::EDX;
          else if (R == X86::AX) NewR = X86::DX;
          else if (R == X86::AL) NewR = X86::DL;
          else if (R == X86::AH) NewR = X86::DH;
        } else if (Reg1 == X86::ECX) {
          if (R == X86::ECX) NewR = X86::EDX;
          else if (R == X86::CX) NewR = X86::DX;
          else if (R == X86::CL) NewR = X86::DL;
          else if (R == X86::CH) NewR = X86::DH;
        } else if (Reg1 == X86::EDX) {
          // Already EDX, no change for this mapping
        }
        // Map old Reg2 (and sub-regs) to EAX
        if (Reg2 == X86::ECX) {
          if (R == X86::ECX) NewR = X86::EAX;
          else if (R == X86::CX) NewR = X86::AX;
          else if (R == X86::CL) NewR = X86::AL;
          else if (R == X86::CH) NewR = X86::AH;
        } else if (Reg2 == X86::EAX) {
          if (R == X86::EAX && NewR == R) NewR = X86::EAX; // already EAX
        } else if (Reg2 == X86::EDX) {
          if (R == X86::EDX && NewR == R) NewR = X86::EAX;
          else if (R == X86::DX && NewR == R) NewR = X86::AX;
          else if (R == X86::DL && NewR == R) NewR = X86::AL;
          else if (R == X86::DH && NewR == R) NewR = X86::AH;
        }
        if (NewR != R) {
          MO.setReg(NewR);
          Changed = true;
        }
      }
    }
  }

  // Fix self-deref: after the rename, the compiler may generate
  // "mov eax, [eax]" (load through a register into itself). MSVC uses ECX
  // as a shuttle register instead. Find this pattern and rewrite to ECX.
  //
  // Pattern: MOV32rm EAX, [EAX+disp] -> MOV32rm ECX, [EAX+disp]
  //          MOV32mr [EDX+disp], EAX  -> MOV32mr [EDX+disp], ECX
  // Also:    MOV8rm  AL,  [EAX+disp] -> MOV8rm  CL,  [EAX+disp]
  //          MOV8mr  [EDX+disp], AL   -> MOV8mr  [EDX+disp], CL
  for (MachineBasicBlock &MBB : MF) {
    for (auto MI = MBB.begin(), ME = MBB.end(); MI != ME; ++MI) {
      if (&*MI == Load1 || &*MI == Load2)
        continue;

      unsigned Opc = MI->getOpcode();
      bool is32 = (Opc == X86::MOV32rm);
      bool is8 = (Opc == X86::MOV8rm);
      if (!is32 && !is8)
        continue;

      // Check for self-deref: dest == base
      Register Dst = MI->getOperand(0).getReg();
      Register Base = MI->getOperand(1).getReg();
      if (Dst != Base)
        continue;

      // Pick shuttle register: ECX for 32-bit, CL for 8-bit
      Register NewDst = is32 ? X86::ECX : X86::CL;

      // Change the load destination
      MI->getOperand(0).setReg(NewDst);

      // Find and update the immediately following store
      auto Next = std::next(MI);
      while (Next != ME && (Next->isPseudo() || Next->isDebugInstr()))
        ++Next;
      if (Next != ME) {
        unsigned StoreOpc = Next->getOpcode();
        if ((is32 && StoreOpc == X86::MOV32mr) ||
            (is8 && StoreOpc == X86::MOV8mr)) {
          // The store's source register (last explicit operand) should match old Dst
          MachineOperand &SrcOp =
              Next->getOperand(Next->getNumExplicitOperands() - 1);
          if (SrcOp.isReg() && SrcOp.getReg() == Dst) {
            SrcOp.setReg(NewDst);
            Changed = true;
          }
        }
      }
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86Msvc6EvalOrderPass() {
  return new X86Msvc6EvalOrderPass();
}

// bw1-decomp: Decompose IMUL with specific constants into LEA/SHL/SUB chains.
//
// MSVC 6.0 decomposes certain multiplications into strength-reduced sequences
// using LEA, SHL, and SUB. For example, x * 276 when x is in ECX becomes:
//   lea eax, [ecx+ecx*2]   ; eax = x*3       (scratch = src*3)
//   shl eax, 3             ; eax = x*24      (scratch <<= 3)
//   sub eax, ecx           ; eax = x*23      (scratch -= src)
//   lea ecx, [eax+eax*2]   ; ecx = x*69      (src = scratch*3)
//   shl ecx, 2             ; ecx = x*276     (src <<= 2)
//
// Register roles: the source register holds the input value and receives the
// final result. The scratch register is the OTHER register from {EAX, ECX}.
//   src=ECX -> scratch=EAX, result in ECX
//   src=EAX -> scratch=ECX, result in EAX
//
// When dest != src (e.g., imul EAX, ECX, 276), the dest register is used as
// scratch and the result lands in src. A final MOV copies the result from src
// back to dest so that subsequent code sees the right value.
//
// Clang generates a single IMUL instruction instead, producing different bytes.
//
// This pass finds IMUL32rri / IMUL32rri8 and for known constants replaces
// the IMUL with the equivalent LEA/SHL/SUB decomposition.
//
// Gate: function attribute "decompose_imul".

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

namespace {
class X86DecomposeImulPass : public MachineFunctionPass {
public:
  static char ID;
  X86DecomposeImulPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 MSVC 6.0 IMUL decomposition into LEA/SHL/SUB";
  }

private:
  const X86InstrInfo *TII = nullptr;

  bool decompose276(MachineBasicBlock &MBB, MachineInstr &MI,
                    Register DstReg, Register SrcReg);
};
char X86DecomposeImulPass::ID = 0;
} // namespace

/// Build: lea DstReg, [BaseReg + IndexReg * Scale]
/// LEA32r operands: dst, base, scale, index, disp, segment
static MachineInstrBuilder buildLEA32_scaled(MachineBasicBlock &MBB,
                                             MachineBasicBlock::iterator InsertPt,
                                             const DebugLoc &DL,
                                             const X86InstrInfo *TII,
                                             Register DstReg,
                                             Register BaseReg,
                                             unsigned Scale,
                                             Register IndexReg) {
  return BuildMI(MBB, InsertPt, DL, TII->get(X86::LEA32r), DstReg)
      .addReg(BaseReg)
      .addImm(Scale)
      .addReg(IndexReg)
      .addImm(0)
      .addReg(0);
}

/// Decompose x * 276 = ((x*3)<<3 - x) * 3 * 4
///
/// The scratch register is always the OTHER register from {EAX, ECX}:
///   src=ECX: lea eax,[ecx+ecx*2]; shl eax,3; sub eax,ecx;
///            lea ecx,[eax+eax*2]; shl ecx,2
///   src=EAX: lea ecx,[eax+eax*2]; shl ecx,3; sub ecx,eax;
///            lea eax,[ecx+ecx*2]; shl eax,2
///
/// When dest != src, DstReg is used as scratch and the result lands in SrcReg.
/// A final MOV32rr copies the result back to DstReg.
bool X86DecomposeImulPass::decompose276(MachineBasicBlock &MBB,
                                        MachineInstr &MI,
                                        Register DstReg,
                                        Register SrcReg) {
  DebugLoc DL = MI.getDebugLoc();
  MachineBasicBlock::iterator InsertPt = MI.getIterator();

  // Determine scratch and result registers.
  // When dest == src, scratch is the other register from {EAX, ECX}.
  // When dest != src, dest is used as scratch and src receives the result.
  Register ScratchReg;
  Register ResultReg;
  if (DstReg == SrcReg) {
    ScratchReg = (SrcReg == X86::EAX) ? X86::ECX : X86::EAX;
    ResultReg = SrcReg;
  } else {
    ScratchReg = DstReg;
    ResultReg = SrcReg;
  }

  // lea scratch, [src+src*2]   ; scratch = x*3
  buildLEA32_scaled(MBB, InsertPt, DL, TII, ScratchReg, SrcReg, 2, SrcReg);

  // shl scratch, 3             ; scratch = x*24
  BuildMI(MBB, InsertPt, DL, TII->get(X86::SHL32ri), ScratchReg)
      .addReg(ScratchReg)
      .addImm(3);

  // sub scratch, src           ; scratch = x*23
  BuildMI(MBB, InsertPt, DL, TII->get(X86::SUB32rr), ScratchReg)
      .addReg(ScratchReg)
      .addReg(SrcReg);

  // lea result, [scratch+scratch*2]   ; result = x*69
  buildLEA32_scaled(MBB, InsertPt, DL, TII, ResultReg, ScratchReg, 2, ScratchReg);

  // shl result, 2              ; result = x*276
  BuildMI(MBB, InsertPt, DL, TII->get(X86::SHL32ri), ResultReg)
      .addReg(ResultReg)
      .addImm(2);

  // When dest != src, result is in SrcReg but the IMUL destination was DstReg.
  // Emit a MOV to place the result where downstream code expects it.
  if (DstReg != SrcReg) {
    BuildMI(MBB, InsertPt, DL, TII->get(X86::MOV32rr), DstReg)
        .addReg(ResultReg);
  }

  MI.eraseFromParent();
  return true;
}

bool X86DecomposeImulPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("decompose_imul"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E; /*in loop*/) {
      MachineInstr &MI = *I++;
      unsigned Opc = MI.getOpcode();

      // Match IMUL32rri and IMUL32rri8 (three-operand immediate multiply).
      if (Opc != X86::IMUL32rri && Opc != X86::IMUL32rri8)
        continue;

      // Operands: [0]=dst, [1]=src, [2]=imm
      Register DstReg = MI.getOperand(0).getReg();
      Register SrcReg = MI.getOperand(1).getReg();

      int64_t ImmVal = MI.getOperand(2).getImm();

      switch (ImmVal) {
      case 276:
        Changed |= decompose276(MBB, MI, DstReg, SrcReg);
        break;
      default:
        // Unknown constant, leave as IMUL.
        break;
      }
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86DecomposeImulPass() {
  return new X86DecomposeImulPass();
}

// bw1-decomp: Convert TEST8mi (test byte [mem], imm) to MOV32rm + TEST8ri AH
// for functions with the "prefer_test_ah" attribute.
//
// Also converts TEST32ri reg, 0x0000NN00 to TEST8ri high_byte(reg), 0xNN
// for functions with the "prefer_test_ah_reg" attribute.
//
// MSVC 6.0 tests bit flags by loading a full dword into a register, then
// testing the high byte:
//   mov eax, [ecx+4]    ; load full dword
//   test ah, 0x80        ; test bit 15 (F6 C4 80)
//
// Clang instead generates a direct memory test on the high byte:
//   test byte ptr [ecx+5], 0x80  ; (F6 41 05 80)
//
// This pass converts the Clang form back to the MSVC form by recognizing
// TEST8mi instructions where the displacement is N+1 (testing the high byte
// of a value at offset N) and replacing them with:
//   MOV32rm EAX, [base + N]
//   TEST8ri AH, imm
//
// The attribute value is a semicolon-separated list of entries:
//   "offset:reg"  e.g. "5:eax;9:ecx"
// Each entry specifies the displacement of the TEST8mi and which 32-bit
// register to use for the load (eax or ecx). The load displacement is
// automatically computed as (offset - 1).
//
// Register test mode ("prefer_test_ah_reg" attribute):
//   MSVC 6.0:  test ah, 0x80  (F6 C4 80 - 3 bytes)
//   Clang:     test eax, 0x00008000  (A9 00 80 00 00 - 5 bytes)
//
// When the immediate in TEST32ri has set bits only in the second byte
// (0x0000NN00), the instruction is replaced with TEST8ri on the high-byte
// sub-register (AH, CH, DH, BH).

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "x86-prefer-test-ah"

namespace {

struct TestAhEntry {
  int64_t TestDisp;    // displacement in the TEST8mi to match
  Register LoadReg32;  // 32-bit register for the MOV (EAX or ECX)
  Register TestReg8;   // high-byte register for the TEST (AH or CH)
};

class X86PreferTestAhPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferTestAhPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 prefer TEST AH for high-byte tests (MSVC 6.0)";
  }

private:
  static bool parseEntries(StringRef AttrVal,
                           SmallVectorImpl<TestAhEntry> &Entries);
  static Register getHighByteReg(Register Reg32);
  bool convertTestMemEntries(MachineFunction &MF, const X86InstrInfo *TII,
                             SmallVectorImpl<TestAhEntry> &Entries);
  bool convertTestRegImm(MachineFunction &MF, const X86InstrInfo *TII);
};

} // end anonymous namespace

char X86PreferTestAhPass::ID = 0;

bool X86PreferTestAhPass::parseEntries(
    StringRef AttrVal, SmallVectorImpl<TestAhEntry> &Entries) {
  // Parse "disp:reg;disp:reg;..." format.
  // Example: "5:eax;9:ecx"
  SmallVector<StringRef, 4> Parts;
  AttrVal.split(Parts, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);

  for (StringRef Part : Parts) {
    auto [DispStr, RegStr] = Part.split(':');
    int64_t Disp;
    if (DispStr.getAsInteger(10, Disp))
      return false;

    Register Reg32, Reg8;
    if (RegStr.equals_insensitive("eax")) {
      Reg32 = X86::EAX;
      Reg8 = X86::AH;
    } else if (RegStr.equals_insensitive("ecx")) {
      Reg32 = X86::ECX;
      Reg8 = X86::CH;
    } else if (RegStr.equals_insensitive("edx")) {
      Reg32 = X86::EDX;
      Reg8 = X86::DH;
    } else {
      return false;
    }

    Entries.push_back({Disp, Reg32, Reg8});
  }
  return !Entries.empty();
}

/// Map a 32-bit register to its high-byte sub-register.
/// Returns Register() (invalid) for registers without a high-byte form.
Register X86PreferTestAhPass::getHighByteReg(Register Reg32) {
  switch (Reg32) {
  case X86::EAX: return X86::AH;
  case X86::ECX: return X86::CH;
  case X86::EDX: return X86::DH;
  case X86::EBX: return X86::BH;
  default: return Register();
  }
}

bool X86PreferTestAhPass::convertTestMemEntries(
    MachineFunction &MF, const X86InstrInfo *TII,
    SmallVectorImpl<TestAhEntry> &Entries) {
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E;) {
      MachineInstr &MI = *I;

      if (MI.getOpcode() != X86::TEST8mi) {
        ++I;
        continue;
      }

      // TEST8mi operands: [base(0), scale(1), index(2), disp(3), seg(4), imm(5)]
      if (MI.getNumOperands() < 6) {
        ++I;
        continue;
      }

      // Get the displacement from the memory operand.
      const MachineOperand &DispOp = MI.getOperand(3);
      if (!DispOp.isImm()) {
        ++I;
        continue;
      }
      int64_t Disp = DispOp.getImm();

      // Find a matching entry.
      const TestAhEntry *Match = nullptr;
      for (const auto &Entry : Entries) {
        if (Entry.TestDisp == Disp) {
          Match = &Entry;
          break;
        }
      }

      if (!Match) {
        ++I;
        continue;
      }

      int64_t ImmVal = MI.getOperand(5).getImm();
      DebugLoc DL = MI.getDebugLoc();

      // Build: MOV32rm Reg32, [base + (Disp - 1)]
      // The load displacement is one less than the TEST displacement because
      // the TEST was accessing the high byte (offset+1) of a word/dword.
      auto MIB = BuildMI(MBB, MI, DL, TII->get(X86::MOV32rm), Match->LoadReg32);
      MIB.add(MI.getOperand(0)); // base
      MIB.add(MI.getOperand(1)); // scale
      MIB.add(MI.getOperand(2)); // index
      MIB.addImm(Disp - 1);     // disp - 1
      MIB.add(MI.getOperand(4)); // segment
      MIB.cloneMemRefs(MI);

      // Build: TEST8ri AH/CH, imm
      BuildMI(MBB, MI, DL, TII->get(X86::TEST8ri))
          .addReg(Match->TestReg8)
          .addImm(ImmVal);

      // Remove original TEST8mi and advance.
      auto NextI = std::next(I);
      MI.eraseFromParent();
      I = NextI;
      Changed = true;
    }
  }

  return Changed;
}

bool X86PreferTestAhPass::convertTestRegImm(MachineFunction &MF,
                                             const X86InstrInfo *TII) {
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E;) {
      MachineInstr &MI = *I;

      if (MI.getOpcode() != X86::TEST32ri) {
        ++I;
        continue;
      }

      // TEST32ri operands: [reg(0), imm(1)]
      if (MI.getNumOperands() < 2) {
        ++I;
        continue;
      }

      Register SrcReg = MI.getOperand(0).getReg();
      int64_t ImmVal = MI.getOperand(1).getImm();

      // Check that set bits are entirely in the second byte (0x0000NN00).
      // Low byte must be zero and upper two bytes must be zero.
      if ((ImmVal & 0xFF) != 0 || (ImmVal & 0xFFFF0000) != 0 ||
          (ImmVal & 0xFF00) == 0) {
        ++I;
        continue;
      }

      Register HighReg = getHighByteReg(SrcReg);
      if (!HighReg.isValid()) {
        ++I;
        continue;
      }

      int64_t ShiftedImm = (ImmVal >> 8) & 0xFF;
      DebugLoc DL = MI.getDebugLoc();

      LLVM_DEBUG(dbgs() << "  Converting TEST32ri to TEST8ri AH: "
                        << MI);

      // Build: TEST8ri high_byte(reg), shifted_imm
      BuildMI(MBB, MI, DL, TII->get(X86::TEST8ri))
          .addReg(HighReg)
          .addImm(ShiftedImm);

      auto NextI = std::next(I);
      MI.eraseFromParent();
      I = NextI;
      Changed = true;
    }
  }

  return Changed;
}

bool X86PreferTestAhPass::runOnMachineFunction(MachineFunction &MF) {
  const Function &F = MF.getFunction();
  bool HasMemAttr = F.hasFnAttribute("prefer_test_ah");
  bool HasRegAttr = F.hasFnAttribute("prefer_test_ah_reg");

  if (!HasMemAttr && !HasRegAttr)
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  // Handle TEST8mi -> MOV32rm + TEST8ri (memory test mode).
  if (HasMemAttr) {
    StringRef AttrVal =
        F.getFnAttribute("prefer_test_ah").getValueAsString();
    SmallVector<TestAhEntry, 4> Entries;
    if (parseEntries(AttrVal, Entries))
      Changed |= convertTestMemEntries(MF, TII, Entries);
  }

  // Handle TEST32ri -> TEST8ri (register test mode).
  if (HasRegAttr)
    Changed |= convertTestRegImm(MF, TII);

  return Changed;
}

FunctionPass *llvm::createX86PreferTestAhPass() {
  return new X86PreferTestAhPass();
}

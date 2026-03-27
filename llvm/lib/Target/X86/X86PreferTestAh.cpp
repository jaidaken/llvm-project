// bw1-decomp: Convert TEST8mi (test byte [mem], imm) to MOV32rm + TEST8ri AH
// for functions with the "prefer_test_ah" attribute.
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

bool X86PreferTestAhPass::runOnMachineFunction(MachineFunction &MF) {
  const Function &F = MF.getFunction();
  if (!F.hasFnAttribute("prefer_test_ah"))
    return false;

  StringRef AttrVal =
      F.getFnAttribute("prefer_test_ah").getValueAsString();
  SmallVector<TestAhEntry, 4> Entries;
  if (!parseEntries(AttrVal, Entries))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
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

FunctionPass *llvm::createX86PreferTestAhPass() {
  return new X86PreferTestAhPass();
}

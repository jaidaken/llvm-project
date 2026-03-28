// bw1-decomp: Convert absolute-offset struct loads to base-adjust + small-offset
// loads, matching MSVC 6.0's struct copy pattern.
//
// MSVC 6.0 generates:
//   add ecx, 0x14          ; adjust base pointer
//   mov edx, [ecx]         ; load at offset 0 (actually this+0x14)
//   mov eax, [ecx+0x04]    ; load at offset 4 (actually this+0x18)
//   mov ecx, [ecx+0x08]    ; load at offset 8 (actually this+0x1c)
//
// Clang generates:
//   mov edx, [ecx+0x14]    ; direct offset
//   mov eax, [ecx+0x18]    ; direct offset
//   mov ecx, [ecx+0x1c]    ; direct offset
//
// The difference matters because [ecx] (no displacement) encodes in 2 bytes
// while [ecx+disp8] needs 3 bytes.
//
// Gate: function attribute "prefer_base_adjust".

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "x86-prefer-base-adjust"

namespace {

class X86PreferBaseAdjustPass : public MachineFunctionPass {
public:
  static char ID;
  X86PreferBaseAdjustPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 prefer base-adjust struct copy pass";
  }
};

} // end anonymous namespace

char X86PreferBaseAdjustPass::ID = 0;

/// Check if MI is a MOV32rm (32-bit register load from memory) using the given
/// base register with no index register and no segment override, and a simple
/// immediate displacement.
static bool isSimpleLoad32(const MachineInstr &MI, Register BaseReg) {
  if (MI.getOpcode() != X86::MOV32rm)
    return false;
  // MOV32rm operands: [0]=dst, [1]=base, [2]=scale, [3]=index, [4]=disp,
  //                   [5]=segment
  if (MI.getOperand(1).getReg() != BaseReg)
    return false;
  // Scale must be 1
  if (MI.getOperand(2).getImm() != 1)
    return false;
  // No index register
  if (MI.getOperand(3).getReg() != X86::NoRegister)
    return false;
  // Displacement must be an immediate
  if (!MI.getOperand(4).isImm())
    return false;
  // No segment override (or default segment)
  if (MI.getOperand(5).getReg() != X86::NoRegister)
    return false;
  return true;
}

/// Check if MI is a MOV32mr (32-bit store to memory) using the given base
/// register with no index register and a simple immediate displacement.
static bool isSimpleStore32(const MachineInstr &MI, Register BaseReg) {
  if (MI.getOpcode() != X86::MOV32mr)
    return false;
  // MOV32mr operands: [0]=base, [1]=scale, [2]=index, [3]=disp, [4]=segment,
  //                   [5]=src
  if (MI.getOperand(0).getReg() != BaseReg)
    return false;
  if (MI.getOperand(1).getImm() != 1)
    return false;
  if (MI.getOperand(2).getReg() != X86::NoRegister)
    return false;
  if (!MI.getOperand(3).isImm())
    return false;
  if (MI.getOperand(4).getReg() != X86::NoRegister)
    return false;
  return true;
}

/// Get the displacement of a MOV32rm instruction.
static int64_t getLoadDisp(const MachineInstr &MI) {
  return MI.getOperand(4).getImm();
}

/// Get the displacement of a MOV32mr instruction.
static int64_t getStoreDisp(const MachineInstr &MI) {
  return MI.getOperand(3).getImm();
}

bool X86PreferBaseAdjustPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute("prefer_base_adjust"))
    return false;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = STI.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Scan for sequences of MOV32rm/MOV32mr that share a base register and
    // have consecutive offsets sharing a common base displacement.
    for (auto I = MBB.begin(), E = MBB.end(); I != E;) {
      MachineInstr &FirstMI = *I;

      // Look for a MOV32rm or MOV32mr as the potential start of a group.
      bool FirstIsLoad = (FirstMI.getOpcode() == X86::MOV32rm);
      bool FirstIsStore = (FirstMI.getOpcode() == X86::MOV32mr);
      if (!FirstIsLoad && !FirstIsStore) {
        ++I;
        continue;
      }

      // Determine the base register and first displacement.
      Register BaseReg;
      int64_t FirstDisp;
      if (FirstIsLoad) {
        BaseReg = FirstMI.getOperand(1).getReg();
        if (!isSimpleLoad32(FirstMI, BaseReg)) {
          ++I;
          continue;
        }
        FirstDisp = getLoadDisp(FirstMI);
      } else {
        BaseReg = FirstMI.getOperand(0).getReg();
        if (!isSimpleStore32(FirstMI, BaseReg)) {
          ++I;
          continue;
        }
        FirstDisp = getStoreDisp(FirstMI);
      }

      // Only consider non-zero base displacements (no point adjusting to 0).
      if (FirstDisp == 0) {
        ++I;
        continue;
      }

      // Collect consecutive loads/stores from the same base register.
      // Track the minimum displacement as the common base offset.
      struct MemOp {
        MachineInstr *MI;
        bool IsLoad;
        int64_t Disp;
      };
      SmallVector<MemOp, 8> Group;
      Group.push_back({&FirstMI, FirstIsLoad, FirstDisp});

      int64_t MinDisp = FirstDisp;
      bool BaseClobbered = false;

      // For loads, check if the dest register is the base register.
      if (FirstIsLoad && FirstMI.getOperand(0).getReg() == BaseReg)
        BaseClobbered = true;

      auto J = std::next(I);
      while (J != E && !BaseClobbered) {
        MachineInstr &MI = *J;

        bool IsLoad = false;
        bool IsStore = false;
        int64_t Disp = 0;

        if (isSimpleLoad32(MI, BaseReg)) {
          IsLoad = true;
          Disp = getLoadDisp(MI);
        } else if (isSimpleStore32(MI, BaseReg)) {
          IsStore = true;
          Disp = getStoreDisp(MI);
        } else {
          // Not a matching load or store using BaseReg as base.
          // Check if this instruction *defines* (writes to) BaseReg.
          // If it only reads BaseReg (e.g. a store that uses the loaded
          // value as its source), we can safely skip over it.
          // Also stop on branches and calls since they change control flow.
          if (MI.isCall() || MI.isBranch()) {
            break;
          }
          bool DefinesBase = false;
          for (const MachineOperand &MO : MI.operands()) {
            if (MO.isReg() && MO.isDef() && MO.getReg() == BaseReg) {
              DefinesBase = true;
              break;
            }
          }
          if (!DefinesBase) {
            ++J;
            continue;
          }
          // Defines BaseReg but is not a simple load/store from it - stop.
          break;
        }

        Group.push_back({&MI, IsLoad, Disp});
        if (Disp < MinDisp)
          MinDisp = Disp;

        // If this load clobbers the base register, it must be the last one.
        if (IsLoad && MI.getOperand(0).getReg() == BaseReg) {
          BaseClobbered = true;
          ++J;
          break;
        }

        ++J;
      }

      // Need at least 2 memory ops to justify the ADD.
      if (Group.size() < 2) {
        I = J;
        continue;
      }

      // The common base offset to subtract. Use the minimum displacement.
      int64_t BaseOffset = MinDisp;
      if (BaseOffset == 0) {
        I = J;
        continue;
      }

      // Insert ADD32ri/ADD32ri8 base, BaseOffset before the first instruction.
      DebugLoc DL = Group[0].MI->getDebugLoc();
      unsigned AddOpc = (BaseOffset >= -128 && BaseOffset <= 127)
                            ? X86::ADD32ri8
                            : X86::ADD32ri;
      BuildMI(MBB, *Group[0].MI, DL, TII->get(AddOpc), BaseReg)
          .addReg(BaseReg)
          .addImm(BaseOffset);

      // Adjust each memory op's displacement.
      for (auto &Op : Group) {
        int64_t NewDisp = Op.Disp - BaseOffset;
        if (Op.IsLoad) {
          // MOV32rm: displacement is operand 4
          Op.MI->getOperand(4).setImm(NewDisp);
        } else {
          // MOV32mr: displacement is operand 3
          Op.MI->getOperand(3).setImm(NewDisp);
        }
      }

      Changed = true;
      I = J;
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86PreferBaseAdjustPass() {
  return new X86PreferBaseAdjustPass();
}

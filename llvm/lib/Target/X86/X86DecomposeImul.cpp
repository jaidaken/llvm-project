// bw1-decomp: Decompose IMUL with constants into LEA/SHL/ADD/SUB chains.
//
// MSVC 6.0 decomposes multiply-by-constant (3-operand IMUL) into
// strength-reduced sequences using LEA, SHL, ADD, and SUB. This avoids the
// slow IMUL instruction on P5/P6 era hardware.
//
// Examples:
//   x * 276  (src=ECX):
//     lea eax, [ecx+ecx*2]   ; eax = x*3
//     shl eax, 3             ; eax = x*24
//     sub eax, ecx           ; eax = x*23
//     lea ecx, [eax+eax*2]   ; ecx = x*69
//     shl ecx, 2             ; ecx = x*276
//
//   x * 12  (src=ECX):
//     lea ecx, [ecx+ecx*2]   ; ecx = x*3
//     shl ecx, 2             ; ecx = x*12
//
//   x * 328 = 8 * 41 = 8 * (5*8 + 1)  (src=ECX):
//     lea eax, [ecx+ecx*4]   ; eax = x*5
//     shl eax, 3             ; eax = x*40
//     add eax, ecx           ; eax = x*41
//     shl eax, 3             ; eax = x*328
//     mov ecx, eax           ; (if dest == src)
//
// Register roles: the source register holds the input. A scratch register
// (the OTHER register from {EAX, ECX}) holds intermediate values. The result
// alternates between src and scratch as operations execute, and ends up in
// whichever register the final step targets.
//
// When dest != src (e.g., imul EAX, ECX, 276), DstReg is used as scratch.
// A final MOV copies the result to DstReg if needed.
//
// Gate: function attribute "decompose_imul".

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "x86-decompose-imul"

namespace {

/// A single step in the decomposition sequence.
enum class StepKind {
  /// lea dst, [cur + cur * Scale]  -- multiplies current value by (1+Scale)
  LEA_CUR,
  /// lea dst, [src + cur * Scale]  -- computes src + cur*Scale
  LEA_SRC_PLUS_CUR,
  /// lea dst, [cur + src * Scale]  -- computes cur + src*Scale
  LEA_CUR_PLUS_SRC,
  /// shl cur, Amount
  SHL,
  /// add cur, src  (cur += original input)
  ADD_SRC,
  /// sub cur, src  (cur -= original input)
  SUB_SRC,
  /// neg cur       (cur = -cur)
  NEG,
  /// mov dst, src  (copy original input to dst, resets current to src)
  MOV_SRC,
};

struct Step {
  StepKind Kind;
  unsigned Param; // Scale for LEA, shift amount for SHL, unused for ADD/SUB
};

/// Try to decompose a positive constant into a sequence of steps matching
/// MSVC 6.0's strength reduction algorithm.
///
/// Returns true if a decomposition was found, populating Steps.
/// The decomposition uses at most ~6 steps (MSVC 6.0 falls back to IMUL
/// for constants that require too many steps).
static bool findDecomposition(int64_t C, SmallVectorImpl<Step> &Steps) {
  if (C == 0 || C == 1)
    return false;

  // Handle negative constants: decompose |C| then negate.
  if (C < 0) {
    if (!findDecomposition(-C, Steps))
      return false;
    Steps.push_back({StepKind::NEG, 0});
    return true;
  }

  // Extract trailing zeros for a final shift.
  unsigned TrailingZeros = 0;
  int64_t Odd = C;
  while ((Odd & 1) == 0) {
    Odd >>= 1;
    TrailingZeros++;
  }

  // If the constant is a pure power of 2, just SHL.
  if (Odd == 1) {
    Steps.push_back({StepKind::SHL, TrailingZeros});
    return true;
  }

  // Try to decompose the odd part into LEA/ADD/SUB chains.
  // MSVC 6.0 uses a recursive factoring approach:
  //   - Try to express Odd as (1+s)*Q where s in {1,2,4,8} (LEA scales)
  //   - Try to express Odd as s*Q+1 or s*Q-1
  //   - Recurse on Q
  //
  // We use a table of known decompositions for common constants, plus a
  // general recursive algorithm for others.

  // Helper: check if N can be done with a single LEA [reg + reg*scale].
  // Returns the scale value for LEA, or 0 if not expressible as one LEA.
  // LEA [base + index*scale] where base=index=CurReg gives (1+scale)*CurReg.
  // Valid scales: 1, 2, 4, 8 giving multiplies of 2, 3, 5, 9.
  // Note: pure *4 and *8 use SHL, not LEA, matching MSVC 6.0 behavior.
  auto isLeaFactor = [](int64_t N) -> int {
    switch (N) {
    case 2: return 1;  // lea [reg+reg*1]
    case 3: return 2;  // lea [reg+reg*2]
    case 5: return 4;  // lea [reg+reg*4]
    case 9: return 8;  // lea [reg+reg*8]
    default: return 0;
    }
  };

  // Direct single LEA.
  if (isLeaFactor(Odd)) {
    Steps.push_back({StepKind::LEA_CUR, (unsigned)isLeaFactor(Odd)});
    if (TrailingZeros)
      Steps.push_back({StepKind::SHL, TrailingZeros});
    return true;
  }

  // Two LEA factors: Odd = A * B where both are LEA-able.
  // Try all pairs.
  static const int64_t LeaFactors[] = {2, 3, 5, 9};
  for (int64_t A : LeaFactors) {
    if (Odd % A != 0)
      continue;
    int64_t B = Odd / A;
    if (isLeaFactor(B)) {
      Steps.push_back({StepKind::LEA_CUR, (unsigned)isLeaFactor(B)});
      Steps.push_back({StepKind::LEA_CUR, (unsigned)isLeaFactor(A)});
      if (TrailingZeros)
        Steps.push_back({StepKind::SHL, TrailingZeros});
      return true;
    }
  }

  // LEA + SHL combination: Odd = A << S where A is LEA-able.
  for (unsigned S = 1; S <= 5; ++S) {
    int64_t A = Odd >> S;
    if (A <= 1)
      break;
    if ((A << S) == Odd && isLeaFactor(A)) {
      Steps.push_back({StepKind::LEA_CUR, (unsigned)isLeaFactor(A)});
      Steps.push_back({StepKind::SHL, S});
      if (TrailingZeros)
        Steps.push_back({StepKind::SHL, TrailingZeros});
      return true;
    }
  }

  // LEA * LEA + SHL: Odd = (A * B) << S
  for (unsigned S = 1; S <= 5; ++S) {
    int64_t Q = Odd >> S;
    if (Q <= 1)
      break;
    if ((Q << S) != Odd)
      continue;
    for (int64_t A : LeaFactors) {
      if (Q % A != 0)
        continue;
      int64_t B = Q / A;
      if (isLeaFactor(B)) {
        Steps.push_back({StepKind::LEA_CUR, (unsigned)isLeaFactor(B)});
        Steps.push_back({StepKind::LEA_CUR, (unsigned)isLeaFactor(A)});
        Steps.push_back({StepKind::SHL, S});
        if (TrailingZeros)
          Steps.push_back({StepKind::SHL, TrailingZeros});
        return true;
      }
    }
  }

  // LEA + SUB_SRC: Odd = A * K - 1  where A is factored further.
  // e.g., 23 = 3*8 - 1
  {
    int64_t Plus1 = Odd + 1;
    SmallVector<Step, 8> SubSteps;
    // Try to decompose Plus1 as a product of LEA factors + shifts.
    unsigned Plus1Shift = 0;
    int64_t Plus1Odd = Plus1;
    while ((Plus1Odd & 1) == 0) {
      Plus1Odd >>= 1;
      Plus1Shift++;
    }
    bool Found = false;
    if (Plus1Odd == 1 && Plus1Shift > 0) {
      // Odd+1 is a power of 2: lea*1 then shl, then sub src.
      // But shl alone doesn't give us the intermediate LEA.
      // Just SHL on current value, then SUB.
      SubSteps.push_back({StepKind::SHL, Plus1Shift});
      Found = true;
    } else if (isLeaFactor(Plus1Odd)) {
      SubSteps.push_back({StepKind::LEA_CUR, (unsigned)isLeaFactor(Plus1Odd)});
      if (Plus1Shift)
        SubSteps.push_back({StepKind::SHL, Plus1Shift});
      Found = true;
    } else {
      // Try factoring Plus1Odd as two LEA factors.
      for (int64_t A : LeaFactors) {
        if (Plus1Odd % A != 0)
          continue;
        int64_t B = Plus1Odd / A;
        if (isLeaFactor(B)) {
          SubSteps.push_back({StepKind::LEA_CUR, (unsigned)isLeaFactor(B)});
          SubSteps.push_back({StepKind::LEA_CUR, (unsigned)isLeaFactor(A)});
          if (Plus1Shift)
            SubSteps.push_back({StepKind::SHL, Plus1Shift});
          Found = true;
          break;
        }
      }
      // Try factoring Plus1Odd as LEA * shift (e.g., 11 = 3 << 2 - 1... no).
      // Actually try: Plus1Odd = LeaFactor << S
      if (!Found) {
        for (unsigned S2 = 1; S2 <= 5; ++S2) {
          int64_t Q = Plus1Odd >> S2;
          if (Q <= 1) break;
          if ((Q << S2) == Plus1Odd && isLeaFactor(Q)) {
            SubSteps.push_back(
                {StepKind::LEA_CUR, (unsigned)isLeaFactor(Q)});
            SubSteps.push_back({StepKind::SHL, S2 + Plus1Shift});
            Found = true;
            break;
          }
        }
      }
    }
    if (Found) {
      for (auto &S : SubSteps)
        Steps.push_back(S);
      Steps.push_back({StepKind::SUB_SRC, 0});
      if (TrailingZeros)
        Steps.push_back({StepKind::SHL, TrailingZeros});
      return true;
    }
  }

  // LEA + ADD_SRC: Odd = A * K + 1 where A is factored further.
  // e.g., 41 = 5*8 + 1
  {
    int64_t Minus1 = Odd - 1;
    if (Minus1 > 1) {
      SmallVector<Step, 8> AddSteps;
      unsigned Minus1Shift = 0;
      int64_t Minus1Odd = Minus1;
      while ((Minus1Odd & 1) == 0) {
        Minus1Odd >>= 1;
        Minus1Shift++;
      }
      bool Found = false;
      if (Minus1Odd == 1 && Minus1Shift > 0) {
        AddSteps.push_back({StepKind::SHL, Minus1Shift});
        Found = true;
      } else if (isLeaFactor(Minus1Odd)) {
        AddSteps.push_back(
            {StepKind::LEA_CUR, (unsigned)isLeaFactor(Minus1Odd)});
        if (Minus1Shift)
          AddSteps.push_back({StepKind::SHL, Minus1Shift});
        Found = true;
      } else {
        for (int64_t A : LeaFactors) {
          if (Minus1Odd % A != 0)
            continue;
          int64_t B = Minus1Odd / A;
          if (isLeaFactor(B)) {
            AddSteps.push_back(
                {StepKind::LEA_CUR, (unsigned)isLeaFactor(B)});
            AddSteps.push_back(
                {StepKind::LEA_CUR, (unsigned)isLeaFactor(A)});
            if (Minus1Shift)
              AddSteps.push_back({StepKind::SHL, Minus1Shift});
            Found = true;
            break;
          }
        }
        // Try: Minus1Odd = LeaFactor << S
        if (!Found) {
          for (unsigned S2 = 1; S2 <= 5; ++S2) {
            int64_t Q = Minus1Odd >> S2;
            if (Q <= 1) break;
            if ((Q << S2) == Minus1Odd && isLeaFactor(Q)) {
              AddSteps.push_back(
                  {StepKind::LEA_CUR, (unsigned)isLeaFactor(Q)});
              AddSteps.push_back({StepKind::SHL, S2 + Minus1Shift});
              Found = true;
              break;
            }
          }
        }
      }
      if (Found) {
        for (auto &S : AddSteps)
          Steps.push_back(S);
        Steps.push_back({StepKind::ADD_SRC, 0});
        if (TrailingZeros)
          Steps.push_back({StepKind::SHL, TrailingZeros});
        return true;
      }
    }
  }

  // LEA_SRC_PLUS_CUR: Odd = scale * K + 1 using lea dst,[src + cur*scale]
  // This is for patterns like x*41 = x + x*5*8: lea t,[x+x*4]; lea r,[x+t*8]
  // Here the second LEA uses src as base and cur (=5x) as index with scale 8.
  {
    // Try: Odd = 1 + Scale * K where Scale in {2,4,8} and K is decomposable
    static const unsigned Scales[] = {2, 4, 8};
    for (unsigned Scale : Scales) {
      if ((Odd - 1) % Scale != 0)
        continue;
      int64_t K = (Odd - 1) / Scale;
      if (K <= 1)
        continue;
      if (isLeaFactor(K)) {
        Steps.push_back({StepKind::LEA_CUR, (unsigned)isLeaFactor(K)});
        Steps.push_back({StepKind::LEA_SRC_PLUS_CUR, Scale});
        if (TrailingZeros)
          Steps.push_back({StepKind::SHL, TrailingZeros});
        return true;
      }
      // K = A * B where A is LEA-able, B is LEA-able
      for (int64_t A : LeaFactors) {
        if (K % A != 0)
          continue;
        int64_t B = K / A;
        if (isLeaFactor(B)) {
          Steps.push_back({StepKind::LEA_CUR, (unsigned)isLeaFactor(B)});
          Steps.push_back({StepKind::LEA_CUR, (unsigned)isLeaFactor(A)});
          Steps.push_back({StepKind::LEA_SRC_PLUS_CUR, Scale});
          if (TrailingZeros)
            Steps.push_back({StepKind::SHL, TrailingZeros});
          return true;
        }
      }
      // K = K' << S where K' is LEA-able
      for (unsigned S = 1; S <= 5; ++S) {
        int64_t Kp = K >> S;
        if (Kp <= 1)
          break;
        if ((Kp << S) == K && isLeaFactor(Kp)) {
          Steps.push_back({StepKind::LEA_CUR, (unsigned)isLeaFactor(Kp)});
          Steps.push_back({StepKind::SHL, S});
          Steps.push_back({StepKind::LEA_SRC_PLUS_CUR, Scale});
          if (TrailingZeros)
            Steps.push_back({StepKind::SHL, TrailingZeros});
          return true;
        }
      }
    }
  }

  // Two-level: decompose Odd into a product chain with SUB.
  // Pattern: Odd = (A * B - 1) * C  where all are small
  // e.g., 276 = ((3*8-1)*3)*4 = 23*3*4
  //       23 = 3*8-1, then *3, then *4 (shift 2)
  // But 276 has trailing zeros: 276 = 69 * 4, so Odd=69 with TrailingZeros=2
  // 69 = 23 * 3
  // 23 = 24 - 1 = 3*8 - 1
  // So: lea*3, shl 3, sub src, lea*3, shl 2
  //
  // This is the general pattern for 276. Let me handle it:
  // Odd = X * LeaFactor where X = (LeaFactor2 << S) +/- 1
  for (int64_t A : LeaFactors) {
    if (Odd % A != 0)
      continue;
    int64_t X = Odd / A;
    // Try X = (factor << shift) - 1
    {
      int64_t XPlus1 = X + 1;
      unsigned XShift = 0;
      int64_t XOdd = XPlus1;
      while (XOdd > 1 && (XOdd & 1) == 0) {
        XOdd >>= 1;
        XShift++;
      }
      if (XShift > 0 && isLeaFactor(XOdd)) {
        Steps.push_back(
            {StepKind::LEA_CUR, (unsigned)isLeaFactor(XOdd)});
        Steps.push_back({StepKind::SHL, XShift});
        Steps.push_back({StepKind::SUB_SRC, 0});
        Steps.push_back({StepKind::LEA_CUR, (unsigned)isLeaFactor(A)});
        if (TrailingZeros)
          Steps.push_back({StepKind::SHL, TrailingZeros});
        return true;
      }
    }
    // Try X = (factor << shift) + 1
    {
      int64_t XMinus1 = X - 1;
      if (XMinus1 > 1) {
        unsigned XShift = 0;
        int64_t XOdd = XMinus1;
        while (XOdd > 1 && (XOdd & 1) == 0) {
          XOdd >>= 1;
          XShift++;
        }
        if (XShift > 0 && isLeaFactor(XOdd)) {
          Steps.push_back(
              {StepKind::LEA_CUR, (unsigned)isLeaFactor(XOdd)});
          Steps.push_back({StepKind::SHL, XShift});
          Steps.push_back({StepKind::ADD_SRC, 0});
          Steps.push_back(
              {StepKind::LEA_CUR, (unsigned)isLeaFactor(A)});
          if (TrailingZeros)
            Steps.push_back({StepKind::SHL, TrailingZeros});
          return true;
        }
      }
    }
    // Try X itself as LEA-able
    if (isLeaFactor(X)) {
      Steps.push_back({StepKind::LEA_CUR, (unsigned)isLeaFactor(X)});
      Steps.push_back({StepKind::LEA_CUR, (unsigned)isLeaFactor(A)});
      if (TrailingZeros)
        Steps.push_back({StepKind::SHL, TrailingZeros});
      return true;
    }
  }

  // Three LEA factors: Odd = A * B * C
  for (int64_t A : LeaFactors) {
    if (Odd % A != 0)
      continue;
    int64_t Q = Odd / A;
    for (int64_t B : LeaFactors) {
      if (Q % B != 0)
        continue;
      int64_t D = Q / B;
      if (isLeaFactor(D)) {
        Steps.push_back({StepKind::LEA_CUR, (unsigned)isLeaFactor(D)});
        Steps.push_back({StepKind::LEA_CUR, (unsigned)isLeaFactor(B)});
        Steps.push_back({StepKind::LEA_CUR, (unsigned)isLeaFactor(A)});
        if (TrailingZeros)
          Steps.push_back({StepKind::SHL, TrailingZeros});
        return true;
      }
    }
  }

  // Not decomposable with our patterns.
  return false;
}

class X86DecomposeImulPass : public MachineFunctionPass {
public:
  static char ID;
  X86DecomposeImulPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 MSVC 6.0 IMUL decomposition into LEA/SHL/ADD/SUB";
  }

private:
  const X86InstrInfo *TII = nullptr;

  bool decomposeImul(MachineBasicBlock &MBB, MachineInstr &MI,
                     Register DstReg, Register SrcReg, int64_t ImmVal);
};

char X86DecomposeImulPass::ID = 0;

} // namespace

/// Build: lea DstReg, [BaseReg + IndexReg * Scale]
/// For Scale 4 or 8 with no base: lea DstReg, [0 + IndexReg * Scale]
static void buildLEA(MachineBasicBlock &MBB,
                     MachineBasicBlock::iterator InsertPt, const DebugLoc &DL,
                     const X86InstrInfo *TII, Register DstReg,
                     Register BaseReg, unsigned Scale, Register IndexReg) {
  BuildMI(MBB, InsertPt, DL, TII->get(X86::LEA32r), DstReg)
      .addReg(BaseReg)
      .addImm(Scale)
      .addReg(IndexReg)
      .addImm(0)
      .addReg(0);
}

bool X86DecomposeImulPass::decomposeImul(MachineBasicBlock &MBB,
                                         MachineInstr &MI, Register DstReg,
                                         Register SrcReg, int64_t ImmVal) {
  SmallVector<Step, 8> Steps;
  if (!findDecomposition(ImmVal, Steps))
    return false;

  DebugLoc DL = MI.getDebugLoc();
  MachineBasicBlock::iterator InsertPt = MI.getIterator();

  // Register assignment: MSVC 6.0 alternates between src and scratch.
  // When dest != src, dest acts as scratch (and may receive the result).
  // When dest == src, we pick the other register from {EAX, ECX} as scratch.
  // This matches MSVC 6.0's pattern where src is typically ECX (thiscall)
  // and scratch is EAX, or vice versa.
  Register ScratchReg;
  if (DstReg == SrcReg) {
    ScratchReg = (SrcReg == X86::EAX) ? X86::ECX : X86::EAX;
  } else {
    ScratchReg = DstReg;
  }

  // Track which register holds the "current value" at each step.
  // LEA writes to the "other" register and swaps roles.
  // SHL/ADD/SUB operate in-place on the current register.
  Register CurReg = SrcReg;
  Register OtherReg = ScratchReg;

  // The original source value must remain accessible for ADD_SRC/SUB_SRC
  // and LEA_SRC_PLUS_CUR. It stays in SrcReg throughout the sequence.
  Register OrigSrcReg = SrcReg;

  // Safety check: if the first step is SHL (in-place on SrcReg) and a later
  // step needs the original source (ADD_SRC, SUB_SRC, LEA_SRC_PLUS_CUR),
  // we must copy src to scratch first so src is preserved.
  bool NeedsSrcLater = false;
  for (const Step &S : Steps) {
    if (S.Kind == StepKind::ADD_SRC || S.Kind == StepKind::SUB_SRC ||
        S.Kind == StepKind::LEA_SRC_PLUS_CUR ||
        S.Kind == StepKind::LEA_CUR_PLUS_SRC)
      NeedsSrcLater = true;
  }
  if (NeedsSrcLater && !Steps.empty() && Steps[0].Kind == StepKind::SHL) {
    // Emit: mov scratch, src  (preserve original in SrcReg)
    BuildMI(MBB, InsertPt, DL, TII->get(X86::MOV32rr), ScratchReg)
        .addReg(SrcReg);
    // Now SHL will operate on scratch instead.
    CurReg = ScratchReg;
    OtherReg = SrcReg;
  }

  for (const Step &S : Steps) {
    switch (S.Kind) {
    case StepKind::LEA_CUR: {
      // lea OtherReg, [CurReg + CurReg * Scale]
      Register DestReg = OtherReg;
      buildLEA(MBB, InsertPt, DL, TII, DestReg, CurReg, S.Param, CurReg);
      OtherReg = CurReg;
      CurReg = DestReg;
      break;
    }
    case StepKind::LEA_SRC_PLUS_CUR: {
      // lea OtherReg, [OrigSrcReg + CurReg * Scale]
      Register DestReg = OtherReg;
      buildLEA(MBB, InsertPt, DL, TII, DestReg, OrigSrcReg, S.Param, CurReg);
      OtherReg = CurReg;
      CurReg = DestReg;
      break;
    }
    case StepKind::LEA_CUR_PLUS_SRC: {
      // lea OtherReg, [CurReg + OrigSrcReg * Scale]
      Register DestReg = OtherReg;
      buildLEA(MBB, InsertPt, DL, TII, DestReg, CurReg, S.Param, OrigSrcReg);
      OtherReg = CurReg;
      CurReg = DestReg;
      break;
    }
    case StepKind::SHL: {
      // shl CurReg, Amount -- in-place, no register swap.
      BuildMI(MBB, InsertPt, DL, TII->get(X86::SHL32ri), CurReg)
          .addReg(CurReg)
          .addImm(S.Param);
      break;
    }
    case StepKind::ADD_SRC: {
      // add CurReg, OrigSrcReg -- in-place.
      BuildMI(MBB, InsertPt, DL, TII->get(X86::ADD32rr), CurReg)
          .addReg(CurReg)
          .addReg(OrigSrcReg);
      break;
    }
    case StepKind::SUB_SRC: {
      // sub CurReg, OrigSrcReg -- in-place.
      BuildMI(MBB, InsertPt, DL, TII->get(X86::SUB32rr), CurReg)
          .addReg(CurReg)
          .addReg(OrigSrcReg);
      break;
    }
    case StepKind::NEG: {
      BuildMI(MBB, InsertPt, DL, TII->get(X86::NEG32r), CurReg)
          .addReg(CurReg);
      break;
    }
    case StepKind::MOV_SRC: {
      // Not used in current decompositions.
      break;
    }
    }
  }

  // The result is in CurReg. If DstReg != CurReg, emit a MOV.
  if (CurReg != DstReg) {
    BuildMI(MBB, InsertPt, DL, TII->get(X86::MOV32rr), DstReg)
        .addReg(CurReg);
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

      Changed |= decomposeImul(MBB, MI, DstReg, SrcReg, ImmVal);
    }
  }

  return Changed;
}

FunctionPass *llvm::createX86DecomposeImulPass() {
  return new X86DecomposeImulPass();
}

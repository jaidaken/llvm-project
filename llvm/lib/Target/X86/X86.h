//===-- X86.h - Top-level interface for X86 representation ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the x86
// target library, as used by the LLVM JIT.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_X86_X86_H
#define LLVM_LIB_TARGET_X86_X86_H

#include "llvm/Support/CodeGen.h"

namespace llvm {

class FunctionPass;
class InstructionSelector;
class PassRegistry;
class X86RegisterBankInfo;
class X86Subtarget;
class X86TargetMachine;

/// This pass converts a legalized DAG into a X86-specific DAG, ready for
/// instruction scheduling.
FunctionPass *createX86ISelDag(X86TargetMachine &TM, CodeGenOptLevel OptLevel);

/// This pass initializes a global base register for PIC on x86-32.
FunctionPass *createX86GlobalBaseRegPass();

/// This pass combines multiple accesses to local-dynamic TLS variables so that
/// the TLS base address for the module is only fetched once per execution path
/// through the function.
FunctionPass *createCleanupLocalDynamicTLSPass();

/// This function returns a pass which converts floating-point register
/// references and pseudo instructions into floating-point stack references and
/// physical instructions.
FunctionPass *createX86FloatingPointStackifierPass();

/// This pass inserts AVX vzeroupper instructions before each call to avoid
/// transition penalty between functions encoded with AVX and SSE.
FunctionPass *createX86IssueVZeroUpperPass();

/// This pass inserts ENDBR instructions before indirect jump/call
/// destinations as part of CET IBT mechanism.
FunctionPass *createX86IndirectBranchTrackingPass();

/// Return a pass that pads short functions with NOOPs.
/// This will prevent a stall when returning on the Atom.
FunctionPass *createX86PadShortFunctions();

/// Return a pass that selectively replaces certain instructions (like add,
/// sub, inc, dec, some shifts, and some multiplies) by equivalent LEA
/// instructions, in order to eliminate execution delays in some processors.
FunctionPass *createX86FixupLEAs();

/// Return a pass that replaces equivalent slower instructions with faster
/// ones.
FunctionPass *createX86FixupInstTuning();

/// Return a pass that reduces the size of vector constant pool loads.
FunctionPass *createX86FixupVectorConstants();

/// Return a pass that removes redundant LEA instructions and redundant address
/// recalculations.
FunctionPass *createX86OptimizeLEAs();

/// Return a pass that transforms setcc + movzx pairs into xor + setcc.
FunctionPass *createX86FixupSetCC();

/// Return a pass that transform inline buffer security check into seperate bb
FunctionPass *createX86WinFixupBufferSecurityCheckPass();

/// Return a pass that avoids creating store forward block issues in the hardware.
FunctionPass *createX86AvoidStoreForwardingBlocks();

/// Return a pass that lowers EFLAGS copy pseudo instructions.
FunctionPass *createX86FlagsCopyLoweringPass();

/// Return a pass that expands DynAlloca pseudo-instructions.
FunctionPass *createX86DynAllocaExpander();

/// Return a pass that config the tile registers.
FunctionPass *createX86TileConfigPass();

/// Return a pass that preconfig the tile registers before fast reg allocation.
FunctionPass *createX86FastPreTileConfigPass();

/// Return a pass that config the tile registers after fast reg allocation.
FunctionPass *createX86FastTileConfigPass();

/// Return a pass that insert pseudo tile config instruction.
FunctionPass *createX86PreTileConfigPass();

/// Return a pass that lower the tile copy instruction.
FunctionPass *createX86LowerTileCopyPass();

/// Return a pass that inserts int3 at the end of the function if it ends with a
/// CALL instruction. The pass does the same for each funclet as well. This
/// ensures that the open interval of function start and end PCs contains all
/// return addresses for the benefit of the Windows x64 unwinder.
FunctionPass *createX86AvoidTrailingCallPass();

/// Return a pass that optimizes the code-size of x86 call sequences. This is
/// done by replacing esp-relative movs with pushes.
FunctionPass *createX86CallFrameOptimization();

/// Return an IR pass that inserts EH registration stack objects and explicit
/// EH state updates. This pass must run after EH preparation, which does
/// Windows-specific but architecture-neutral preparation.
FunctionPass *createX86WinEHStatePass();

/// Return a Machine IR pass that expands X86-specific pseudo
/// instructions into a sequence of actual instructions. This pass
/// must run after prologue/epilogue insertion and before lowering
/// the MachineInstr to MC.
FunctionPass *createX86ExpandPseudoPass();

/// Return a Machine IR pass that expands MOVZX32rm8/MOVZX32rm16 into
/// XOR32rr_REV + MOV8rm/MOV16rm for functions with the expand_movzx attribute.
FunctionPass *createX86ExpandMovzxPass();
FunctionPass *createX86FixupMovzxOverlapPass();

/// Return a Machine IR pass that converts MOVZX+RET to MOV partial reg+RET
/// and MOV32ri small_imm+RET to MOV8ri+RET (MSVC 6.0 partial return pattern).
FunctionPass *createX86Msvc6PartialReturnPass();

/// Return a Machine IR pass that swaps EAX<->ECX throughout a function when
/// LLVM assigned them opposite to MSVC 6.0's fastcall this-ptr pattern.
FunctionPass *createX86Msvc6FastcallRegFixPass();

/// Return a Machine IR pass that converts reg-reg arithmetic ops to their
/// reversed encoding variants (e.g., ADD32rr -> ADD32rr_REV) for MSVC 6.0.
FunctionPass *createX86ReversedOpsPass();

/// Return a Machine IR pass that converts CMP32mr to CMP32rm (swapping memory
/// and register operands) and flips following condition codes for MSVC 6.0.
FunctionPass *createX86CmpRevPass();

/// Return a Machine IR pass that folds load+op+store into memory-direct
/// arithmetic (e.g., mov r,[m]; add r,s; mov [m],r -> add [m],s).
FunctionPass *createX86PreferAddMemPass();

/// Return a Machine IR pass that rewrites fucompp/fnstsw/sahf/setcc to
/// fcomp/fnstsw/test ah,N matching MSVC 6.0 FPU comparison patterns.
FunctionPass *createX86PreferFcompFnstswPass();

/// Return a Machine IR pass that rewrites fld+fstp pairs to integer
/// mov+mov for float parameters (MSVC 6.0 treats floats as raw 32-bit).
FunctionPass *createX86PreferIntegerFloatMovePass();
FunctionPass *createX86PreferIntFloatForwardPass();

/// Return a Machine IR pass that rewrites fld+fstp float parameter forwarding
/// to mov+mov integer moves for functions with the float_param_raw_push
/// attribute (MSVC 6.0 treats forwarded floats as raw 32-bit values).
FunctionPass *createX86FloatParamRawPushPass();

/// Return a Machine IR pass that converts XOR32rr/XOR32rr_REV self-xor
/// zeroing idioms to XOR8rr/XOR8rr_REV for functions with prefer_xor8.
FunctionPass *createX86PreferXOR8Pass();

/// Return a Machine IR pass that replaces movzx+add/sub+mov byte sequences
/// with INC8m/DEC8m for functions with prefer_inc_dec_byte.
FunctionPass *createX86PreferIncDecBytePass();

/// Return a Machine IR pass that folds load+dec+store sequences into
/// memory-direct DEC for functions with prefer_memory_dec.
FunctionPass *createX86PreferMemoryDecPass();

/// Return a Machine IR pass that converts FLDZ/FLD1 pseudo instructions to
/// constant pool memory loads for functions with the suppress_fp_imm attribute.
FunctionPass *createX86SuppressFPImmPass();

/// Return a Machine IR pass that converts MOV32ri reg, 0xFFFFFFFF to
/// OR32ri8 reg, -1 for functions with the prefer_or_minus_one attribute.
FunctionPass *createX86OrMinusOnePass();

/// Return a Machine IR pass that prevents folding of (~byte >> N) & 1 into
/// test+sete for functions with the no_test_sete_fold attribute.
FunctionPass *createX86NoTestSeteFoldPass();

/// Return a Machine IR pass that converts boolean NOT to neg+sbb+inc
/// for functions with the prefer_neg_sbb attribute.
FunctionPass *createX86PreferNegSbbPass();

/// Return a Machine IR pass that removes fstp+fld truncation round-trips
/// for functions with the no_float_truncation attribute (MSVC 6.0 pattern).
FunctionPass *createX86NoFloatTruncationPass();

/// Return a Machine IR pass that converts XOR+CMP+SETcc to SUB8+NEG8+SBB+INC
/// for functions with the prefer_8bit_ops attribute.
FunctionPass *createX86Prefer8BitOpsPass();

/// Return a Machine IR pass that routes sete through ECX instead of AL
/// for functions with the prefer_sete_ecx attribute.
FunctionPass *createX86PreferSeteEcxPass();

/// Return a Machine IR pass that folds fld+fmulp into memory-form fmul
/// for functions with the prefer_fmul_mem attribute.
FunctionPass *createX86PreferFmulMemPass();

/// Return a Machine IR pass that replaces add esp, 4 with pop ecx
/// for functions with the prefer_pop_cleanup attribute.
FunctionPass *createX86PreferPopCleanupPass();

/// Return a Machine IR pass that converts TEST8mi (test byte [mem], imm)
/// to MOV32rm + TEST8ri AH/CH for high-byte tests (MSVC 6.0 pattern).
FunctionPass *createX86PreferTestAhPass();

/// Return a Machine IR pass that converts MOVZX32rr16 to MOV32rr + AND32ri
/// 0xFFFF for functions with the prefer_and_mask attribute.
FunctionPass *createX86PreferAndMaskPass();
FunctionPass *createX86PreferMovPushPass();
FunctionPass *createX86PreferPreloadStackParamPass();
FunctionPass *createX86PreferPushImmPass();
FunctionPass *createX86PreferPushFstpPass();
FunctionPass *createX86PreferRegisterPushPass();
FunctionPass *createX86PreferThiscallReorderPass();
FunctionPass *createX86PreferDirectEcxLoadPass();
FunctionPass *createX86PreferPushBeforeEcxPass();
FunctionPass *createX86DuplicateEcxRestorePass();

/// Return a Machine IR pass that sinks MOV32ri EAX, imm past a TEST/CMP + Jcc
/// sequence into the fall-through block for functions with defer_return_value.
FunctionPass *createX86DeferReturnValuePass();

/// Return a Machine IR pass that sinks XOR32rr EAX, EAX from early in the
/// function to just before the first EAX use for functions with defer_zero_eax.
FunctionPass *createX86DeferZeroEaxPass();
FunctionPass *createX86InterleaveEcxRestorePass();
FunctionPass *createX86PreferBatchPushPass();
FunctionPass *createX86PreferSequentialParamLoadPass();
FunctionPass *createX86HoistPushLoadsPass();
FunctionPass *createX86BatchLoadBeforeStorePass();
FunctionPass *createX86Msvc6EvalOrderPass();
FunctionPass *createX86PreferVtableEdxPass();
FunctionPass *createX86PreferBranchBoolPass();
FunctionPass *createX86PreventSetccMergePass();
FunctionPass *createX86PreferMovAndCmpPass();
FunctionPass *createX86MergeReturnZeroPass();
FunctionPass *createX86PreferRtlPushOrderPass();

/// Return a Machine IR pass that decomposes IMUL with specific constants
/// into LEA/SHL/SUB chains matching MSVC 6.0 strength reduction output.
FunctionPass *createX86DecomposeImulPass();

/// Return a Machine IR pass that converts TAILJMPm to CALL32m + RET
/// for functions with the call_tail attribute (MSVC 6.0 call-through-vtable).
FunctionPass *createX86CallTailPass();

/// Return a Machine IR pass that splits a near conditional jump over RET
/// into je skip; jmp target; skip: ret (MSVC 6.0 null-check thunk pattern).
FunctionPass *createX86SplitCondJmpPass();

/// Return a Machine IR pass that converts XOR32rr + INC32r to MOV32ri 1
/// for functions with the prefer_mov_imm attribute.
FunctionPass *createX86PreferMovImmPass();

/// Return a Machine IR pass that moves SUB ESP before callee-save pushes
/// and ADD ESP after callee-save pops (MSVC 6.0 sub-before-push pattern).
FunctionPass *createX86ReorderSubEspPass();

/// Return a Machine IR pass that forces `this` (ECX) into ESI at function
/// entry and rewrites all subsequent ECX uses to ESI (MSVC 6.0 pattern).
FunctionPass *createX86ForceThisToEsiPass();

/// Return a Machine IR pass that forces `this` (ECX) into EAX at function
/// entry and rewrites all subsequent ECX uses to EAX (MSVC 6.0 pattern for
/// small member functions without sub-calls).
FunctionPass *createX86ForceThisToEaxPass();

/// Return a Machine IR pass that swaps EBX<->ESI when the primary memory
/// base is in EBX but MSVC 6.0 expects it in ESI.
FunctionPass *createX86SwapBufRegisterPass();
FunctionPass *createX86SwapCmpRegistersPass();

/// Return a Machine IR pass that rewrites ECX field load + EAX sete to
/// EDX field load + ECX sete for MSVC 6.0 bitfield accessor functions.
FunctionPass *createX86Msvc6RegSwapPass();

/// Return a Machine IR pass that rewrites add/cmp/ja loop latches to
/// dec ebp/jne with trip count precomputation.
FunctionPass *createX86PreferTripCountLoopPass();

/// Return a Machine IR pass that hoists pointer advance before loads
/// and converts positive offsets to negative.
FunctionPass *createX86PreferEarlyBufAdvancePass();

/// Return a Machine IR pass that interleaves s2 updates across DO16
/// iterations to match MSVC 6.0's software-pipelined pattern.
FunctionPass *createX86InterleaveS2UpdatePass();

/// Return a Machine IR pass that hoists len -= k before the DO16 loop
/// and eliminates stack spills.
FunctionPass *createX86HoistLenSubPass();

/// Return a Machine IR pass that removes NOP alignment padding from
/// functions with msvc6_regalloc attribute.
FunctionPass *createX86StripNopPaddingPass();

/// Return a Machine IR pass that rewrites unsigned JCC conditions to
/// signed equivalents (jb->jl, ja->jg, etc.).
FunctionPass *createX86PreferSignedJccPass();

/// Return a Machine IR pass that restructures functions to match MSVC 6.0
/// block layout, split prologue, and interleaved epilogue.
FunctionPass *createX86Msvc6RestructurePass();

/// Return a Machine IR pass that moves field stores from before the vtable
/// load to between pushes and the vtable call (MSVC 6.0 interleaved pattern).
FunctionPass *createX86InterleaveStoreWithCallPass();

/// Return a Machine IR pass that moves ADD/LEA instructions from between
/// two consecutive PUSHes to before both PUSHes, matching MSVC 6.0's
/// interleaved scheduling of arithmetic between push arguments.
FunctionPass *createX86InterleaveAddPushPass();

/// Return a Machine IR pass that reorders instructions to match MSVC 6.0's
/// AST-order depth-first evaluation, adjusting registers simultaneously.
FunctionPass *createX86Msvc6SchedulePass();

/// Return a Machine IR pass that reorders consecutive stores to the same base
/// register to match MSVC 6.0's source/declaration order, as specified by the
/// store_order function attribute.
FunctionPass *createX86ReorderStoresPass();

/// Return a Machine IR pass that removes the XOR zeroing before bare
/// subreg loads when the upper bits are dead (MSVC 6.0 pattern).
FunctionPass *createX86SuppressMovzxPass();

/// Return a Machine IR pass that inserts a redundant CMP [mem], 0 after
/// DEC [mem] to reproduce MSVC 6.0's dec+cmp pattern.
FunctionPass *createX86InsertRedundantCmpPass();

/// Return a Machine IR pass that unfolds CMP [mem],imm into
/// MOV reg,[mem] + CMP reg,imm for MSVC 6.0 bool accessor patterns.
FunctionPass *createX86UnfoldCmpMemPass();

/// Return a Machine IR pass that unfolds memory-source ALU instructions
/// (e.g., ADD32rm) into MOV+register-register ALU (e.g., MOV32rm+ADD32rr)
/// for functions with the unfold_alu_mem attribute (MSVC 6.0 pattern).
FunctionPass *createX86UnfoldAluMemPass();

/// Return a Machine IR pass that converts immediate-zero stores (MOV32mi 0)
/// into XOR+MOV32mr for functions with the zero_via_xor attribute (MSVC 6.0
/// pattern of zeroing a register then storing through it).
FunctionPass *createX86ZeroViaXorPass();

/// Return a Machine IR pass that splits merged 32-bit immediate stores back
/// into two 16-bit stores for functions with the split_word_stores attribute
/// (MSVC 6.0 emitted adjacent word stores that the compiler merged).
FunctionPass *createX86SplitWordStoresPass();

/// This pass converts X86 cmov instructions into branch when profitable.
FunctionPass *createX86CmovConverterPass();

/// Return a Machine IR pass that selectively replaces
/// certain byte and word instructions by equivalent 32 bit instructions,
/// in order to eliminate partial register usage, false dependences on
/// the upper portions of registers, and to save code size.
FunctionPass *createX86FixupBWInsts();

/// Return a Machine IR pass that reassigns instruction chains from one domain
/// to another, when profitable.
FunctionPass *createX86DomainReassignmentPass();

/// This pass compress instructions from EVEX space to legacy/VEX/EVEX space when
/// possible in order to reduce code size or facilitate HW decoding.
FunctionPass *createX86CompressEVEXPass();

/// This pass creates the thunks for the retpoline feature.
FunctionPass *createX86IndirectThunksPass();

/// This pass replaces ret instructions with jmp's to __x86_return thunk.
FunctionPass *createX86ReturnThunksPass();

/// This pass ensures instructions featuring a memory operand
/// have distinctive <LineNumber, Discriminator> (with respect to each other)
FunctionPass *createX86DiscriminateMemOpsPass();

/// This pass applies profiling information to insert cache prefetches.
FunctionPass *createX86InsertPrefetchPass();

/// This pass insert wait instruction after X87 instructions which could raise
/// fp exceptions when strict-fp enabled.
FunctionPass *createX86InsertX87waitPass();

/// This pass optimizes arithmetic based on knowledge that is only used by
/// a reduction sequence and is therefore safe to reassociate in interesting
/// ways.
FunctionPass *createX86PartialReductionPass();

InstructionSelector *createX86InstructionSelector(const X86TargetMachine &TM,
                                                  const X86Subtarget &,
                                                  const X86RegisterBankInfo &);

FunctionPass *createX86LoadValueInjectionLoadHardeningPass();
FunctionPass *createX86LoadValueInjectionRetHardeningPass();
FunctionPass *createX86SpeculativeLoadHardeningPass();
FunctionPass *createX86SpeculativeExecutionSideEffectSuppression();
FunctionPass *createX86ArgumentStackSlotPass();

void initializeCompressEVEXPassPass(PassRegistry &);
void initializeFPSPass(PassRegistry &);
void initializeFixupBWInstPassPass(PassRegistry &);
void initializeFixupLEAPassPass(PassRegistry &);
void initializeX86ArgumentStackSlotPassPass(PassRegistry &);
void initializeX86FixupInstTuningPassPass(PassRegistry &);
void initializeX86FixupVectorConstantsPassPass(PassRegistry &);
void initializeWinEHStatePassPass(PassRegistry &);
void initializeX86AvoidSFBPassPass(PassRegistry &);
void initializeX86AvoidTrailingCallPassPass(PassRegistry &);
void initializeX86CallFrameOptimizationPass(PassRegistry &);
void initializeX86CmovConverterPassPass(PassRegistry &);
void initializeX86DAGToDAGISelLegacyPass(PassRegistry &);
void initializeX86DomainReassignmentPass(PassRegistry &);
void initializeX86DynAllocaExpanderPass(PassRegistry &);
void initializeX86ExecutionDomainFixPass(PassRegistry &);
void initializeX86ExpandPseudoPass(PassRegistry &);
void initializeX86FastPreTileConfigPass(PassRegistry &);
void initializeX86FastTileConfigPass(PassRegistry &);
void initializeX86FixupSetCCPassPass(PassRegistry &);
void initializeX86WinFixupBufferSecurityCheckPassPass(PassRegistry &);
void initializeX86FlagsCopyLoweringPassPass(PassRegistry &);
void initializeX86LoadValueInjectionLoadHardeningPassPass(PassRegistry &);
void initializeX86LoadValueInjectionRetHardeningPassPass(PassRegistry &);
void initializeX86LowerAMXIntrinsicsLegacyPassPass(PassRegistry &);
void initializeX86LowerAMXTypeLegacyPassPass(PassRegistry &);
void initializeX86LowerTileCopyPass(PassRegistry &);
void initializeX86OptimizeLEAPassPass(PassRegistry &);
void initializeX86PartialReductionPass(PassRegistry &);
void initializeX86PreTileConfigPass(PassRegistry &);
void initializeX86ReturnThunksPass(PassRegistry &);
void initializeX86SpeculativeExecutionSideEffectSuppressionPass(PassRegistry &);
void initializeX86SpeculativeLoadHardeningPassPass(PassRegistry &);
void initializeX86TileConfigPass(PassRegistry &);

namespace X86AS {
enum : unsigned {
  GS = 256,
  FS = 257,
  SS = 258,
  PTR32_SPTR = 270,
  PTR32_UPTR = 271,
  PTR64 = 272
};
} // End X86AS namespace

} // End llvm namespace

#endif

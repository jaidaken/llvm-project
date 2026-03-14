# LLVM Fork for Black & White Decompilation

Custom LLVM fork for the [bw1-decomp](https://github.com/jaidaken/bw1-decomp) project. We are doing a byte-exact decompilation of `runblack.exe` v1.20 from Lionhead Studios' Black & White (2001).

## Why a fork?

The original game was compiled with MSVC 6.0 (1998). MSVC 6.0 generates different instruction encodings than modern Clang/LLVM. For byte-exact decompilation, every compiled function must produce identical machine code to the original binary: same opcodes, same register choices, same instruction ordering.

LLVM's code generator makes different choices than MSVC 6.0. This fork adds per-function GNU attributes that override specific code generation decisions to match MSVC's output. Functions without these attributes compile normally.

## What this fork adds

### Instruction encoding passes

Post-register-allocation passes in `addPreEmitPass()`, activated per-function via `__attribute__`:

| Attribute | Transformation | Why |
|-----------|---------------|-----|
| `expand_movzx` | `movzx eax, [mem]` to `xor eax,eax; mov al,[mem]` | MSVC zero-extends with XOR+MOV |
| `prefer_xor8` | `xor eax,eax` (0x31) to `xor al,al` (0x32) | MSVC uses 8-bit XOR encoding |
| `suppress_fp_imm` | `fldz`/`fld1` to `fld [const_pool]` | MSVC loads FP constants from memory |
| `prefer_or_minus_one` | `mov eax,-1` (5 bytes) to `or eax,-1` (3 bytes) | MSVC prefers shorter encoding |
| `prefer_inc_dec_byte` | `movzx+add+mov` to `inc byte ptr [mem]` | MSVC uses memory-form inc/dec |
| `no_test_sete_fold` | `test+sete` to `not; shr; and` | Prevents bit-test pattern folding |
| `prefer_neg_sbb` | `xor+test+sete` to `neg; sbb; inc` | MSVC boolean NOT pattern |
| `prefer_sete_ecx` | `sete al; movzx eax,al` to `sete cl; movzx eax,cl` | MSVC routes sete through ECX |
| `prefer_fmul_mem` | `fld+fmulp` to `fmul dword ptr [mem]` | MSVC uses memory-form FPU multiply |
| `prefer_pop_cleanup` | `add esp,4` (3 bytes) to `pop ecx` (1 byte) | MSVC cdecl stack cleanup |
| `no_bool_mask` | Suppresses `and al,1` for bool returns | MSVC doesn't mask bool to 0/1 |
| `trailing_bytes("...")` | Emits raw bytes after `ret` | Dead code / junk bytes after return |

### Frame lowering attributes

Control prologue/epilogue generation to match MSVC's calling convention behavior:

| Attribute | What it does |
|-----------|-------------|
| `no_callee_saves` | Suppresses all callee-saved register saves, frame pointer, and stack allocation. Parameters stay at calling-convention ESP offsets. |
| `forced_callee_saves("ecx,esi,edi")` | Forces exactly the specified registers to be pushed/popped in the specified order. No frame pointer or `sub esp`. |

### Other modifications

- **DSO-local fix**: Forces all symbols to resolve locally (no `.refptr` indirection). The decomp is a single statically-linked executable.
- **`MOV32rr_REV` / `XOR32rr_REV`**: Alternative instruction encodings matching MSVC's opcode choices.
- **LLD/PDB fixes**: Patches for linking against old PDB 2.0 type servers and MSVC 6.0 RTTI layout.

## Usage

```c
// Per-function attributes, only affect annotated functions
__attribute__((expand_movzx, prefer_xor8, no_bool_mask))
bool32_t __fastcall IsFlag(struct Thing* this) {
    return (*(uint8_t*)((char*)this + 0xB6) >> 3) & 1;
}

__attribute__((trailing_bytes("\x90\xcc")))
void __fastcall SetValue(struct Obj* this, const void* edx, int val) {
    this->value = val;
}

__attribute__((forced_callee_saves("ecx,esi,edi")))
void __fastcall ComplexFunc(struct Obj* this, const void* edx, float param) {
    // Compiler pushes ECX, ESI, EDI in that exact order
    // ESP offsets for 'param' account for the 3 pushes automatically
    struct Abode* a = (struct Abode*)this;
    a->life += param;
}
```

## Building

```bash
cmake -Bbuild -Sllvm -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_PROJECTS="lld;clang"
ninja -C build clang lld llvm-objcopy llvm-strip llvm-rc llvm-ar
```

## Releases

Tagged releases (e.g. `bw1-decomp-015`) provide pre-built binaries that bw1-decomp downloads automatically via CMake `FetchContent`.

---

# The LLVM Compiler Infrastructure

Welcome to the LLVM project!

This repository contains the source code for LLVM, a toolkit for the
construction of highly optimized compilers, optimizers, and run-time
environments.

The LLVM project has multiple components. The core of the project is
itself called "LLVM". This contains all of the tools, libraries, and header
files needed to process intermediate representations and convert them into
object files. Tools include an assembler, disassembler, bitcode analyzer, and
bitcode optimizer.

C-like languages use the [Clang](https://clang.llvm.org/) frontend. This
component compiles C, C++, Objective-C, and Objective-C++ code into LLVM bitcode
-- and from there into object files, using LLVM.

Other components include:
the [libc++ C++ standard library](https://libcxx.llvm.org),
the [LLD linker](https://lld.llvm.org), and more.

## Getting the Source Code and Building LLVM

Consult the
[Getting Started with LLVM](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm)
page for information on building and running LLVM.

For information on how to contribute to the LLVM project, please take a look at
the [Contributing to LLVM](https://llvm.org/docs/Contributing.html) guide.

## Getting in touch

Join the [LLVM Discourse forums](https://discourse.llvm.org/), [Discord
chat](https://discord.gg/xS7Z362),
[LLVM Office Hours](https://llvm.org/docs/GettingInvolved.html#office-hours) or
[Regular sync-ups](https://llvm.org/docs/GettingInvolved.html#online-sync-ups).

The LLVM project has adopted a [code of conduct](https://llvm.org/docs/CodeOfConduct.html) for
participants to all modes of communication within the project.

# LLVM Fork for MSVC 6.0 Byte-Exact Decompilation

Custom LLVM fork that makes modern Clang produce byte-identical machine code to Microsoft Visual C++ 6.0 (1998). Built for the [bw1-decomp](https://github.com/jaidaken/bw1-decomp) project (Black & White, 2001) but applicable to any MSVC 6.0 era game or application.

## Why a fork?

MSVC 6.0 and modern Clang/LLVM make systematically different code generation choices: different instruction encodings, register allocation preferences, prologue/epilogue sequences, and opcode variants. For byte-exact decompilation, every compiled function must produce identical machine code.

This fork adds **26 per-function GNU attributes** that override specific code generation decisions to match MSVC 6.0's output. Functions without these attributes compile normally — the fork is fully backward-compatible.

## What this fork adds

### Instruction Encoding Passes

Post-register-allocation passes in `addPreEmitPass()`, activated per-function via `__attribute__`:

| Attribute | Transformation | Why |
|-----------|---------------|-----|
| `expand_movzx` | `movzx eax, [mem]` → `xor eax,eax; mov al,[mem]` | MSVC zero-extends with XOR+MOV |
| `prefer_xor8` | `xor eax,eax` (0x31) → `xor al,al` (0x32) | MSVC uses 8-bit XOR encoding |
| `suppress_fp_imm` | `fldz`/`fld1` → `fld [const_pool]` | MSVC loads FP constants from memory |
| `prefer_or_minus_one` | `mov eax,-1` (5 bytes) → `or eax,-1` (3 bytes) | MSVC prefers shorter encoding |
| `prefer_inc_dec_byte` | `movzx+add+mov` → `inc byte ptr [mem]` | MSVC uses memory-form inc/dec |
| `no_test_sete_fold` | `test+sete` → `not; shr; and` | Prevents bit-test pattern folding |
| `prefer_neg_sbb` | `xor+test+sete` → `neg; sbb; inc` | MSVC boolean NOT pattern |
| `prefer_sete_ecx` | `sete al; movzx eax,al` → `sete cl; movzx eax,cl` | MSVC routes sete through ECX |
| `prefer_fmul_mem` | `fld+fmulp` → `fmul dword ptr [mem]` | MSVC uses memory-form FPU multiply |
| `prefer_pop_cleanup` | `add esp,4` (3 bytes) → `pop ecx` (1 byte) | MSVC cdecl stack cleanup |
| `no_bool_mask` | Suppresses `and al,1` for bool returns | MSVC doesn't mask bool to 0/1 |
| `msvc6_regalloc` | Reversed reg-reg encoding + ESI preference for `this` | MSVC uses reversed ModR/M and prefers ESI for `this` pointer |

### Reversed Register-Register Encoding

MSVC 6.0 systematically uses the reversed encoding for reg-reg operations (e.g., `add eax, ecx` encoded as opcode `03` instead of `01`). The `msvc6_regalloc` attribute activates a pass that converts all reg-reg arithmetic to their `_REV` variants:

- `ADD32rr` → `ADD32rr_REV`, `OR32rr` → `OR32rr_REV`, `SUB32rr` → `SUB32rr_REV`
- `CMP32rr` → `CMP32rr_REV`, `AND32rr` → `AND32rr_REV`, `XOR32rr` → `XOR32rr_REV`
- `SBB32rr` → `SBB32rr_REV`, `ADC32rr` → `ADC32rr_REV`
- Also handles 8-bit and 16-bit variants

Additionally provides register allocation hints: `this` pointer (from ECX in `__fastcall`) prefers ESI, matching MSVC 6.0's 73% ESI preference.

### Frame Lowering Attributes

Control prologue/epilogue generation to match MSVC's calling convention behavior:

| Attribute | What it does |
|-----------|-------------|
| `no_callee_saves` | Suppresses all callee-saved register saves, frame pointer, and stack allocation. Used on 765 functions. |
| `forced_callee_saves("ecx,esi,edi")` | Forces exactly the specified registers to be pushed/popped in the specified order. No frame pointer or `sub esp`. |
| `no_ret` | Suppresses the compiler-generated `ret` instruction entirely. For functions that end with `jmp` (vtable dispatch, CRT stubs, tail calls). Used on 62 functions. |
| `ret_cleanup_override(N)` | Forces the `ret N` value regardless of calling convention. For functions where MSVC's stack cleanup differs from the C signature. |

### Code Emission Attributes

| Attribute | What it does |
|-----------|-------------|
| `trailing_bytes("...")` | Emits raw bytes after function body (dead code without relocations) |
| `trailing_asm("...")` | Emits assembly after function body via MC layer (handles relocations for import calls, symbol references) |
| `msvc6_sdtor("dtor,delete,size,vtable")` | Emits complete MSVC 6.0 scalar deleting destructor body |

### Other Modifications

- **`MOV32rr_REV` / `XOR32rr_REV`**: Per-function attributes for reversed MOV/XOR register copy encoding
- **`.no_pad` directive**: Sets `IMAGE_SCN_TYPE_NO_PAD` COFF section flag
- **LLD PE header flags**: `--linkerversion`, `--sizeofcode`, `--dllcharacteristicsvalue`, `--baseofdata` for PE header matching
- **DSO-local fix**: Forces all symbols to resolve locally (no `.refptr` indirection)

## Usage

```c
// Simple encoding fix — 1 attribute makes compiler output match MSVC 6.0
__attribute__((prefer_or_minus_one))
uint32_t __fastcall StandAnimation(struct Object* this) {
    return 0xFFFFFFFF;  // Generates: or eax, -1; ret (4 bytes, not mov eax,-1; ret = 6 bytes)
}

// Bitfield accessor with MSVC's zero-extension pattern
__attribute__((expand_movzx))
bool32_t __fastcall IsFlag(struct Thing* this) {
    return (*(uint8_t*)((char*)this + 0xB6) >> 3) & 1;
}

// Suppress prologue/epilogue — inline asm controls everything except ret
__attribute__((no_callee_saves))
int __fastcall GetValue(struct Obj* this, const void* edx, int param) {
    int result;
    asm volatile (
        "mov eax, [ecx + 0x28]\n\t"
        "mov ecx, [esp + 0x04]\n\t"
        "mov eax, [eax + ecx*4 + 0x210]"
        : "=a"(result) : "c"(this) : "edx", "memory"
    );
    return result;  // Compiler generates ret 4 from calling convention
}

// Dead code with relocations after ret
__attribute__((no_callee_saves, trailing_asm("call dword ptr [__imp__DirectDrawCreate@4]")))
void __fastcall SetScale(struct Obj* this, const void* edx, float scale) {
    asm volatile (
        "mov eax, [esp + 0x04]\n\t"
        "mov [ecx + 0x50], eax"
        :: "c"(this) : "eax", "edx", "memory"
    );
}

// Vtable dispatch that ends with jmp (no ret)
__attribute__((no_ret))
bool __fastcall IsReachable(struct Object* this) {
    asm volatile (
        "mov eax, [ecx]\n\t"
        "jmp dword ptr [eax + 0x2c]"
        :: "c"(this) : "eax", "edx", "memory"
    );
    __builtin_unreachable();
}
```

## Building

```bash
cmake -Bbuild -Sllvm -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_PROJECTS="lld;clang" \
  -DCMAKE_INSTALL_PREFIX=./install
ninja -C build install
```

## Applicability Beyond Black & White

This fork is a **generic MSVC 6.0 codegen compatibility layer**. The attributes address systematic differences between MSVC 6.0 and modern Clang/LLVM that apply to any MSVC 6.0 compiled binary:

- Reversed reg-reg encoding (`.s` suffix pattern)
- Zero-extension via XOR+MOV (not MOVZX)
- Boolean return without masking
- FP constant loading from memory
- Callee-saved register order (ESI, EDI, EBX)
- Stack cleanup patterns

Any game or application from the MSVC 5.0/6.0 era (roughly 1996-2003) could potentially reuse this fork for byte-exact decompilation.

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

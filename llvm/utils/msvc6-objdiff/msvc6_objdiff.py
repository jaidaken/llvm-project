#!/usr/bin/env python3
"""msvc6_objdiff - Byte-level function diff for MSVC 6.0 decompilation projects.

Compares compiled COFF object file functions against reference bytes from
various sources and suggests custom LLVM attributes to match MSVC 6.0
code generation.

Part of the bw1-decomp LLVM fork: https://github.com/jaidaken/llvm-project

Subcommands:
    diff     Compare one function between compiled .o and a reference source
    batch    Compare all functions in a .o against reference ASM directory
    suggest  Standalone attribute suggestion from two hex byte sequences

Reference sources (for diff/batch):
    --ref-obj FILE [--ref-symbol SYM]   Another .o file via llvm-objdump
    --ref-bin FILE (--rva RVA | --start-address VA) [--size N]
                                         PE binary via llvm-objdump
    --ref-hex "HH... HH..."             Hex bytes (spaces separate instructions)
    --ref-asm FILE                       Single annotated ASM file
    --ref-asm-dir DIR                    Directory of annotated ASM files

Examples:
    # Compare .o function against PE binary at an RVA
    msvc6_objdiff.py diff compiled.o MyFunc --ref-bin game.exe --rva 0x1234 --size 48

    # Compare .o function against another .o file
    msvc6_objdiff.py diff compiled.o MyFunc --ref-obj reference.o

    # Compare .o function against hex bytes (spaces = instruction boundaries)
    msvc6_objdiff.py diff compiled.o MyFunc --ref-hex "33c0 8a4108 c3"

    # Batch compare all functions in .o against ASM directory
    msvc6_objdiff.py batch compiled.o --ref-asm-dir ./asm/

    # Standalone suggestion (no .o needed)
    msvc6_objdiff.py suggest "33c0 8a4108 c3" "0fb64108 c3"
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple


# ============================================================
# Data types
# ============================================================

class Instr(NamedTuple):
    """A single disassembled instruction."""
    raw: bytes          # instruction bytes
    text: str           # assembly text (for display)
    relocs: frozenset   # byte offsets within `raw` that are relocations


# ============================================================
# MSVC 6.0 attribute suggestion tables
# ============================================================

# (expected_first_byte, compiled_first_byte) -> suggestion string
OPCODE_SUGGESTIONS: dict[tuple[int, int], str] = {
    # Reversed register encoding — MSVC 6.0 systematically uses the alternate
    # ModR/M direction for all reg-reg operations.
    (0x03, 0x01): "reversed ADD — expected 03 (ADD r, r/m), got 01 (ADD r/m, r)",
    (0x01, 0x03): "reversed ADD — expected 01 (ADD r/m, r), got 03 (ADD r, r/m)",
    (0x0B, 0x09): "reversed OR — expected 0B (OR r, r/m), got 09 (OR r/m, r)",
    (0x09, 0x0B): "reversed OR — expected 09 (OR r/m, r), got 0B (OR r, r/m)",
    (0x13, 0x11): "reversed ADC — expected 13 (ADC r, r/m), got 11 (ADC r/m, r)",
    (0x11, 0x13): "reversed ADC — expected 11 (ADC r/m, r), got 13 (ADC r, r/m)",
    (0x1B, 0x19): "reversed SBB — expected 1B (SBB r, r/m), got 19 (SBB r/m, r)",
    (0x19, 0x1B): "reversed SBB — expected 19 (SBB r/m, r), got 1B (SBB r, r/m)",
    (0x23, 0x21): "reversed AND — expected 23 (AND r, r/m), got 21 (AND r/m, r)",
    (0x21, 0x23): "reversed AND — expected 21 (AND r/m, r), got 23 (AND r, r/m)",
    (0x2B, 0x29): "reversed SUB — expected 2B (SUB r, r/m), got 29 (SUB r/m, r)",
    (0x29, 0x2B): "reversed SUB — expected 29 (SUB r/m, r), got 2B (SUB r, r/m)",
    (0x33, 0x31): "XOR32rr_REV — expected 33 (XOR r, r/m), got 31 (XOR r/m, r)",
    (0x31, 0x33): "XOR32rr_REV — expected 31 (XOR r/m, r), got 33 (XOR r, r/m)",
    (0x3B, 0x39): "reversed CMP — expected 3B (CMP r, r/m), got 39 (CMP r/m, r)",
    (0x39, 0x3B): "reversed CMP — expected 39 (CMP r/m, r), got 3B (CMP r, r/m)",
    (0x8B, 0x89): "MOV32rr_REV — expected 8B (MOV r, r/m), got 89 (MOV r/m, r)",
    (0x89, 0x8B): "MOV32rr_REV — expected 89 (MOV r/m, r), got 8B (MOV r, r/m)",
    # XOR width preference
    (0x32, 0x33): "prefer_xor8 — expected 32 (xor al, al), got 33 (xor eax, eax)",
    (0x33, 0x32): "prefer_xor8 — expected 33 (xor eax, eax), got 32 (xor al, al)",
}

# (expected_prefix | None, compiled_prefix | None, suggestion)
# None means "match any bytes"
PATTERN_SUGGESTIONS: list[tuple[bytes | None, bytes | None, str]] = [
    # prefer_or_minus_one: or eax,-1 (83 C8 FF) vs mov eax,-1 (B8 FF FF FF FF)
    (bytes([0x83, 0xC8, 0xFF]), None,
     "prefer_or_minus_one — expected or eax,-1 (83 C8 FF), got mov eax,-1"),
    (None, bytes([0x83, 0xC8, 0xFF]),
     "prefer_or_minus_one — compiled has or eax,-1, but expected doesn't"),
    # prefer_pop_cleanup: pop ecx (59) vs add esp,4 (83 C4 04)
    (bytes([0x59]), bytes([0x83, 0xC4, 0x04]),
     "prefer_pop_cleanup — expected pop ecx (59), got add esp,4 (83 C4 04)"),
    (bytes([0x83, 0xC4, 0x04]), bytes([0x59]),
     "prefer_pop_cleanup — expected add esp,4, got pop ecx"),
    # suppress_fp_imm: fldz (D9 EE) / fld1 (D9 E8) vs fld [mem] (D9 05)
    (bytes([0xD9, 0xEE]), None,
     "suppress_fp_imm — expected fldz (D9 EE)"),
    (bytes([0xD9, 0xE8]), None,
     "suppress_fp_imm — expected fld1 (D9 E8)"),
    (None, bytes([0xD9, 0xEE]),
     "suppress_fp_imm — compiled has fldz (D9 EE)"),
    (None, bytes([0xD9, 0xE8]),
     "suppress_fp_imm — compiled has fld1 (D9 E8)"),
    (bytes([0xD9, 0x05]), None,
     "suppress_fp_imm — expected fld [mem] (D9 05)"),
    # expand_movzx: movzx (0F B6) vs xor+mov
    (None, bytes([0x0F, 0xB6]),
     "expand_movzx — compiled has movzx (0F B6), expected xor+mov sequence"),
    (bytes([0x0F, 0xB6]), None,
     "expand_movzx — expected has movzx (0F B6)"),
    # prefer_sete_ecx: sete cl (0F 94 C1) vs sete al (0F 94 C0)
    (bytes([0x0F, 0x94, 0xC1]), bytes([0x0F, 0x94, 0xC0]),
     "prefer_sete_ecx — expected sete cl, got sete al"),
    (bytes([0x0F, 0x94, 0xC0]), bytes([0x0F, 0x94, 0xC1]),
     "prefer_sete_ecx — expected sete al, got sete cl"),
    # prefer_neg_sbb: neg (F7 D8/D9/DA) vs xor (33)
    (bytes([0xF7, 0xD8]), bytes([0x33]),
     "prefer_neg_sbb — expected neg eax (F7 D8), got xor"),
    (bytes([0xF7, 0xD9]), bytes([0x33]),
     "prefer_neg_sbb — expected neg ecx (F7 D9), got xor"),
    (bytes([0xF7, 0xDA]), bytes([0x33]),
     "prefer_neg_sbb — expected neg edx (F7 DA), got xor"),
    (bytes([0x1B]), bytes([0x85]),
     "prefer_neg_sbb — expected sbb (1B), got test (85)"),
    # no_test_sete_fold: not al (F6 D0) vs test al (A8/F6 C0)
    (bytes([0xF6, 0xD0]), bytes([0xA8]),
     "no_test_sete_fold — expected not al (F6 D0), got test al (A8)"),
    (bytes([0xF6, 0xD0]), bytes([0xF6, 0xC0]),
     "no_test_sete_fold — expected not al (F6 D0), got test al (F6 C0)"),
    (bytes([0xC1, 0xE8]), bytes([0x0F, 0x94]),
     "no_test_sete_fold — expected shr eax (C1 E8), got sete (0F 94)"),
    # no_bool_mask: and al,1 (24 01) or and eax,1 (83 E0 01)
    (None, bytes([0x24, 0x01]),
     "no_bool_mask — compiled has and al,1 (24 01)"),
    (None, bytes([0x83, 0xE0, 0x01]),
     "no_bool_mask — compiled has and eax,1 (83 E0 01)"),
    # prefer_fmul_mem: fmul [mem] (D8) vs fmulp (DE C9)
    (bytes([0xD8]), bytes([0xDE, 0xC9]),
     "prefer_fmul_mem — expected fmul [mem] (D8), got fmulp (DE C9)"),
    # prefer_inc_dec_byte: inc/dec byte [mem] (FE) vs movzx (0F B6)
    (bytes([0xFE]), bytes([0x0F, 0xB6]),
     "prefer_inc_dec_byte — expected inc/dec byte (FE), got movzx (0F B6)"),
]

# Callee-saved register push opcodes
CALLEE_SAVE_PUSHES: dict[int, str] = {
    0x53: "ebx",
    0x55: "ebp",
    0x56: "esi",
    0x57: "edi",
}


# ============================================================
# LLVM tool resolution
# ============================================================

def find_tool(name: str, bin_dir: str | None = None) -> str:
    """Locate an LLVM tool binary.

    Search order: --llvm-bin-dir flag, LLVM_BIN_DIR env var, paths relative
    to this script (build/bin, install/bin), then PATH.
    """
    candidates = []
    if bin_dir:
        candidates.append(Path(bin_dir) / name)
    env_dir = os.environ.get("LLVM_BIN_DIR")
    if env_dir:
        candidates.append(Path(env_dir) / name)
    script_dir = Path(__file__).resolve().parent
    for rel in ["../../../build/bin", "../../../install/bin"]:
        candidates.append(script_dir / rel / name)
    for c in candidates:
        if c.exists():
            return str(c)
    return name  # fall back to PATH


# ============================================================
# PE utilities
# ============================================================

def pe_read_image_base(pe_path: Path) -> int:
    """Read ImageBase from a PE32 file header."""
    with open(pe_path, "rb") as f:
        magic = f.read(2)
        if magic != b"MZ":
            raise ValueError(f"{pe_path} is not a PE file (no MZ signature)")
        f.seek(0x3C)
        pe_offset = struct.unpack("<I", f.read(4))[0]
        f.seek(pe_offset)
        if f.read(4) != b"PE\0\0":
            raise ValueError(f"{pe_path} has no PE signature")
        f.seek(pe_offset + 4 + 20)  # skip COFF header
        opt_magic = struct.unpack("<H", f.read(2))[0]
        if opt_magic == 0x10B:  # PE32
            f.seek(pe_offset + 4 + 20 + 28)
            return struct.unpack("<I", f.read(4))[0]
        elif opt_magic == 0x20B:  # PE32+
            f.seek(pe_offset + 4 + 20 + 24)
            return struct.unpack("<Q", f.read(8))[0]
        raise ValueError(f"Unknown optional header magic: 0x{opt_magic:x}")


def pe_rva_to_offset(pe_path: Path, rva: int) -> int | None:
    """Convert an RVA to a file offset using PE section headers."""
    with open(pe_path, "rb") as f:
        f.seek(0x3C)
        pe_offset = struct.unpack("<I", f.read(4))[0]
        f.seek(pe_offset + 4)
        _, num_sections, _, _, _, opt_size, _ = struct.unpack(
            "<HHIIIHH", f.read(20)
        )
        f.seek(pe_offset + 4 + 20 + opt_size)
        for _ in range(num_sections):
            _name = f.read(8)
            virt_size, virt_addr = struct.unpack("<II", f.read(8))
            raw_size, raw_ptr = struct.unpack("<II", f.read(8))
            f.read(16)  # skip remaining fields
            if virt_addr <= rva < virt_addr + max(virt_size, raw_size):
                return raw_ptr + (rva - virt_addr)
    return None


# ============================================================
# Instruction extraction: llvm-objdump output parser
# ============================================================

_OBJDUMP_INSTR_RE = re.compile(
    r"^\s*([0-9a-f]+):\s+((?:[0-9a-f]{2}\s)+)\s*(?:\t(.*))?$", re.I
)
_OBJDUMP_RELOC_RE = re.compile(
    r"^\s+([0-9a-f]+):\s+IMAGE_REL_", re.I
)
_OBJDUMP_LABEL_RE = re.compile(
    r"^[0-9a-f]+\s+<([^>]+)>:", re.I
)


def _parse_objdump_output(
    text: str, symbol: str | None = None, stop_at_next_label: bool = True
) -> list[Instr]:
    """Parse llvm-objdump -dr output into instruction list.

    If symbol is provided, extract only that function. Otherwise extract
    all instructions (used for address-range disassembly of PE binaries).
    """
    lines = text.splitlines()
    instrs: list[Instr] = []
    in_function = symbol is None  # if no symbol filter, capture everything
    pending_relocs: dict[int, set] = {}  # offset -> set of reloc byte positions

    for i, line in enumerate(lines):
        # Check for function labels
        label_m = _OBJDUMP_LABEL_RE.match(line)
        if label_m:
            label = label_m.group(1)
            if symbol is not None:
                if not in_function and symbol in label:
                    in_function = True
                    continue
                elif in_function and stop_at_next_label:
                    break
            continue

        if not in_function:
            continue

        # Check for relocation lines
        reloc_m = _OBJDUMP_RELOC_RE.match(line)
        if reloc_m:
            reloc_offset = int(reloc_m.group(1), 16)
            # Associate this relocation with the most recent instruction
            if instrs:
                last_instr = instrs[-1]
                # The relocation offset is absolute within the section.
                # We need to compute which bytes within the instruction
                # are affected. The instruction starts at its section offset.
                instr_start = sum(len(inst.raw) for inst in instrs[:-1])
                rel_pos = reloc_offset - instr_start
                if 0 <= rel_pos < len(last_instr.raw):
                    new_relocs = set(last_instr.relocs)
                    # COFF relocations are typically 4 bytes (DIR32, REL32)
                    for off in range(rel_pos, min(rel_pos + 4, len(last_instr.raw))):
                        new_relocs.add(off)
                    instrs[-1] = Instr(
                        last_instr.raw, last_instr.text, frozenset(new_relocs)
                    )
            continue

        # Check for instruction lines
        instr_m = _OBJDUMP_INSTR_RE.match(line)
        if instr_m:
            raw = bytes.fromhex(instr_m.group(2).replace(" ", ""))
            asm_text = (instr_m.group(3) or "").strip()
            instrs.append(Instr(raw=raw, text=asm_text, relocs=frozenset()))

    return instrs


def disasm_obj(
    obj_path: Path, symbol: str, objdump: str = "llvm-objdump"
) -> list[Instr]:
    """Extract instructions for a symbol from a COFF .o file."""
    result = subprocess.run(
        [objdump, "-dr", str(obj_path)],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"llvm-objdump failed on {obj_path}:\n{result.stderr}"
        )
    instrs = _parse_objdump_output(result.stdout, symbol)
    if not instrs:
        raise ValueError(
            f"Symbol '{symbol}' not found in {obj_path}. "
            f"Run `{objdump} -t {obj_path}` to list available symbols."
        )
    return instrs


def disasm_obj_symbols(obj_path: Path, objdump: str = "llvm-objdump") -> list[str]:
    """List all function symbols in a .o file (for batch mode)."""
    result = subprocess.run(
        [objdump, "-t", str(obj_path)],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"llvm-objdump -t failed:\n{result.stderr}")
    symbols = []
    for line in result.stdout.splitlines():
        # Format: OFFSET FLAGS SECTION SIZE NAME
        parts = line.split()
        if len(parts) >= 6 and ".text" in parts:
            name = parts[-1]
            if not name.startswith("."):
                symbols.append(name)
    return symbols


# ============================================================
# Instruction extraction: PE binary
# ============================================================

def disasm_pe(
    pe_path: Path,
    start_va: int,
    size: int | None,
    objdump: str = "llvm-objdump",
    auto_trim: bool = True,
) -> list[Instr]:
    """Disassemble a PE binary at a given virtual address range.

    If size is None and auto_trim is True, reads up to 65536 bytes and trims
    at the first ret followed by int3/nop padding.
    """
    max_size = size if size is not None else 65536
    stop_va = start_va + max_size
    result = subprocess.run(
        [
            objdump, "-d",
            f"--start-address=0x{start_va:x}",
            f"--stop-address=0x{stop_va:x}",
            str(pe_path),
        ],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"llvm-objdump failed on {pe_path}:\n{result.stderr}"
        )
    instrs = _parse_objdump_output(result.stdout, symbol=None, stop_at_next_label=False)
    if not instrs:
        raise ValueError(
            f"No instructions found at VA 0x{start_va:x} in {pe_path}"
        )
    if size is None and auto_trim:
        instrs = _trim_at_function_end(instrs)
    return instrs


def _trim_at_function_end(instrs: list[Instr]) -> list[Instr]:
    """Trim instruction list at function boundary (ret + int3/nop padding)."""
    for i, instr in enumerate(instrs):
        is_ret = instr.raw[0:1] in (b"\xc3", b"\xc2")
        if is_ret and i + 1 < len(instrs):
            next_byte = instrs[i + 1].raw[0:1]
            if next_byte in (b"\xcc", b"\x90"):
                return instrs[: i + 1]
    return instrs


# ============================================================
# Instruction extraction: hex string
# ============================================================

def parse_hex_instrs(hex_str: str) -> list[Instr]:
    """Parse space-separated hex instruction groups.

    Each whitespace-separated group is treated as one instruction.
    Example: "33c0 8a4108 c3" -> 3 instructions.
    """
    groups = hex_str.strip().split()
    instrs = []
    for group in groups:
        try:
            raw = bytes.fromhex(group)
        except ValueError as e:
            raise ValueError(f"Invalid hex group '{group}': {e}") from e
        instrs.append(Instr(raw=raw, text="", relocs=frozenset()))
    return instrs


# ============================================================
# Instruction extraction: annotated ASM files
# ============================================================

# Format: bw1-decomp style — `instruction // 0xADDRESS HEXBYTES`
_BW1_BYTE_RE = re.compile(r"//\s*0x[0-9a-f]+\s+((?:[0-9a-f]{2})+)\s*$", re.I)
# Format: objdump style — `  ADDRESS: HH HH HH  \tinstruction`
_OBJDUMP_ASM_RE = re.compile(
    r"^\s*([0-9a-f]+):\s+((?:[0-9a-f]{2}\s)+)\s{2,}(.+)?$", re.I
)
# Format: IDA listing — `.text:ADDRESS HH HH HH  instruction`
_IDA_ASM_RE = re.compile(
    r"^\.[a-z]+:[0-9a-f]+\s+((?:[0-9A-F]{2}\s)+)\s*(.*)?$", re.I
)
# Function label (MSVC-mangled or C-style)
_LABEL_RE = re.compile(r"^([?_@\w][^:\s]*):(?:\s|$)")
# Local labels to skip
_LOCAL_LABEL_RE = re.compile(r"^(?:\.L|LAB|[0-9]+:)")


def detect_asm_format(lines: list[str]) -> str | None:
    """Auto-detect annotated ASM format from the first 50 non-blank lines."""
    for line in lines[:50]:
        if _BW1_BYTE_RE.search(line):
            return "bw1"
        if _OBJDUMP_ASM_RE.match(line):
            return "objdump"
        if _IDA_ASM_RE.match(line):
            return "ida"
    return None


def parse_asm_instrs(
    lines: list[str], fmt: str, start_line: int = 0
) -> list[Instr]:
    """Parse instruction bytes from annotated ASM lines starting at start_line.

    Stops at the next function label.
    """
    instrs = []
    saw_ret = False

    for line in lines[start_line:]:
        stripped = line.rstrip()
        if not stripped:
            continue

        # Stop at next function label (but not local labels)
        if instrs and _LABEL_RE.match(stripped) and not _LOCAL_LABEL_RE.match(stripped):
            break

        raw = None
        asm_text = ""

        if fmt == "bw1":
            m = _BW1_BYTE_RE.search(stripped)
            if m:
                raw = bytes.fromhex(m.group(1))
                # Extract asm text (everything before the // comment, after optional label:)
                text_part = stripped[: m.start()].strip()
                colon_idx = text_part.find(":")
                if colon_idx >= 0 and _LABEL_RE.match(text_part):
                    text_part = text_part[colon_idx + 1:].strip()
                asm_text = text_part
        elif fmt == "objdump":
            m = _OBJDUMP_ASM_RE.match(stripped)
            if m:
                raw = bytes.fromhex(m.group(2).replace(" ", ""))
                asm_text = (m.group(3) or "").strip()
        elif fmt == "ida":
            m = _IDA_ASM_RE.match(stripped)
            if m:
                raw = bytes.fromhex(m.group(1).replace(" ", ""))
                asm_text = (m.group(2) or "").strip()

        if raw is None:
            continue

        # Stop at NOP/int3 after ret (function padding)
        if saw_ret and raw[0:1] in (b"\xcc", b"\x90"):
            break
        saw_ret = raw[0:1] in (b"\xc3", b"\xc2")
        instrs.append(Instr(raw=raw, text=asm_text, relocs=frozenset()))

    return instrs


def build_symbol_index(
    asm_dir: Path,
) -> dict[str, tuple[Path, int]]:
    """Scan a directory of ASM files and build symbol -> (file, line) index."""
    index: dict[str, tuple[Path, int]] = {}
    for asm_file in sorted(asm_dir.glob("*.asm")):
        with open(asm_file, "r", errors="replace") as f:
            for lineno, line in enumerate(f):
                m = _LABEL_RE.match(line.rstrip())
                if m and not _LOCAL_LABEL_RE.match(line):
                    index[m.group(1)] = (asm_file, lineno)
    return index


def ref_from_asm_dir(
    asm_dir: Path, symbol: str, index: dict[str, tuple[Path, int]] | None = None
) -> list[Instr]:
    """Look up a symbol in a directory of ASM files and extract its instructions."""
    if index is None:
        index = build_symbol_index(asm_dir)
    if symbol not in index:
        raise ValueError(
            f"Symbol '{symbol}' not found in ASM directory {asm_dir}"
        )
    asm_file, lineno = index[symbol]
    with open(asm_file, "r", errors="replace") as f:
        lines = f.readlines()
    fmt = detect_asm_format(lines)
    if fmt is None:
        raise ValueError(
            f"Cannot detect ASM format of {asm_file}. Supported: bw1, objdump, IDA."
        )
    return parse_asm_instrs(lines, fmt, start_line=lineno)


def ref_from_asm_file(asm_path: Path, symbol: str) -> list[Instr]:
    """Extract instructions for a symbol from a single annotated ASM file."""
    with open(asm_path, "r", errors="replace") as f:
        lines = f.readlines()
    fmt = detect_asm_format(lines)
    if fmt is None:
        raise ValueError(
            f"Cannot detect ASM format of {asm_path}. Supported: bw1, objdump, IDA."
        )
    # Find the symbol
    for lineno, line in enumerate(lines):
        m = _LABEL_RE.match(line.rstrip())
        if m and m.group(1) == symbol:
            return parse_asm_instrs(lines, fmt, start_line=lineno)
    raise ValueError(f"Symbol '{symbol}' not found in {asm_path}")


# ============================================================
# Byte comparison and masking
# ============================================================

def bytes_match(expected: bytes, got: bytes, got_relocs: frozenset) -> bool:
    """Compare instruction bytes, ignoring positions covered by relocations."""
    if len(expected) != len(got):
        return False
    for i in range(len(expected)):
        if i in got_relocs:
            continue
        if expected[i] != got[i]:
            return False
    return True


def masked_hex(raw: bytes, relocs: frozenset) -> str:
    """Format bytes as hex with 'xx' for relocation positions."""
    parts = []
    for i, b in enumerate(raw):
        parts.append("xx" if i in relocs else f"{b:02x}")
    return " ".join(parts)


# ============================================================
# Needleman-Wunsch instruction alignment
# ============================================================

MATCH_SCORE = 2
MISMATCH_SCORE = -1
GAP_SCORE = -2
ALIGN_LIMIT = 500


def align_instructions(
    expected: list[Instr], compiled: list[Instr]
) -> list[tuple[Instr | None, Instr | None]]:
    """Align two instruction sequences using Needleman-Wunsch.

    Returns list of (expected_instr | None, compiled_instr | None) pairs.
    None entries represent gaps in the alignment.
    """
    n, m = len(expected), len(compiled)
    if n == 0:
        return [(None, c) for c in compiled]
    if m == 0:
        return [(e, None) for e in expected]
    if n > ALIGN_LIMIT or m > ALIGN_LIMIT:
        return _index_align(expected, compiled)

    # Build score matrix
    score = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] + GAP_SCORE
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] + GAP_SCORE
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = bytes_match(
                expected[i - 1].raw, compiled[j - 1].raw, compiled[j - 1].relocs
            )
            diag = score[i - 1][j - 1] + (MATCH_SCORE if match else MISMATCH_SCORE)
            up = score[i - 1][j] + GAP_SCORE
            left = score[i][j - 1] + GAP_SCORE
            score[i][j] = max(diag, up, left)

    # Traceback
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            match = bytes_match(
                expected[i - 1].raw, compiled[j - 1].raw, compiled[j - 1].relocs
            )
            diag = score[i - 1][j - 1] + (MATCH_SCORE if match else MISMATCH_SCORE)
            if score[i][j] == diag:
                alignment.append((expected[i - 1], compiled[j - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and score[i][j] == score[i - 1][j] + GAP_SCORE:
            alignment.append((expected[i - 1], None))
            i -= 1
        else:
            alignment.append((None, compiled[j - 1]))
            j -= 1

    alignment.reverse()
    return alignment


def _index_align(
    expected: list[Instr], compiled: list[Instr]
) -> list[tuple[Instr | None, Instr | None]]:
    """Simple index-based alignment for large functions."""
    result = []
    for i in range(max(len(expected), len(compiled))):
        e = expected[i] if i < len(expected) else None
        c = compiled[i] if i < len(compiled) else None
        result.append((e, c))
    return result


# ============================================================
# Suggestion engine
# ============================================================

def suggest_fixes(expected_bytes: bytes, got_bytes: bytes) -> list[str]:
    """Suggest attributes for a single instruction mismatch."""
    suggestions = []

    # Opcode-level check (first byte pairs)
    if expected_bytes and got_bytes:
        key = (expected_bytes[0], got_bytes[0])
        if key in OPCODE_SUGGESTIONS:
            suggestions.append(OPCODE_SUGGESTIONS[key])

    # Pattern-level check (byte prefix matching)
    for exp_pat, got_pat, suggestion in PATTERN_SUGGESTIONS:
        exp_match = exp_pat is None or expected_bytes[: len(exp_pat)] == exp_pat
        got_match = got_pat is None or got_bytes[: len(got_pat)] == got_pat
        if exp_match and got_match:
            suggestions.append(suggestion)

    return suggestions


def suggest_function_fixes(
    expected_instrs: list[Instr], compiled_instrs: list[Instr]
) -> list[str]:
    """Suggest attributes based on whole-function analysis."""
    suggestions = []

    # --- Prologue analysis: callee-save register detection ---
    exp_saves = []
    for instr in expected_instrs[:6]:
        if len(instr.raw) == 1 and instr.raw[0] in CALLEE_SAVE_PUSHES:
            exp_saves.append(CALLEE_SAVE_PUSHES[instr.raw[0]])
        else:
            break
    comp_saves = []
    for instr in compiled_instrs[:6]:
        if len(instr.raw) == 1 and instr.raw[0] in CALLEE_SAVE_PUSHES:
            comp_saves.append(CALLEE_SAVE_PUSHES[instr.raw[0]])
        else:
            break

    if comp_saves and not exp_saves:
        suggestions.append(
            f"no_callee_saves — compiled pushes {','.join(comp_saves)} "
            f"but expected has no callee saves"
        )
    elif comp_saves and exp_saves and comp_saves != exp_saves:
        suggestions.append(
            f'forced_callee_saves("{",".join(exp_saves)}") — '
            f"expected pushes {','.join(exp_saves)}, "
            f"compiled pushes {','.join(comp_saves)}"
        )

    # --- Epilogue analysis: ret / ret N ---
    exp_last = expected_instrs[-1].raw if expected_instrs else b""
    comp_last = compiled_instrs[-1].raw if compiled_instrs else b""

    exp_has_ret = exp_last[0:1] in (b"\xc3", b"\xc2")
    comp_has_ret = comp_last[0:1] in (b"\xc3", b"\xc2")

    if comp_has_ret and not exp_has_ret:
        suggestions.append(
            "no_ret — compiled ends with ret but expected does not"
        )
    elif exp_has_ret and comp_has_ret:
        # Check ret N value
        exp_cleanup = (
            struct.unpack("<H", exp_last[1:3])[0]
            if exp_last[0:1] == b"\xc2" and len(exp_last) >= 3
            else 0
        )
        comp_cleanup = (
            struct.unpack("<H", comp_last[1:3])[0]
            if comp_last[0:1] == b"\xc2" and len(comp_last) >= 3
            else 0
        )
        if exp_cleanup != comp_cleanup:
            suggestions.append(
                f'ret_cleanup_override("{exp_cleanup}") — '
                f"expected ret {exp_cleanup}, compiled ret {comp_cleanup}"
            )

    # --- Instruction count analysis ---
    def _count_prefix(instrs: list[Instr], prefix: bytes) -> int:
        return sum(1 for i in instrs if i.raw[: len(prefix)] == prefix)

    # movzx (0F B6) count
    exp_movzx = _count_prefix(expected_instrs, bytes([0x0F, 0xB6]))
    comp_movzx = _count_prefix(compiled_instrs, bytes([0x0F, 0xB6]))
    if comp_movzx > exp_movzx:
        suggestions.append(
            f"expand_movzx — compiled has {comp_movzx} movzx (0F B6), "
            f"expected has {exp_movzx}"
        )

    # fmulp (DE C9)
    exp_fmulp = _count_prefix(expected_instrs, bytes([0xDE, 0xC9]))
    comp_fmulp = _count_prefix(compiled_instrs, bytes([0xDE, 0xC9]))
    if comp_fmulp > exp_fmulp:
        suggestions.append(
            f"prefer_fmul_mem — compiled has {comp_fmulp} fmulp, "
            f"expected has {exp_fmulp}"
        )

    # inc/dec byte (FE)
    exp_incdec = _count_prefix(expected_instrs, bytes([0xFE]))
    comp_incdec = _count_prefix(compiled_instrs, bytes([0xFE]))
    if exp_incdec > comp_incdec:
        suggestions.append(
            f"prefer_inc_dec_byte — expected has {exp_incdec} inc/dec byte, "
            f"compiled has {comp_incdec}"
        )

    return suggestions


def collect_all_suggestions(
    alignment: list[tuple[Instr | None, Instr | None]],
    expected_instrs: list[Instr],
    compiled_instrs: list[Instr],
) -> list[str]:
    """Collect and deduplicate suggestions from all layers."""
    seen = set()
    suggestions = []

    def _add(s: str) -> None:
        if s not in seen:
            seen.add(s)
            suggestions.append(s)

    # Per-instruction suggestions from aligned pairs
    for exp, comp in alignment:
        if exp is not None and comp is not None:
            if not bytes_match(exp.raw, comp.raw, comp.relocs):
                for s in suggest_fixes(exp.raw, comp.raw):
                    _add(s)

    # Function-level suggestions
    for s in suggest_function_fixes(expected_instrs, compiled_instrs):
        _add(s)

    return suggestions


# ============================================================
# Output formatting
# ============================================================

def format_diff(
    symbol: str,
    alignment: list[tuple[Instr | None, Instr | None]],
    expected_instrs: list[Instr],
    compiled_instrs: list[Instr],
) -> str:
    """Format an aligned diff for display."""
    lines = []
    exp_bytes = sum(len(i.raw) for i in expected_instrs)
    comp_bytes = sum(len(i.raw) for i in compiled_instrs)
    lines.append(f"msvc6_objdiff: {symbol}")
    lines.append("=" * 60)
    lines.append(
        f"Compiled: {comp_bytes} bytes ({len(compiled_instrs)} instructions)"
    )
    lines.append(
        f"Expected: {exp_bytes} bytes ({len(expected_instrs)} instructions)"
    )
    lines.append("")

    for exp, comp in alignment:
        if exp is None and comp is not None:
            # Gap in expected (extra instruction in compiled)
            comp_hex = masked_hex(comp.raw, comp.relocs)
            lines.append(
                f"  ++ Got:      {comp_hex:<36s} {comp.text}"
            )
        elif comp is None and exp is not None:
            # Gap in compiled (extra instruction in expected)
            exp_hex = exp.raw.hex()
            exp_hex_fmt = " ".join(
                exp_hex[i: i + 2] for i in range(0, len(exp_hex), 2)
            )
            lines.append(
                f"  -- Expected: {exp_hex_fmt:<36s} {exp.text}"
            )
        elif exp is not None and comp is not None:
            match = bytes_match(exp.raw, comp.raw, comp.relocs)
            exp_hex = exp.raw.hex()
            exp_hex_fmt = " ".join(
                exp_hex[i: i + 2] for i in range(0, len(exp_hex), 2)
            )
            comp_hex = masked_hex(comp.raw, comp.relocs)
            marker = "  " if match else ">>"
            lines.append(
                f"  {marker} Expected: {exp_hex_fmt:<36s} {exp.text}"
            )
            lines.append(
                f"  {marker} Got:      {comp_hex:<36s} {comp.text}"
            )
            if not match:
                fixes = suggest_fixes(exp.raw, comp.raw)
                for fix in fixes:
                    lines.append(f"     -> TRY: {fix}")
            lines.append("")

    # Function-level suggestions
    func_suggestions = suggest_function_fixes(expected_instrs, compiled_instrs)
    if func_suggestions:
        lines.append("  Function-level suggestions:")
        for s in func_suggestions:
            lines.append(f"     -> TRY: {s}")
        lines.append("")

    # Deduplicated attribute summary
    all_suggestions = collect_all_suggestions(
        alignment, expected_instrs, compiled_instrs
    )
    if all_suggestions:
        lines.append("  Suggested attributes:")
        # Extract attribute names from suggestion text
        attr_names = set()
        for s in all_suggestions:
            # Take the first word (the attribute name) from each suggestion
            name = s.split(" — ")[0].split("(")[0].strip()
            if name.startswith("reversed"):
                name = "msvc6_regalloc"
            attr_names.add(name)
        if attr_names:
            attrs = ", ".join(sorted(attr_names))
            lines.append(f"    __attribute__(({attrs}))")
    else:
        lines.append("  No suggestions — bytes match!")

    return "\n".join(lines)


def format_batch_line(
    symbol: str, status: str, exp_bytes: int, comp_bytes: int
) -> str:
    """Format one line of batch mode output."""
    sym_display = symbol[:60] + "..." if len(symbol) > 63 else symbol
    if status == "MATCH":
        return f"  {sym_display:<65s} MATCH      {comp_bytes:>6d} bytes"
    elif status == "MISMATCH":
        return (
            f"  {sym_display:<65s} MISMATCH   "
            f"{comp_bytes:>6d} vs {exp_bytes} bytes"
        )
    else:
        return f"  {sym_display:<65s} {status}"


# ============================================================
# CLI subcommands
# ============================================================

def cmd_diff(args: argparse.Namespace) -> int:
    """Execute the 'diff' subcommand."""
    objdump = find_tool("llvm-objdump", args.llvm_bin_dir)

    # Get compiled instructions
    compiled = disasm_obj(Path(args.obj_file), args.symbol, objdump)

    # Get reference instructions
    expected = _resolve_reference(args, objdump)

    # Align and display
    alignment = align_instructions(expected, compiled)
    print(format_diff(args.symbol, alignment, expected, compiled))
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    """Execute the 'batch' subcommand."""
    objdump = find_tool("llvm-objdump", args.llvm_bin_dir)
    obj_path = Path(args.obj_file)

    symbols = disasm_obj_symbols(obj_path, objdump)
    if not symbols:
        print(f"No function symbols found in {obj_path}", file=sys.stderr)
        return 1

    # Build ASM index if using --ref-asm-dir
    asm_index = None
    if args.ref_asm_dir:
        asm_index = build_symbol_index(Path(args.ref_asm_dir))

    print(f"msvc6_objdiff batch: {obj_path} ({len(symbols)} symbols)")
    print("=" * 78)

    counts = {"MATCH": 0, "MISMATCH": 0, "NO_REF": 0}

    # Pre-disassemble entire .o for efficiency
    full_result = subprocess.run(
        [objdump, "-dr", str(obj_path)],
        capture_output=True, text=True, timeout=60,
    )
    full_text = full_result.stdout

    for symbol in symbols:
        compiled = _parse_objdump_output(full_text, symbol)
        if not compiled:
            continue
        comp_bytes = sum(len(i.raw) for i in compiled)

        try:
            expected = _resolve_reference_for_batch(
                args, symbol, objdump, asm_index
            )
        except (ValueError, RuntimeError):
            print(format_batch_line(symbol, "NO_REF", 0, comp_bytes))
            counts["NO_REF"] += 1
            continue

        exp_bytes = sum(len(i.raw) for i in expected)

        # Check for match
        if len(expected) == len(compiled) and all(
            bytes_match(e.raw, c.raw, c.relocs)
            for e, c in zip(expected, compiled)
        ):
            status = "MATCH"
        else:
            status = "MISMATCH"

        print(format_batch_line(symbol, status, exp_bytes, comp_bytes))
        counts[status] += 1

    print("=" * 78)
    total = sum(counts.values())
    print(
        f"Summary: {counts['MATCH']} MATCH, {counts['MISMATCH']} MISMATCH, "
        f"{counts['NO_REF']} NO_REF ({total} total)"
    )
    return 0


def cmd_suggest(args: argparse.Namespace) -> int:
    """Execute the 'suggest' subcommand."""
    expected = parse_hex_instrs(args.expected_hex)
    compiled = parse_hex_instrs(args.compiled_hex)

    alignment = align_instructions(expected, compiled)
    print(format_diff("(hex input)", alignment, expected, compiled))
    return 0


def _resolve_reference(
    args: argparse.Namespace, objdump: str
) -> list[Instr]:
    """Resolve reference instructions from CLI args."""
    if args.ref_obj:
        ref_symbol = getattr(args, "ref_symbol", None) or args.symbol
        return disasm_obj(Path(args.ref_obj), ref_symbol, objdump)
    elif args.ref_bin:
        pe_path = Path(args.ref_bin)
        if args.rva is not None:
            image_base = pe_read_image_base(pe_path)
            start_va = image_base + args.rva
        elif args.start_address is not None:
            start_va = args.start_address
        else:
            raise ValueError(
                "--ref-bin requires either --rva or --start-address"
            )
        return disasm_pe(pe_path, start_va, args.size, objdump)
    elif args.ref_hex:
        return parse_hex_instrs(args.ref_hex)
    elif args.ref_asm:
        return ref_from_asm_file(Path(args.ref_asm), args.symbol)
    elif args.ref_asm_dir:
        return ref_from_asm_dir(Path(args.ref_asm_dir), args.symbol)
    else:
        raise ValueError(
            "No reference source specified. Use --ref-obj, --ref-bin, "
            "--ref-hex, --ref-asm, or --ref-asm-dir."
        )


def _resolve_reference_for_batch(
    args: argparse.Namespace,
    symbol: str,
    objdump: str,
    asm_index: dict | None,
) -> list[Instr]:
    """Resolve reference for a single symbol in batch mode."""
    if args.ref_asm_dir and asm_index is not None:
        return ref_from_asm_dir(Path(args.ref_asm_dir), symbol, asm_index)
    elif args.ref_obj:
        return disasm_obj(Path(args.ref_obj), symbol, objdump)
    raise ValueError("Batch mode requires --ref-asm-dir or --ref-obj")


# ============================================================
# Argument parsing
# ============================================================

def _parse_int(value: str) -> int:
    """Parse an integer from decimal or hex (0x...) string."""
    return int(value, 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="msvc6_objdiff",
        description=(
            "Byte-level function diff for MSVC 6.0 decompilation. "
            "Compares compiled COFF object files against reference bytes "
            "and suggests custom LLVM attributes."
        ),
    )
    parser.add_argument(
        "--llvm-bin-dir",
        help="Path to LLVM tool binaries (overrides LLVM_BIN_DIR env var)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- diff ---
    diff_parser = subparsers.add_parser(
        "diff", help="Compare one function between .o and reference"
    )
    diff_parser.add_argument("obj_file", help="Compiled .o file")
    diff_parser.add_argument("symbol", help="Function symbol name")
    _add_ref_args(diff_parser)

    # --- batch ---
    batch_parser = subparsers.add_parser(
        "batch", help="Compare all functions in .o against reference"
    )
    batch_parser.add_argument("obj_file", help="Compiled .o file")
    _add_ref_args(batch_parser)

    # --- suggest ---
    suggest_parser = subparsers.add_parser(
        "suggest",
        help="Standalone suggestion from hex bytes (no .o needed)",
    )
    suggest_parser.add_argument(
        "expected_hex",
        help='Expected bytes (space-separated instructions, e.g. "33c0 8a4108 c3")',
    )
    suggest_parser.add_argument(
        "compiled_hex",
        help='Compiled bytes (space-separated instructions)',
    )

    return parser


def _add_ref_args(parser: argparse.ArgumentParser) -> None:
    """Add reference source arguments to a subparser."""
    ref = parser.add_argument_group("reference source (choose one)")
    ref.add_argument("--ref-obj", metavar="FILE", help="Reference .o file")
    ref.add_argument(
        "--ref-symbol", metavar="SYM",
        help="Symbol in reference .o (defaults to same as compiled symbol)",
    )
    ref.add_argument("--ref-bin", metavar="FILE", help="Reference PE binary")
    ref.add_argument(
        "--rva", type=_parse_int,
        help="RVA in PE binary (ImageBase added automatically)",
    )
    ref.add_argument(
        "--start-address", type=_parse_int,
        help="Virtual address in PE binary (used directly)",
    )
    ref.add_argument(
        "--size", type=_parse_int,
        help="Function size in bytes (for --ref-bin; auto-detected if omitted)",
    )
    ref.add_argument(
        "--ref-hex", metavar="HEX",
        help='Hex bytes (space-separated instructions: "33c0 8a4108 c3")',
    )
    ref.add_argument(
        "--ref-asm", metavar="FILE", help="Single annotated ASM file",
    )
    ref.add_argument(
        "--ref-asm-dir", metavar="DIR",
        help="Directory of annotated ASM files (for batch mode)",
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "diff":
            return cmd_diff(args)
        elif args.command == "batch":
            return cmd_batch(args)
        elif args.command == "suggest":
            return cmd_suggest(args)
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except subprocess.TimeoutExpired:
        print("Error: llvm-objdump timed out", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

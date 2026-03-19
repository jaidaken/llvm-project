#!/usr/bin/env python3
"""msvc6_discover - Attribute discovery tool for MSVC 6.0 binaries.

Scans a PE binary to fingerprint MSVC 6.0 code generation patterns in each
function and recommends custom LLVM attributes BEFORE you write any C code.

Part of the bw1-decomp LLVM fork: https://github.com/jaidaken/llvm-project

Subcommands:
    scan      Auto-detect functions via prologue scanning, analyze all
    analyze   Analyze functions from a CSV function list
    exports   Analyze PE-exported functions only

Examples:
    # Auto-detect functions and analyze (needs llvm-objdump)
    msvc6_discover.py scan game.exe

    # Analyze from IDA/Ghidra CSV export
    msvc6_discover.py analyze game.exe --functions funcs.csv

    # Exported functions only, output as C header
    msvc6_discover.py exports game.dll --format header -o attrs.h

    # Fast mode: raw byte scanning, no llvm-objdump needed
    msvc6_discover.py scan game.exe --no-objdump --format csv > attrs.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

from msvc6_objdiff import (
    CALLEE_SAVE_PUSHES,
    OPCODE_SUGGESTIONS,
    Instr,
    _parse_objdump_output,
    find_tool,
    pe_read_image_base,
    pe_rva_to_offset,
)


# ============================================================
# Data types
# ============================================================

class FunctionEntry(NamedTuple):
    """A function to analyze."""
    name: str
    rva: int
    size: int | None  # None = auto-detect at ret+padding


class DiscoveryResult(NamedTuple):
    """Analysis result for one function."""
    name: str
    rva: int
    size: int
    attributes: dict[str, str | bool]
    opt_level: str  # "O1", "O2", "mixed", "unknown"
    notes: list[str]


# ============================================================
# PE .text section reader
# ============================================================

def pe_read_text_section(pe_path: Path) -> tuple[int, int, bytes]:
    """Read the .text section from a PE file.

    Returns (section_rva, section_virtual_size, raw_bytes).
    """
    with open(pe_path, "rb") as f:
        f.seek(0x3C)
        pe_offset = struct.unpack("<I", f.read(4))[0]
        f.seek(pe_offset + 4)
        _, num_sections, _, _, _, opt_size, _ = struct.unpack(
            "<HHIIIHH", f.read(20)
        )
        f.seek(pe_offset + 4 + 20 + opt_size)
        for _ in range(num_sections):
            name = f.read(8).rstrip(b"\x00")
            virt_size, virt_addr = struct.unpack("<II", f.read(8))
            raw_size, raw_ptr = struct.unpack("<II", f.read(8))
            f.read(16)
            if name in (b".text", b".TEXT", b"CODE"):
                f.seek(raw_ptr)
                data = f.read(raw_size)
                return virt_addr, virt_size, data
    raise ValueError(f"No .text section found in {pe_path}")


def pe_read_exports(pe_path: Path) -> list[FunctionEntry]:
    """Extract exported functions from the PE export directory."""
    entries = []
    with open(pe_path, "rb") as f:
        f.seek(0x3C)
        pe_offset = struct.unpack("<I", f.read(4))[0]
        f.seek(pe_offset + 4 + 20)
        opt_magic = struct.unpack("<H", f.read(2))[0]
        if opt_magic == 0x10B:
            num_dd_offset = pe_offset + 4 + 20 + 92
            export_dd_offset = pe_offset + 4 + 20 + 96
        elif opt_magic == 0x20B:
            num_dd_offset = pe_offset + 4 + 20 + 108
            export_dd_offset = pe_offset + 4 + 20 + 112
        else:
            return entries
        f.seek(num_dd_offset)
        num_dd = struct.unpack("<I", f.read(4))[0]
        if num_dd < 1:
            return entries
        f.seek(export_dd_offset)
        export_rva, export_size = struct.unpack("<II", f.read(8))
        if export_rva == 0 or export_size == 0:
            return entries
        export_offset = pe_rva_to_offset(pe_path, export_rva)
        if export_offset is None:
            return entries
        f.seek(export_offset + 24)
        num_funcs, num_names = struct.unpack("<II", f.read(8))
        addr_table_rva, name_ptr_rva, ordinal_rva = struct.unpack(
            "<III", f.read(12)
        )
        addr_off = pe_rva_to_offset(pe_path, addr_table_rva)
        name_off = pe_rva_to_offset(pe_path, name_ptr_rva)
        if addr_off is None or name_off is None:
            return entries
        f.seek(name_off)
        name_ptrs = [
            struct.unpack("<I", f.read(4))[0] for _ in range(num_names)
        ]
        ordinal_off = pe_rva_to_offset(pe_path, ordinal_rva)
        if ordinal_off is None:
            return entries
        f.seek(ordinal_off)
        ordinals = [
            struct.unpack("<H", f.read(2))[0] for _ in range(num_names)
        ]
        f.seek(addr_off)
        func_rvas = [
            struct.unpack("<I", f.read(4))[0] for _ in range(num_funcs)
        ]
        for i, name_rva in enumerate(name_ptrs):
            noff = pe_rva_to_offset(pe_path, name_rva)
            if noff is None:
                continue
            f.seek(noff)
            name_bytes = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00" or ch == b"":
                    break
                name_bytes += ch
            name = name_bytes.decode("ascii", errors="replace")
            func_rva = func_rvas[ordinals[i]] if ordinals[i] < num_funcs else 0
            # Skip forwarded exports (RVA points inside export directory)
            if export_rva <= func_rva < export_rva + export_size:
                continue
            if func_rva:
                entries.append(FunctionEntry(name=name, rva=func_rva, size=None))
    return entries


# ============================================================
# Function list parsing
# ============================================================

def parse_function_list(
    path: Path, image_base: int | None = None
) -> list[FunctionEntry]:
    """Parse a function list from CSV. Auto-detects column layout."""
    with open(path, "r", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t ;")
        has_header = csv.Sniffer().has_header(sample)
        reader = csv.reader(f, dialect)
        if has_header:
            header = [h.strip().lower() for h in next(reader)]
        else:
            header = None
        entries = []
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            entry = _parse_row(row, header, image_base)
            if entry:
                entries.append(entry)
    return entries


def _parse_row(
    row: list[str], header: list[str] | None, image_base: int | None
) -> FunctionEntry | None:
    """Parse a single CSV row into a FunctionEntry."""
    def _get(names: list[str]) -> str | None:
        if header:
            for n in names:
                if n in header:
                    idx = header.index(n)
                    return row[idx].strip() if idx < len(row) else None
        return None

    name = _get(["name", "function", "symbol", "func_name"]) or ""
    rva_str = _get(["rva", "start_rva", "offset"])
    addr_str = _get(["address", "start", "va", "start_address"])
    size_str = _get(["size", "length", "bytes"])
    end_str = _get(["end", "end_rva", "end_address"])

    # Fallback: positional columns if no header
    if not header and len(row) >= 2:
        if not rva_str and not addr_str:
            rva_str = row[0].strip()
            size_str = row[1].strip() if len(row) >= 2 else None
            name = row[2].strip() if len(row) >= 3 else ""

    rva: int | None = None
    if rva_str:
        rva = int(rva_str, 0)
    elif addr_str:
        addr = int(addr_str, 0)
        if image_base is not None:
            rva = addr - image_base
        else:
            rva = addr

    if rva is None:
        return None

    size: int | None = None
    if size_str:
        size = int(size_str.rstrip("hH"), 0)
    elif end_str:
        end = int(end_str, 0)
        size = end - rva

    if not name:
        name = f"func_{rva:08x}"

    return FunctionEntry(name=name, rva=rva, size=size)


# ============================================================
# Prologue-based function scanner
# ============================================================

def scan_for_functions(text_data: bytes, text_rva: int) -> list[FunctionEntry]:
    """Scan .text section for MSVC 6.0 function prologues.

    Finds functions by looking for common prologue patterns after int3/nop
    padding.
    """
    entries = []
    i = 0
    n = len(text_data)

    while i < n - 4:
        # Skip non-padding
        if text_data[i] not in (0xCC, 0x90):
            i += 1
            continue

        # Find end of padding
        pad_start = i
        while i < n and text_data[i] in (0xCC, 0x90):
            i += 1
        if i >= n - 2:
            break

        # Check for prologue at this position
        rva = text_rva + i
        is_prologue = False

        # push ebp; mov ebp, esp (55 8B EC)
        if i + 2 < n and text_data[i:i + 3] == b"\x55\x8b\xec":
            is_prologue = True
        # push esi/edi/ebx at aligned-ish address
        elif text_data[i] in (0x53, 0x55, 0x56, 0x57):
            is_prologue = True
        # sub esp, imm8 (83 EC XX)
        elif i + 2 < n and text_data[i:i + 2] == b"\x83\xec":
            is_prologue = True
        # mov eax, [ecx+N] or similar (common for no-prologue leaf funcs)
        elif text_data[i] == 0x8B and i + 1 < n:
            is_prologue = True

        if is_prologue and (i - pad_start) >= 1:
            # Estimate size: scan until next padding or max 65536
            end = i + 1
            while end < min(i + 65536, n):
                if text_data[end] in (0xC3, 0xC2):
                    # Found ret, check for padding after
                    ret_end = end + 1
                    if text_data[end] == 0xC2:
                        ret_end = end + 3
                    if ret_end < n and text_data[ret_end] in (0xCC, 0x90):
                        end = ret_end
                        break
                end += 1
            size = end - i
            entries.append(
                FunctionEntry(name=f"func_{rva:08x}", rva=rva, size=size)
            )
            i = end
        # else: just continue scanning

    return entries


# ============================================================
# Function raw byte extraction
# ============================================================

def extract_raw_bytes(
    text_data: bytes, text_rva: int, entry: FunctionEntry
) -> bytes:
    """Extract raw function bytes from the .text section."""
    offset = entry.rva - text_rva
    if offset < 0 or offset >= len(text_data):
        return b""
    if entry.size is not None:
        return text_data[offset:offset + entry.size]
    # Auto-detect size: read until ret + padding, max 65536
    end = offset
    limit = min(offset + 65536, len(text_data))
    while end < limit:
        if text_data[end] in (0xC3, 0xC2):
            ret_end = end + (3 if text_data[end] == 0xC2 else 1)
            if ret_end < len(text_data) and text_data[ret_end] in (0xCC, 0x90):
                return text_data[offset:ret_end]
            # Might not have padding, keep going
        end += 1
    return text_data[offset:min(offset + 256, len(text_data))]


# ============================================================
# Pattern detectors
# ============================================================

def detect_callee_saves(instrs: list[Instr]) -> tuple[str | None, str | bool, str]:
    """Detect prologue callee-save register pattern."""
    saves = []
    for instr in instrs[:6]:
        if len(instr.raw) == 1 and instr.raw[0] in CALLEE_SAVE_PUSHES:
            saves.append(CALLEE_SAVE_PUSHES[instr.raw[0]])
        else:
            break

    # Check for frame pointer prologue: push ebp; mov ebp, esp
    if (
        len(instrs) >= 2
        and instrs[0].raw == b"\x55"
        and instrs[1].raw[:2] == b"\x8b\xec"
    ):
        return None, True, f"frame pointer prologue (push ebp; mov ebp, esp)"

    if not saves:
        return "no_callee_saves", True, "no callee-saved register pushes"
    return "forced_callee_saves", ",".join(saves), f"pushes {','.join(saves)}"


def detect_ret_cleanup(
    instrs: list[Instr], raw_bytes: bytes
) -> tuple[str | None, str | bool, str]:
    """Detect epilogue ret/ret N/jmp pattern."""
    if not instrs:
        # Fallback to raw bytes
        if raw_bytes and raw_bytes[-1] == 0xC3:
            return None, True, "bare ret"
        if len(raw_bytes) >= 3 and raw_bytes[-3] == 0xC2:
            n = struct.unpack("<H", raw_bytes[-2:])[0]
            return "ret_cleanup_override", str(n), f"ret {n}"
        return None, True, "unknown epilogue"

    last = instrs[-1]
    if last.raw[0:1] == b"\xc3":
        return None, True, "bare ret"
    if last.raw[0:1] == b"\xc2" and len(last.raw) >= 3:
        n = struct.unpack("<H", last.raw[1:3])[0]
        return "ret_cleanup_override", str(n), f"ret {n}"
    if last.raw[0:1] in (b"\xe9", b"\xeb") or last.raw[:2] == b"\xff\x25":
        return "no_ret", True, "ends with jmp (no ret)"
    return None, True, "non-standard epilogue"


def detect_expand_movzx(raw_bytes: bytes) -> tuple[int, str]:
    """Count xor reg,reg + mov byte_reg,[mem] patterns (expand_movzx indicator)."""
    count = 0
    # Self-xor patterns: 33 C0 (eax), 33 C9 (ecx), 33 D2 (edx)
    # Also 31 C0, 31 C9, 31 D2 (reversed encoding)
    xor_self = {
        b"\x33\xc0", b"\x33\xc9", b"\x33\xd2",
        b"\x31\xc0", b"\x31\xc9", b"\x31\xd2",
    }
    i = 0
    while i < len(raw_bytes) - 3:
        if raw_bytes[i:i + 2] in xor_self:
            # Check if next non-trivial byte is a byte MOV (8A = mov r8, r/m8)
            j = i + 2
            if j < len(raw_bytes) and raw_bytes[j] == 0x8A:
                count += 1
                i = j + 2
                continue
        i += 1
    note = f"{count}x xor+mov byte expansion" if count else ""
    return count, note


def detect_or_minus_one(raw_bytes: bytes) -> tuple[int, str]:
    """Count or reg,-1 patterns (83 C8/C9/CA/CB FF)."""
    count = 0
    for i in range(len(raw_bytes) - 2):
        if (
            raw_bytes[i] == 0x83
            and raw_bytes[i + 1] in (0xC8, 0xC9, 0xCA, 0xCB)
            and raw_bytes[i + 2] == 0xFF
        ):
            count += 1
    note = f"{count}x or reg,-1" if count else ""
    return count, note


def detect_pop_cleanup(
    instrs: list[Instr], raw_bytes: bytes
) -> tuple[int, str]:
    """Count pop ecx/edx used as stack cleanup (not callee-save restore)."""
    count = 0
    # In structured mode, skip the first few instructions (prologue)
    start = 0
    if instrs:
        for idx, instr in enumerate(instrs):
            if len(instr.raw) != 1 or instr.raw[0] not in CALLEE_SAVE_PUSHES:
                start = idx
                break
        for instr in instrs[start:]:
            if instr.raw == b"\x59":  # pop ecx
                count += 1
    else:
        # Raw bytes: count 59 that aren't at the very start
        for i in range(4, len(raw_bytes)):
            if raw_bytes[i] == 0x59:
                count += 1
    note = f"{count}x pop ecx cleanup" if count else ""
    return count, note


def detect_suppress_fp_imm(raw_bytes: bytes) -> tuple[bool, str]:
    """Detect fld [mem] usage (D9 05/0D/... for 32-bit loads) without fldz/fld1."""
    has_fld_mem = False
    has_fldz = False
    has_fld1 = False
    for i in range(len(raw_bytes) - 1):
        if raw_bytes[i] == 0xD9:
            modrm = raw_bytes[i + 1]
            if modrm == 0xEE:
                has_fldz = True
            elif modrm == 0xE8:
                has_fld1 = True
            elif (modrm & 0xC7) == 0x05 or (modrm & 0xC0) != 0xC0:
                # Memory operand fld
                reg = (modrm >> 3) & 7
                if reg == 0:  # fld
                    has_fld_mem = True
    if has_fld_mem and not has_fldz and not has_fld1:
        return True, "fld [mem] for float constants (no fldz/fld1)"
    return False, ""


def detect_no_test_sete_fold(raw_bytes: bytes) -> tuple[int, str]:
    """Count not al + shr + and eax,1 patterns."""
    count = 0
    for i in range(len(raw_bytes) - 5):
        if (
            raw_bytes[i:i + 2] == b"\xf6\xd0"  # not al
            and raw_bytes[i + 2] == 0xC1  # shr (C1 E8 = shr eax)
            and raw_bytes[i + 3] in (0xE8, 0xE9, 0xEA)
        ):
            count += 1
    note = f"{count}x not+shr+and pattern" if count else ""
    return count, note


def detect_prefer_neg_sbb(raw_bytes: bytes) -> tuple[int, str]:
    """Count neg + sbb + inc patterns."""
    count = 0
    for i in range(len(raw_bytes) - 4):
        if (
            raw_bytes[i] == 0xF7
            and raw_bytes[i + 1] in (0xD8, 0xD9, 0xDA)  # neg eax/ecx/edx
        ):
            # Look for sbb within next 4 bytes
            for j in range(i + 2, min(i + 6, len(raw_bytes) - 1)):
                if raw_bytes[j] == 0x1B:  # sbb r, r/m
                    count += 1
                    break
    note = f"{count}x neg+sbb pattern" if count else ""
    return count, note


def detect_prefer_sete_ecx(raw_bytes: bytes) -> tuple[int, str]:
    """Count sete cl (0F 94 C1) patterns."""
    count = 0
    for i in range(len(raw_bytes) - 2):
        if raw_bytes[i:i + 3] == b"\x0f\x94\xc1":
            count += 1
    note = f"{count}x sete cl" if count else ""
    return count, note


def detect_prefer_xor8(raw_bytes: bytes) -> tuple[int, str]:
    """Count xor al,al (32 C0) patterns."""
    count = 0
    for i in range(len(raw_bytes) - 1):
        if raw_bytes[i:i + 2] == b"\x32\xc0":
            count += 1
    note = f"{count}x xor al,al (8-bit)" if count else ""
    return count, note


def detect_prefer_inc_dec_byte(raw_bytes: bytes) -> tuple[int, str]:
    """Count inc/dec byte [mem] (FE /0 or FE /1) patterns."""
    count = 0
    for i in range(len(raw_bytes) - 1):
        if raw_bytes[i] == 0xFE:
            modrm = raw_bytes[i + 1]
            reg = (modrm >> 3) & 7
            mod = (modrm >> 6) & 3
            if reg in (0, 1) and mod != 3:  # /0=inc, /1=dec, mod!=3 means memory
                count += 1
    note = f"{count}x inc/dec byte [mem]" if count else ""
    return count, note


def detect_prefer_fmul_mem(raw_bytes: bytes) -> tuple[int, str]:
    """Count fmul [mem] (D8 /1 with memory operand) patterns."""
    count = 0
    for i in range(len(raw_bytes) - 1):
        if raw_bytes[i] == 0xD8:
            modrm = raw_bytes[i + 1]
            reg = (modrm >> 3) & 7
            mod = (modrm >> 6) & 3
            if reg == 1 and mod != 3:  # fmul with memory operand
                count += 1
    note = f"{count}x fmul [mem]" if count else ""
    return count, note


def detect_reversed_encoding(instrs: list[Instr]) -> tuple[int, int, str]:
    """Count reversed vs standard ModR/M encodings for reg-reg operations.

    Returns (reversed_count, total_reg_reg, note).
    """
    reversed_pairs = {
        (exp, got) for (exp, got) in OPCODE_SUGGESTIONS
        if exp != got  # skip xor width entries
        and abs(exp - got) == 2  # reversed pairs differ by 2
    }
    # Build a set of "msvc direction" opcodes (the ones MSVC 6.0 prefers)
    msvc_opcodes = set()
    for exp, got in reversed_pairs:
        # In OPCODE_SUGGESTIONS, both directions are listed.
        # MSVC prefers the r, r/m encoding: 03, 0B, 13, 1B, 23, 2B, 33, 3B, 8B
        if exp in (0x03, 0x0B, 0x13, 0x1B, 0x23, 0x2B, 0x33, 0x3B, 0x8B):
            msvc_opcodes.add(exp)
    modern_opcodes = set()
    for exp, got in reversed_pairs:
        if exp in (0x01, 0x09, 0x11, 0x19, 0x21, 0x29, 0x31, 0x39, 0x89):
            modern_opcodes.add(exp)

    reversed_count = 0
    total = 0

    for instr in instrs:
        if len(instr.raw) < 2:
            continue
        opcode = instr.raw[0]
        modrm = instr.raw[1]
        mod = (modrm >> 6) & 3
        if mod != 3:  # Not reg-reg
            continue
        if opcode in msvc_opcodes:
            reversed_count += 1
            total += 1
        elif opcode in modern_opcodes:
            total += 1

    note = ""
    if total > 0:
        pct = reversed_count * 100 // total
        note = f"{reversed_count}/{total} reg-reg ops use reversed encoding ({pct}%)"
    return reversed_count, total, note


def detect_trailing_bytes(
    raw_bytes: bytes, instrs: list[Instr]
) -> tuple[bool, bool, str]:
    """Detect dead code after ret.

    Returns (has_trailing_bytes, has_trailing_asm, note).
    """
    # Find the ret position in raw bytes
    ret_pos = None
    for i in range(len(raw_bytes) - 1, -1, -1):
        if raw_bytes[i] in (0xC3, 0xC2):
            ret_end = i + (3 if raw_bytes[i] == 0xC2 else 1)
            ret_pos = ret_end
            break

    if ret_pos is None or ret_pos >= len(raw_bytes):
        return False, False, ""

    trailing = raw_bytes[ret_pos:]
    # Strip padding at the end
    end = len(trailing)
    while end > 0 and trailing[end - 1] in (0xCC, 0x90):
        end -= 1
    trailing = trailing[:end]

    if not trailing:
        return False, False, ""

    # Check if trailing bytes contain call/jmp (need relocations = trailing_asm)
    has_call = any(
        trailing[i] in (0xE8, 0xFF) for i in range(len(trailing))
    )
    note = f"{len(trailing)} bytes of dead code after ret"
    if has_call:
        note += " (contains call/jmp → trailing_asm)"
        return False, True, note
    return True, False, note


def detect_fpu_comparison(raw_bytes: bytes) -> tuple[int, str]:
    """Count fcomp+fnstsw+test ah patterns (MSVC 6.0 FPU comparison style)."""
    count = 0
    for i in range(len(raw_bytes) - 3):
        # fnstsw ax = DF E0
        if raw_bytes[i:i + 2] == b"\xdf\xe0":
            # Check for test ah nearby (F6 C4 XX)
            for j in range(i + 2, min(i + 6, len(raw_bytes) - 2)):
                if raw_bytes[j] == 0xF6 and raw_bytes[j + 1] == 0xC4:
                    count += 1
                    break
    note = f"{count}x fcomp/fnstsw/test ah (MSVC FPU comparison)" if count else ""
    return count, note


# ============================================================
# Pattern dispatcher
# ============================================================

def detect_patterns(
    instrs: list[Instr], raw_bytes: bytes
) -> tuple[dict[str, str | bool], list[str]]:
    """Run all pattern detectors on a function."""
    attrs: dict[str, str | bool] = {}
    notes: list[str] = []

    def _add(name: str | None, value: str | bool, note: str) -> None:
        if name:
            attrs[name] = value
        if note:
            notes.append(note)

    # Prologue
    if instrs:
        cs_name, cs_val, cs_note = detect_callee_saves(instrs)
        _add(cs_name, cs_val, cs_note)

    # Epilogue
    ret_name, ret_val, ret_note = detect_ret_cleanup(instrs, raw_bytes)
    _add(ret_name, ret_val, ret_note)

    # Encoding patterns (work on raw bytes)
    movzx_count, movzx_note = detect_expand_movzx(raw_bytes)
    if movzx_count:
        _add("expand_movzx", True, movzx_note)

    xor8_count, xor8_note = detect_prefer_xor8(raw_bytes)
    if xor8_count:
        _add("prefer_xor8", True, xor8_note)

    or_count, or_note = detect_or_minus_one(raw_bytes)
    if or_count:
        _add("prefer_or_minus_one", True, or_note)

    pop_count, pop_note = detect_pop_cleanup(instrs, raw_bytes)
    if pop_count:
        _add("prefer_pop_cleanup", True, pop_note)

    fp_imm, fp_note = detect_suppress_fp_imm(raw_bytes)
    if fp_imm:
        _add("suppress_fp_imm", True, fp_note)

    tsf_count, tsf_note = detect_no_test_sete_fold(raw_bytes)
    if tsf_count:
        _add("no_test_sete_fold", True, tsf_note)

    ns_count, ns_note = detect_prefer_neg_sbb(raw_bytes)
    if ns_count:
        _add("prefer_neg_sbb", True, ns_note)

    sete_count, sete_note = detect_prefer_sete_ecx(raw_bytes)
    if sete_count:
        _add("prefer_sete_ecx", True, sete_note)

    idb_count, idb_note = detect_prefer_inc_dec_byte(raw_bytes)
    if idb_count:
        _add("prefer_inc_dec_byte", True, idb_note)

    fmul_count, fmul_note = detect_prefer_fmul_mem(raw_bytes)
    if fmul_count:
        _add("prefer_fmul_mem", True, fmul_note)

    # Reversed encoding (needs structured instrs for accuracy)
    if instrs:
        rev_count, rev_total, rev_note = detect_reversed_encoding(instrs)
        if rev_total > 0 and rev_count > rev_total // 2:
            _add("msvc6_regalloc", True, rev_note)
        elif rev_note:
            notes.append(rev_note)

    # Trailing bytes
    has_tb, has_ta, trail_note = detect_trailing_bytes(raw_bytes, instrs)
    if has_ta:
        _add("trailing_asm", True, trail_note)
    elif has_tb:
        _add("trailing_bytes", True, trail_note)

    # FPU comparison (informational)
    fpu_count, fpu_note = detect_fpu_comparison(raw_bytes)
    if fpu_count:
        notes.append(fpu_note)

    return attrs, notes


# ============================================================
# Optimization level classification
# ============================================================

O1_INDICATORS = {
    "prefer_or_minus_one": 3,
    "prefer_inc_dec_byte": 2,
    "prefer_pop_cleanup": 3,
    "prefer_neg_sbb": 2,
}

O2_INDICATORS = {
    "expand_movzx": 3,
    "prefer_xor8": 2,
    "no_test_sete_fold": 2,
}


def classify_opt_level(attributes: dict[str, str | bool]) -> str:
    """Classify function as O1/O2/mixed/unknown."""
    o1_score = sum(w for a, w in O1_INDICATORS.items() if a in attributes)
    o2_score = sum(w for a, w in O2_INDICATORS.items() if a in attributes)
    if o1_score > 0 and o2_score > 0:
        return "mixed"
    if o1_score > 0:
        return "O1"
    if o2_score > 0:
        return "O2"
    return "unknown"


# ============================================================
# Orchestration
# ============================================================

def analyze_function(
    entry: FunctionEntry,
    instrs: list[Instr],
    raw_bytes: bytes,
) -> DiscoveryResult:
    """Analyze a single function."""
    attrs, notes = detect_patterns(instrs, raw_bytes)
    opt_level = classify_opt_level(attrs)
    return DiscoveryResult(
        name=entry.name,
        rva=entry.rva,
        size=len(raw_bytes),
        attributes=attrs,
        opt_level=opt_level,
        notes=notes,
    )


def analyze_binary(
    pe_path: Path,
    entries: list[FunctionEntry],
    objdump: str | None = None,
) -> list[DiscoveryResult]:
    """Analyze all functions in a PE binary."""
    image_base = pe_read_image_base(pe_path)
    text_rva, text_size, text_data = pe_read_text_section(pe_path)

    # If objdump available, disassemble full binary once
    full_disasm: str | None = None
    if objdump:
        try:
            text_va_start = image_base + text_rva
            text_va_end = text_va_start + text_size
            result = subprocess.run(
                [
                    objdump, "-d",
                    f"--start-address=0x{text_va_start:x}",
                    f"--stop-address=0x{text_va_end:x}",
                    str(pe_path),
                ],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                full_disasm = result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    results = []
    for entry in entries:
        raw_bytes = extract_raw_bytes(text_data, text_rva, entry)
        if not raw_bytes:
            continue

        instrs: list[Instr] = []
        if full_disasm:
            va = image_base + entry.rva
            size = entry.size or len(raw_bytes)
            # Parse instructions for this address range from full disasm
            instrs = _extract_instrs_from_full_disasm(
                full_disasm, va, size
            )

        result = analyze_function(entry, instrs, raw_bytes)
        results.append(result)

    return results


def _extract_instrs_from_full_disasm(
    full_text: str, start_va: int, size: int
) -> list[Instr]:
    """Extract instructions for an address range from full disassembly output."""
    instrs = []
    in_range = False
    end_va = start_va + size

    for line in full_text.splitlines():
        # Quick check for instruction lines (starts with whitespace + hex addr)
        stripped = line.lstrip()
        if not stripped or stripped[0] not in "0123456789abcdef":
            continue
        colon_idx = stripped.find(":")
        if colon_idx < 0:
            continue
        try:
            addr = int(stripped[:colon_idx], 16)
        except ValueError:
            continue

        if addr >= start_va and addr < end_va:
            in_range = True
            # Parse instruction bytes
            rest = stripped[colon_idx + 1:].strip()
            parts = rest.split("\t", 1)
            hex_part = parts[0].strip()
            asm_text = parts[1].strip() if len(parts) > 1 else ""
            try:
                raw = bytes.fromhex(hex_part.replace(" ", ""))
                instrs.append(Instr(raw=raw, text=asm_text, relocs=frozenset()))
            except ValueError:
                pass
        elif in_range and addr >= end_va:
            break

    return instrs


# ============================================================
# Output formatters
# ============================================================

ALL_ATTRS = [
    "no_callee_saves", "forced_callee_saves", "no_ret", "ret_cleanup_override",
    "expand_movzx", "prefer_xor8", "suppress_fp_imm", "prefer_or_minus_one",
    "prefer_inc_dec_byte", "no_test_sete_fold", "prefer_neg_sbb",
    "prefer_sete_ecx", "prefer_pop_cleanup", "prefer_fmul_mem",
    "msvc6_regalloc", "trailing_bytes", "trailing_asm", "no_bool_mask",
]


def format_attr_c(attrs: dict[str, str | bool]) -> str:
    """Format attributes as C __attribute__ syntax."""
    parts = []
    for name, value in sorted(attrs.items()):
        if isinstance(value, str):
            parts.append(f'{name}("{value}")')
        else:
            parts.append(name)
    if not parts:
        return ""
    return f"__attribute__(({', '.join(parts)}))"


def format_report(results: list[DiscoveryResult], pe_name: str = "") -> str:
    """Format human-readable analysis report."""
    lines = []
    lines.append(f"msvc6_discover: {pe_name} ({len(results)} functions)")
    lines.append("=" * 70)
    lines.append("")

    for r in results:
        attr_str = ", ".join(
            f'{k}("{v}")' if isinstance(v, str) else k
            for k, v in sorted(r.attributes.items())
        )
        lines.append(f"{r.name} (RVA 0x{r.rva:08x}, {r.size} bytes) [{r.opt_level}]")
        lines.append(f"  Attributes: {attr_str or '(none)'}")
        if r.notes:
            lines.append(f"  Notes: {'; '.join(r.notes)}")
        c_attr = format_attr_c(r.attributes)
        if c_attr:
            lines.append(f"  C: {c_attr}")
        lines.append("")

    # Summary
    lines.append("=" * 70)
    opt_counts = {"O1": 0, "O2": 0, "mixed": 0, "unknown": 0}
    attr_counts: dict[str, int] = {}
    for r in results:
        opt_counts[r.opt_level] = opt_counts.get(r.opt_level, 0) + 1
        for a in r.attributes:
            attr_counts[a] = attr_counts.get(a, 0) + 1

    lines.append(f"Summary: {len(results)} functions analyzed")
    opt_parts = []
    for level in ("O1", "O2", "mixed", "unknown"):
        c = opt_counts[level]
        pct = c * 100 / len(results) if results else 0
        opt_parts.append(f"{level}: {c} ({pct:.1f}%)")
    lines.append(f"  {', '.join(opt_parts)}")

    top_attrs = sorted(attr_counts.items(), key=lambda x: -x[1])[:10]
    if top_attrs:
        top_str = ", ".join(f"{a} ({c})" for a, c in top_attrs)
        lines.append(f"  Top attributes: {top_str}")

    return "\n".join(lines)


def format_csv_output(results: list[DiscoveryResult]) -> str:
    """Format machine-readable CSV output."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    header = ["name", "rva", "size", "opt_level"] + ALL_ATTRS
    writer.writerow(header)
    for r in results:
        row = [r.name, f"0x{r.rva:x}", r.size, r.opt_level]
        for attr_name in ALL_ATTRS:
            val = r.attributes.get(attr_name, "")
            if val is True:
                row.append("true")
            elif val:
                row.append(str(val))
            else:
                row.append("")
        writer.writerow(row)
    return buf.getvalue()


def format_c_header(results: list[DiscoveryResult], pe_name: str = "") -> str:
    """Format C header with attribute annotations."""
    lines = [
        f"// Generated by msvc6_discover.py from {pe_name}",
        f"// {len(results)} functions analyzed",
        "",
    ]
    for r in results:
        c_attr = format_attr_c(r.attributes)
        lines.append(f"// {r.name} (RVA 0x{r.rva:08x}, {r.size} bytes) [{r.opt_level}]")
        if c_attr:
            lines.append(c_attr)
        lines.append(f"void {_safe_c_name(r.name)}(void); // TODO: correct signature")
        lines.append("")
    return "\n".join(lines)


def _safe_c_name(name: str) -> str:
    """Convert a mangled symbol name to a safe C identifier."""
    safe = name.replace("?", "").replace("@", "_").replace("$", "_")
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe or "unknown"


# ============================================================
# CLI
# ============================================================

def _parse_int(value: str) -> int:
    return int(value, 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="msvc6_discover",
        description="Attribute discovery tool for MSVC 6.0 binaries.",
    )
    parser.add_argument("--llvm-bin-dir", help="Path to LLVM tool binaries")
    parser.add_argument(
        "--format", choices=["report", "csv", "header"], default="report",
        help="Output format (default: report)",
    )
    parser.add_argument("-o", "--output", help="Write to file instead of stdout")
    parser.add_argument(
        "--no-objdump", action="store_true",
        help="Skip llvm-objdump, use raw byte scanning only",
    )
    parser.add_argument(
        "--min-size", type=int, default=4,
        help="Skip functions smaller than N bytes (default: 4)",
    )
    parser.add_argument(
        "--image-base", type=_parse_int,
        help="Override PE image base (for rebased binaries)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_p = subparsers.add_parser(
        "scan", help="Auto-detect functions via prologue scanning",
    )
    scan_p.add_argument("pe_file", help="PE binary to analyze")

    analyze_p = subparsers.add_parser(
        "analyze", help="Analyze functions from a CSV function list",
    )
    analyze_p.add_argument("pe_file", help="PE binary to analyze")
    analyze_p.add_argument(
        "--functions", "-f", required=True,
        help="CSV function list (columns: name, rva, size)",
    )

    exports_p = subparsers.add_parser(
        "exports", help="Analyze PE-exported functions only",
    )
    exports_p.add_argument("pe_file", help="PE binary to analyze")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    pe_path = Path(args.pe_file)

    if not pe_path.exists():
        print(f"Error: {pe_path} not found", file=sys.stderr)
        return 1

    objdump: str | None = None
    if not args.no_objdump:
        objdump = find_tool("llvm-objdump", args.llvm_bin_dir)

    try:
        if args.command == "scan":
            text_rva, text_size, text_data = pe_read_text_section(pe_path)
            entries = scan_for_functions(text_data, text_rva)
        elif args.command == "analyze":
            image_base = args.image_base or pe_read_image_base(pe_path)
            entries = parse_function_list(Path(args.functions), image_base)
        elif args.command == "exports":
            entries = pe_read_exports(pe_path)
        else:
            return 1

        entries = [e for e in entries if e.size is None or e.size >= args.min_size]

        if not entries:
            print("No functions found to analyze.", file=sys.stderr)
            return 1

        results = analyze_binary(pe_path, entries, objdump)

        pe_name = pe_path.name
        if args.format == "report":
            output = format_report(results, pe_name)
        elif args.format == "csv":
            output = format_csv_output(results)
        elif args.format == "header":
            output = format_c_header(results, pe_name)
        else:
            output = format_report(results, pe_name)

        if args.output:
            Path(args.output).write_text(output)
            print(f"Written to {args.output}", file=sys.stderr)
        else:
            print(output)

    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except subprocess.TimeoutExpired:
        print("Error: llvm-objdump timed out", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

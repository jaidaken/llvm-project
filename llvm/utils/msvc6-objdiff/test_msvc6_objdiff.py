#!/usr/bin/env python3
"""Tests for msvc6_objdiff.py.

Tests are self-contained — no external files or LLVM tools required for the
core suggestion/alignment/parsing tests. Tests that require llvm-objdump are
skipped if it's not available.

Run:
    python3 -m pytest test_msvc6_objdiff.py -v
    python3 -m unittest test_msvc6_objdiff -v
"""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from msvc6_objdiff import (
    CALLEE_SAVE_PUSHES,
    OPCODE_SUGGESTIONS,
    PATTERN_SUGGESTIONS,
    Instr,
    align_instructions,
    build_symbol_index,
    bytes_match,
    collect_all_suggestions,
    detect_asm_format,
    format_batch_line,
    format_diff,
    masked_hex,
    parse_asm_instrs,
    parse_hex_instrs,
    pe_read_image_base,
    pe_rva_to_offset,
    suggest_fixes,
    suggest_function_fixes,
    _parse_objdump_output,
)


def _instr(hex_str: str, text: str = "", relocs: frozenset = frozenset()) -> Instr:
    """Helper to create Instr from hex string."""
    return Instr(raw=bytes.fromhex(hex_str), text=text, relocs=relocs)


# ============================================================
# Hex parsing
# ============================================================


class TestParseHexInstrs(unittest.TestCase):
    def test_basic(self):
        instrs = parse_hex_instrs("33c0 8a4108 c3")
        self.assertEqual(len(instrs), 3)
        self.assertEqual(instrs[0].raw, bytes.fromhex("33c0"))
        self.assertEqual(instrs[1].raw, bytes.fromhex("8a4108"))
        self.assertEqual(instrs[2].raw, bytes.fromhex("c3"))

    def test_single_instruction(self):
        instrs = parse_hex_instrs("c3")
        self.assertEqual(len(instrs), 1)
        self.assertEqual(instrs[0].raw, b"\xc3")

    def test_empty_string(self):
        instrs = parse_hex_instrs("  ")
        self.assertEqual(len(instrs), 0)

    def test_invalid_hex(self):
        with self.assertRaises(ValueError):
            parse_hex_instrs("zzzz")

    def test_no_relocs(self):
        instrs = parse_hex_instrs("e800000000")
        self.assertEqual(instrs[0].relocs, frozenset())


# ============================================================
# Opcode suggestions
# ============================================================


class TestOpcodeSuggestions(unittest.TestCase):
    def test_reversed_mov(self):
        suggestions = suggest_fixes(bytes([0x8B, 0xC1]), bytes([0x89, 0xC8]))
        self.assertTrue(any("MOV32rr_REV" in s for s in suggestions))

    def test_reversed_xor(self):
        suggestions = suggest_fixes(bytes([0x33, 0xC0]), bytes([0x31, 0xC0]))
        self.assertTrue(any("XOR32rr_REV" in s for s in suggestions))

    def test_reversed_add(self):
        suggestions = suggest_fixes(bytes([0x03, 0xC1]), bytes([0x01, 0xC8]))
        self.assertTrue(any("reversed ADD" in s for s in suggestions))

    def test_reversed_or(self):
        suggestions = suggest_fixes(bytes([0x0B, 0xC1]), bytes([0x09, 0xC8]))
        self.assertTrue(any("reversed OR" in s for s in suggestions))

    def test_reversed_sub(self):
        suggestions = suggest_fixes(bytes([0x2B, 0xC1]), bytes([0x29, 0xC8]))
        self.assertTrue(any("reversed SUB" in s for s in suggestions))

    def test_reversed_cmp(self):
        suggestions = suggest_fixes(bytes([0x3B, 0xC1]), bytes([0x39, 0xC8]))
        self.assertTrue(any("reversed CMP" in s for s in suggestions))

    def test_reversed_sbb(self):
        suggestions = suggest_fixes(bytes([0x1B, 0xC0]), bytes([0x19, 0xC0]))
        self.assertTrue(any("reversed SBB" in s for s in suggestions))

    def test_xor_width(self):
        suggestions = suggest_fixes(bytes([0x32, 0xC0]), bytes([0x33, 0xC0]))
        self.assertTrue(any("prefer_xor8" in s for s in suggestions))

    def test_no_suggestion_for_matching(self):
        suggestions = suggest_fixes(bytes([0xC3]), bytes([0xC3]))
        self.assertEqual(suggestions, [])

    def test_all_pairs_symmetric(self):
        """Ensure suggestion table covers both directions for reversals."""
        for (a, b), _ in OPCODE_SUGGESTIONS.items():
            self.assertIn(
                (b, a), OPCODE_SUGGESTIONS,
                f"Missing reverse pair for ({a:#x}, {b:#x})",
            )


# ============================================================
# Pattern suggestions
# ============================================================


class TestPatternSuggestions(unittest.TestCase):
    def test_or_minus_one(self):
        suggestions = suggest_fixes(
            bytes([0x83, 0xC8, 0xFF]),
            bytes([0xB8, 0xFF, 0xFF, 0xFF, 0xFF]),
        )
        self.assertTrue(any("prefer_or_minus_one" in s for s in suggestions))

    def test_pop_cleanup(self):
        suggestions = suggest_fixes(
            bytes([0x59]),
            bytes([0x83, 0xC4, 0x04]),
        )
        self.assertTrue(any("prefer_pop_cleanup" in s for s in suggestions))

    def test_suppress_fp_imm_fldz(self):
        suggestions = suggest_fixes(
            bytes([0xD9, 0x05, 0x00, 0x00, 0x00, 0x00]),
            bytes([0xD9, 0xEE]),
        )
        self.assertTrue(any("suppress_fp_imm" in s for s in suggestions))

    def test_expand_movzx(self):
        suggestions = suggest_fixes(
            bytes([0x33, 0xC0]),
            bytes([0x0F, 0xB6, 0x41, 0x08]),
        )
        self.assertTrue(any("expand_movzx" in s for s in suggestions))

    def test_sete_ecx(self):
        suggestions = suggest_fixes(
            bytes([0x0F, 0x94, 0xC1]),
            bytes([0x0F, 0x94, 0xC0]),
        )
        self.assertTrue(any("prefer_sete_ecx" in s for s in suggestions))

    def test_neg_sbb(self):
        suggestions = suggest_fixes(
            bytes([0xF7, 0xD8]),
            bytes([0x33, 0xC0]),
        )
        self.assertTrue(any("prefer_neg_sbb" in s for s in suggestions))

    def test_no_test_sete_fold(self):
        suggestions = suggest_fixes(
            bytes([0xF6, 0xD0]),
            bytes([0xA8, 0x01]),
        )
        self.assertTrue(any("no_test_sete_fold" in s for s in suggestions))

    def test_no_bool_mask_and_al(self):
        suggestions = suggest_fixes(
            bytes([0xC3]),
            bytes([0x24, 0x01]),
        )
        self.assertTrue(any("no_bool_mask" in s for s in suggestions))

    def test_no_bool_mask_and_eax(self):
        suggestions = suggest_fixes(
            bytes([0xC3]),
            bytes([0x83, 0xE0, 0x01]),
        )
        self.assertTrue(any("no_bool_mask" in s for s in suggestions))

    def test_fmul_mem(self):
        suggestions = suggest_fixes(
            bytes([0xD8, 0x0D, 0x00, 0x00, 0x00, 0x00]),
            bytes([0xDE, 0xC9]),
        )
        self.assertTrue(any("prefer_fmul_mem" in s for s in suggestions))

    def test_inc_dec_byte(self):
        suggestions = suggest_fixes(
            bytes([0xFE, 0x81, 0xB6, 0x00, 0x00, 0x00]),
            bytes([0x0F, 0xB6, 0x81, 0xB6, 0x00, 0x00]),
        )
        self.assertTrue(any("prefer_inc_dec_byte" in s for s in suggestions))


# ============================================================
# Function-level suggestions
# ============================================================


class TestFunctionSuggestions(unittest.TestCase):
    def test_no_callee_saves(self):
        expected = [_instr("8bc1"), _instr("c3")]
        compiled = [_instr("56"), _instr("8bc1"), _instr("5e"), _instr("c3")]
        suggestions = suggest_function_fixes(expected, compiled)
        self.assertTrue(any("no_callee_saves" in s for s in suggestions))

    def test_forced_callee_saves(self):
        expected = [_instr("56"), _instr("57"), _instr("c3")]  # esi, edi
        compiled = [_instr("53"), _instr("56"), _instr("c3")]  # ebx, esi
        suggestions = suggest_function_fixes(expected, compiled)
        self.assertTrue(any("forced_callee_saves" in s for s in suggestions))

    def test_no_ret(self):
        expected = [_instr("8bc1"), _instr("e9aabbccdd")]  # ends with jmp
        compiled = [_instr("8bc1"), _instr("c3")]  # ends with ret
        suggestions = suggest_function_fixes(expected, compiled)
        self.assertTrue(any("no_ret" in s for s in suggestions))

    def test_ret_cleanup_override(self):
        expected = [_instr("8bc1"), _instr("c20400")]  # ret 4
        compiled = [_instr("8bc1"), _instr("c20800")]  # ret 8
        suggestions = suggest_function_fixes(expected, compiled)
        self.assertTrue(any("ret_cleanup_override" in s for s in suggestions))

    def test_ret_vs_ret_n(self):
        expected = [_instr("8bc1"), _instr("c20400")]  # ret 4
        compiled = [_instr("8bc1"), _instr("c3")]  # bare ret
        suggestions = suggest_function_fixes(expected, compiled)
        self.assertTrue(any("ret_cleanup_override" in s for s in suggestions))

    def test_movzx_count(self):
        expected = [_instr("33c0"), _instr("8a4108"), _instr("c3")]
        compiled = [_instr("0fb64108"), _instr("c3")]
        suggestions = suggest_function_fixes(expected, compiled)
        self.assertTrue(any("expand_movzx" in s for s in suggestions))

    def test_no_suggestions_when_matching(self):
        instrs = [_instr("8bc1"), _instr("c3")]
        suggestions = suggest_function_fixes(instrs, instrs)
        self.assertEqual(suggestions, [])


# ============================================================
# Byte comparison and masking
# ============================================================


class TestBytesMatch(unittest.TestCase):
    def test_identical(self):
        self.assertTrue(bytes_match(b"\x33\xc0", b"\x33\xc0", frozenset()))

    def test_different(self):
        self.assertFalse(bytes_match(b"\x33\xc0", b"\x31\xc0", frozenset()))

    def test_different_length(self):
        self.assertFalse(bytes_match(b"\xc3", b"\xc2\x04\x00", frozenset()))

    def test_relocation_masked(self):
        exp = bytes([0xE8, 0x5B, 0x4E, 0xFA, 0xFF])
        got = bytes([0xE8, 0x00, 0x00, 0x00, 0x00])
        relocs = frozenset({1, 2, 3, 4})
        self.assertTrue(bytes_match(exp, got, relocs))

    def test_relocation_opcode_differs(self):
        exp = bytes([0xE9, 0x5B, 0x4E, 0xFA, 0xFF])  # jmp
        got = bytes([0xE8, 0x00, 0x00, 0x00, 0x00])  # call
        relocs = frozenset({1, 2, 3, 4})
        self.assertFalse(bytes_match(exp, got, relocs))


class TestMaskedHex(unittest.TestCase):
    def test_no_relocs(self):
        self.assertEqual(masked_hex(b"\x33\xc0", frozenset()), "33 c0")

    def test_with_relocs(self):
        result = masked_hex(b"\xe8\x00\x00\x00\x00", frozenset({1, 2, 3, 4}))
        self.assertEqual(result, "e8 xx xx xx xx")

    def test_empty(self):
        self.assertEqual(masked_hex(b"", frozenset()), "")


# ============================================================
# Needleman-Wunsch alignment
# ============================================================


class TestAlignment(unittest.TestCase):
    def test_identical(self):
        instrs = [_instr("33c0"), _instr("c3")]
        alignment = align_instructions(instrs, instrs)
        self.assertEqual(len(alignment), 2)
        for exp, comp in alignment:
            self.assertIsNotNone(exp)
            self.assertIsNotNone(comp)

    def test_extra_in_compiled(self):
        expected = [_instr("33c0"), _instr("c3")]
        compiled = [_instr("33c0"), _instr("2401"), _instr("c3")]
        alignment = align_instructions(expected, compiled)
        # Should have a gap for the extra instruction
        gaps = [(e, c) for e, c in alignment if e is None]
        self.assertGreaterEqual(len(gaps), 1)

    def test_extra_in_expected(self):
        expected = [_instr("33c0"), _instr("8a4108"), _instr("c3")]
        compiled = [_instr("0fb64108"), _instr("c3")]
        alignment = align_instructions(expected, compiled)
        # Should show gaps for the xor+mov vs movzx difference
        self.assertGreater(len(alignment), 2)

    def test_empty_expected(self):
        compiled = [_instr("c3")]
        alignment = align_instructions([], compiled)
        self.assertEqual(len(alignment), 1)
        self.assertIsNone(alignment[0][0])
        self.assertIsNotNone(alignment[0][1])

    def test_empty_compiled(self):
        expected = [_instr("c3")]
        alignment = align_instructions(expected, [])
        self.assertEqual(len(alignment), 1)
        self.assertIsNotNone(alignment[0][0])
        self.assertIsNone(alignment[0][1])

    def test_both_empty(self):
        alignment = align_instructions([], [])
        self.assertEqual(len(alignment), 0)

    def test_relocation_aware_match(self):
        """Instructions differing only in relocation bytes should align as matches."""
        expected = [_instr("e85b4efaff")]
        compiled = [
            Instr(
                raw=bytes.fromhex("e800000000"),
                text="call target",
                relocs=frozenset({1, 2, 3, 4}),
            )
        ]
        alignment = align_instructions(expected, compiled)
        self.assertEqual(len(alignment), 1)
        self.assertIsNotNone(alignment[0][0])
        self.assertIsNotNone(alignment[0][1])


# ============================================================
# ASM format detection and parsing
# ============================================================


class TestAsmFormatDetection(unittest.TestCase):
    def test_bw1_format(self):
        lines = [
            "?Func@@QAEXXZ:\n",
            "    mov eax, [ecx+8]   // 0x00401234 8b4108\n",
            "    ret                 // 0x00401237 c3\n",
        ]
        self.assertEqual(detect_asm_format(lines), "bw1")

    def test_objdump_format(self):
        lines = [
            "  401234: 8b 41 08                 mov    eax,DWORD PTR [ecx+0x8]\n",
            "  401237: c3                       ret\n",
        ]
        self.assertEqual(detect_asm_format(lines), "objdump")

    def test_ida_format(self):
        lines = [
            ".text:00401234 8B 41 08                mov     eax, [ecx+8]\n",
            ".text:00401237 C3                      retn\n",
        ]
        self.assertEqual(detect_asm_format(lines), "ida")

    def test_unknown_format(self):
        lines = ["some random text\n", "more text\n"]
        self.assertIsNone(detect_asm_format(lines))


class TestAsmParsing(unittest.TestCase):
    def test_bw1_basic(self):
        lines = [
            "?Func@@QAEXXZ:\n",
            "    mov eax, [ecx+8]   // 0x00401234 8b4108\n",
            "    ret                 // 0x00401237 c3\n",
        ]
        instrs = parse_asm_instrs(lines, "bw1", start_line=0)
        self.assertEqual(len(instrs), 2)
        self.assertEqual(instrs[0].raw, bytes.fromhex("8b4108"))
        self.assertEqual(instrs[1].raw, bytes.fromhex("c3"))

    def test_bw1_stops_at_next_label(self):
        lines = [
            "?Func1@@QAEXXZ:\n",
            "    mov eax, [ecx+8]   // 0x00401234 8b4108\n",
            "    ret                 // 0x00401237 c3\n",
            "?Func2@@QAEXXZ:\n",
            "    xor eax, eax       // 0x00401238 33c0\n",
        ]
        instrs = parse_asm_instrs(lines, "bw1", start_line=0)
        self.assertEqual(len(instrs), 2)

    def test_bw1_stops_at_nop_after_ret(self):
        lines = [
            "?Func@@QAEXXZ:\n",
            "    ret                 // 0x00401234 c3\n",
            "    nop                 // 0x00401235 90\n",
            "    mov eax, ecx       // 0x00401236 8bc1\n",
        ]
        instrs = parse_asm_instrs(lines, "bw1", start_line=0)
        self.assertEqual(len(instrs), 1)  # just ret, nop trimmed

    def test_objdump_basic(self):
        lines = [
            "  401234: 8b 41 08                 mov    eax,DWORD PTR [ecx+0x8]\n",
            "  401237: c3                       ret\n",
        ]
        instrs = parse_asm_instrs(lines, "objdump", start_line=0)
        self.assertEqual(len(instrs), 2)
        self.assertEqual(instrs[0].raw, bytes.fromhex("8b4108"))

    def test_ida_basic(self):
        lines = [
            ".text:00401234 8B 41 08                mov     eax, [ecx+8]\n",
            ".text:00401237 C3                      retn\n",
        ]
        instrs = parse_asm_instrs(lines, "ida", start_line=0)
        self.assertEqual(len(instrs), 2)
        self.assertEqual(instrs[0].raw, bytes.fromhex("8b4108"))


# ============================================================
# Symbol index
# ============================================================


class TestSymbolIndex(unittest.TestCase):
    def test_build_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            asm_file = Path(tmpdir) / "test.asm"
            asm_file.write_text(
                "?Func1@@QAEXXZ:\n"
                "    mov eax, [ecx+8]   // 0x00401234 8b4108\n"
                "    ret                 // 0x00401237 c3\n"
                "?Func2@@QAEXXZ:\n"
                "    xor eax, eax       // 0x00401238 33c0\n"
                "    ret                 // 0x0040123a c3\n"
            )
            index = build_symbol_index(Path(tmpdir))
            self.assertIn("?Func1@@QAEXXZ", index)
            self.assertIn("?Func2@@QAEXXZ", index)
            self.assertEqual(index["?Func1@@QAEXXZ"][0], asm_file)
            self.assertEqual(index["?Func1@@QAEXXZ"][1], 0)
            self.assertEqual(index["?Func2@@QAEXXZ"][1], 3)

    def test_ignores_local_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            asm_file = Path(tmpdir) / "test.asm"
            asm_file.write_text(
                "?Func@@QAEXXZ:\n"
                ".Lbl_1:\n"
                "    jmp .Lbl_1          // 0x00401234 ebfe\n"
            )
            index = build_symbol_index(Path(tmpdir))
            self.assertIn("?Func@@QAEXXZ", index)
            self.assertNotIn(".Lbl_1", index)


# ============================================================
# PE utilities
# ============================================================


def _build_minimal_pe(image_base: int = 0x00400000) -> bytes:
    """Build a minimal valid PE32 header for testing."""
    # DOS header
    dos = bytearray(64)
    dos[0:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, 64)  # e_lfanew

    # PE signature
    pe_sig = b"PE\0\0"

    # COFF header (20 bytes)
    coff = bytearray(20)
    struct.pack_into("<H", coff, 0, 0x14C)  # Machine: i386
    struct.pack_into("<H", coff, 2, 1)  # NumberOfSections
    struct.pack_into("<H", coff, 16, 96)  # SizeOfOptionalHeader

    # Optional header (96 bytes for PE32)
    opt = bytearray(96)
    struct.pack_into("<H", opt, 0, 0x10B)  # Magic: PE32
    struct.pack_into("<I", opt, 28, image_base)  # ImageBase
    struct.pack_into("<I", opt, 32, 0x1000)  # SectionAlignment
    struct.pack_into("<I", opt, 36, 0x200)  # FileAlignment

    # Section header (40 bytes)
    section = bytearray(40)
    section[0:6] = b".text\0"
    struct.pack_into("<I", section, 8, 0x1000)  # VirtualSize
    struct.pack_into("<I", section, 12, 0x1000)  # VirtualAddress (RVA)
    struct.pack_into("<I", section, 16, 0x200)  # SizeOfRawData
    struct.pack_into("<I", section, 20, 0x200)  # PointerToRawData

    # Pad to PointerToRawData and add some code bytes
    header = bytes(dos) + pe_sig + bytes(coff) + bytes(opt) + bytes(section)
    padding = b"\0" * (0x200 - len(header))
    code = b"\x33\xc0\x8a\x41\x08\xc3"  # xor eax,eax; mov al,[ecx+8]; ret
    code_padding = b"\xcc" * (0x200 - len(code))

    return header + padding + code + code_padding


class TestPEUtilities(unittest.TestCase):
    def test_read_image_base(self):
        pe_data = _build_minimal_pe(0x00400000)
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f:
            f.write(pe_data)
            f.flush()
            result = pe_read_image_base(Path(f.name))
            self.assertEqual(result, 0x00400000)

    def test_read_image_base_custom(self):
        pe_data = _build_minimal_pe(0x10000000)
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f:
            f.write(pe_data)
            f.flush()
            result = pe_read_image_base(Path(f.name))
            self.assertEqual(result, 0x10000000)

    def test_rva_to_offset(self):
        pe_data = _build_minimal_pe()
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f:
            f.write(pe_data)
            f.flush()
            # RVA 0x1000 -> file offset 0x200 (based on our section header)
            result = pe_rva_to_offset(Path(f.name), 0x1000)
            self.assertEqual(result, 0x200)

    def test_rva_to_offset_with_displacement(self):
        pe_data = _build_minimal_pe()
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f:
            f.write(pe_data)
            f.flush()
            # RVA 0x1004 -> file offset 0x204
            result = pe_rva_to_offset(Path(f.name), 0x1004)
            self.assertEqual(result, 0x204)

    def test_rva_to_offset_not_found(self):
        pe_data = _build_minimal_pe()
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f:
            f.write(pe_data)
            f.flush()
            result = pe_rva_to_offset(Path(f.name), 0x5000)
            self.assertIsNone(result)

    def test_invalid_pe(self):
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f:
            f.write(b"not a PE file")
            f.flush()
            with self.assertRaises(ValueError):
                pe_read_image_base(Path(f.name))


# ============================================================
# objdump output parsing
# ============================================================


class TestObjdumpParsing(unittest.TestCase):
    def test_basic_function(self):
        output = (
            "file.o:\tfile format coff-i386\n"
            "\n"
            "Disassembly of section .text:\n"
            "\n"
            "00000000 <_MyFunc>:\n"
            "       0: 33 c0                \txorl\t%eax, %eax\n"
            "       2: 8a 41 08             \tmovb\t0x8(%ecx), %al\n"
            "       5: c3                   \tretl\n"
        )
        instrs = _parse_objdump_output(output, "_MyFunc")
        self.assertEqual(len(instrs), 3)
        self.assertEqual(instrs[0].raw, bytes.fromhex("33c0"))
        self.assertEqual(instrs[1].raw, bytes.fromhex("8a4108"))
        self.assertEqual(instrs[2].raw, bytes.fromhex("c3"))

    def test_stops_at_next_function(self):
        output = (
            "00000000 <_Func1>:\n"
            "       0: 33 c0                \txorl\t%eax, %eax\n"
            "       2: c3                   \tretl\n"
            "\n"
            "00000003 <_Func2>:\n"
            "       3: 8b c1               \tmovl\t%ecx, %eax\n"
            "       5: c3                   \tretl\n"
        )
        instrs = _parse_objdump_output(output, "_Func1")
        self.assertEqual(len(instrs), 2)

    def test_with_relocation(self):
        output = (
            "00000000 <_MyFunc>:\n"
            "       0: e8 00 00 00 00       \tcalll\t0x5\n"
            "                00000001: IMAGE_REL_I386_REL32\t_target\n"
            "       5: c3                   \tretl\n"
        )
        instrs = _parse_objdump_output(output, "_MyFunc")
        self.assertEqual(len(instrs), 2)
        # The call instruction should have relocations at bytes 1-4
        self.assertIn(1, instrs[0].relocs)
        self.assertIn(4, instrs[0].relocs)

    def test_no_symbol_filter(self):
        output = (
            "00000000 <_Func1>:\n"
            "       0: 33 c0                \txorl\t%eax, %eax\n"
            "       2: c3                   \tretl\n"
            "\n"
            "00000003 <_Func2>:\n"
            "       3: 8b c1               \tmovl\t%ecx, %eax\n"
            "       5: c3                   \tretl\n"
        )
        instrs = _parse_objdump_output(output, symbol=None, stop_at_next_label=False)
        self.assertEqual(len(instrs), 4)  # all instructions from both functions

    def test_symbol_not_found(self):
        output = (
            "00000000 <_Func1>:\n"
            "       0: c3                   \tretl\n"
        )
        instrs = _parse_objdump_output(output, "_NonExistent")
        self.assertEqual(len(instrs), 0)


# ============================================================
# Output formatting
# ============================================================


class TestFormatting(unittest.TestCase):
    def test_batch_line_match(self):
        line = format_batch_line("_MyFunc", "MATCH", 10, 10)
        self.assertIn("MATCH", line)
        self.assertIn("_MyFunc", line)

    def test_batch_line_mismatch(self):
        line = format_batch_line("_MyFunc", "MISMATCH", 12, 10)
        self.assertIn("MISMATCH", line)
        self.assertIn("10", line)
        self.assertIn("12", line)

    def test_batch_line_long_symbol(self):
        symbol = "?" + "A" * 100 + "@@QAEXXZ"
        line = format_batch_line(symbol, "MATCH", 10, 10)
        self.assertIn("...", line)

    def test_format_diff_matching(self):
        instrs = [_instr("33c0"), _instr("c3")]
        alignment = align_instructions(instrs, instrs)
        output = format_diff("_test", alignment, instrs, instrs)
        self.assertIn("No suggestions", output)

    def test_format_diff_mismatching(self):
        expected = [_instr("33c0"), _instr("c3")]
        compiled = [_instr("0fb641ff"), _instr("c3")]
        alignment = align_instructions(expected, compiled)
        output = format_diff("_test", alignment, expected, compiled)
        self.assertIn(">>", output)
        self.assertIn("Suggested attributes", output)


# ============================================================
# Integration: collect_all_suggestions
# ============================================================


class TestCollectAllSuggestions(unittest.TestCase):
    def test_deduplicates(self):
        expected = [_instr("33c0"), _instr("8a4108"), _instr("c3")]
        compiled = [_instr("0fb64108"), _instr("c3")]
        alignment = align_instructions(expected, compiled)
        suggestions = collect_all_suggestions(alignment, expected, compiled)
        # Should mention expand_movzx but not duplicate it
        movzx_count = sum(1 for s in suggestions if "expand_movzx" in s)
        self.assertGreaterEqual(movzx_count, 1)

    def test_empty_when_matching(self):
        instrs = [_instr("8bc1"), _instr("c3")]
        alignment = align_instructions(instrs, instrs)
        suggestions = collect_all_suggestions(alignment, instrs, instrs)
        self.assertEqual(suggestions, [])


if __name__ == "__main__":
    unittest.main()

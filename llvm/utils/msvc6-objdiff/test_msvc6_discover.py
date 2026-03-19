#!/usr/bin/env python3
"""Tests for msvc6_discover.py.

Self-contained — no external files or LLVM tools required.

Run:
    python3 -m pytest test_msvc6_discover.py -v
    python3 -m unittest test_msvc6_discover -v
"""

from __future__ import annotations

import csv
import io
import struct
import tempfile
import unittest
from pathlib import Path

from msvc6_objdiff import Instr

from msvc6_discover import (
    ALL_ATTRS,
    O1_INDICATORS,
    O2_INDICATORS,
    DiscoveryResult,
    FunctionEntry,
    analyze_function,
    classify_opt_level,
    detect_callee_saves,
    detect_expand_movzx,
    detect_fpu_comparison,
    detect_no_test_sete_fold,
    detect_or_minus_one,
    detect_pop_cleanup,
    detect_prefer_fmul_mem,
    detect_prefer_inc_dec_byte,
    detect_prefer_neg_sbb,
    detect_prefer_sete_ecx,
    detect_prefer_xor8,
    detect_ret_cleanup,
    detect_reversed_encoding,
    detect_suppress_fp_imm,
    detect_trailing_bytes,
    format_attr_c,
    format_c_header,
    format_csv_output,
    format_report,
    parse_function_list,
    scan_for_functions,
)


def _instr(hex_str: str, text: str = "") -> Instr:
    return Instr(raw=bytes.fromhex(hex_str), text=text, relocs=frozenset())


# ============================================================
# Prologue / callee saves detection
# ============================================================


class TestDetectCalleeSaves(unittest.TestCase):
    def test_no_pushes(self):
        instrs = [_instr("8bc1"), _instr("c3")]
        name, val, note = detect_callee_saves(instrs)
        self.assertEqual(name, "no_callee_saves")
        self.assertTrue(val)

    def test_esi_push(self):
        instrs = [_instr("56"), _instr("8bc1"), _instr("5e"), _instr("c3")]
        name, val, note = detect_callee_saves(instrs)
        self.assertEqual(name, "forced_callee_saves")
        self.assertEqual(val, "esi")

    def test_esi_edi_push(self):
        instrs = [_instr("56"), _instr("57"), _instr("8bc1"), _instr("c3")]
        name, val, note = detect_callee_saves(instrs)
        self.assertEqual(name, "forced_callee_saves")
        self.assertEqual(val, "esi,edi")

    def test_frame_pointer_prologue(self):
        instrs = [_instr("55"), _instr("8bec"), _instr("c3")]
        name, val, note = detect_callee_saves(instrs)
        self.assertIsNone(name)  # frame pointer, not callee saves
        self.assertIn("frame pointer", note)

    def test_ebx_esi_edi_push(self):
        instrs = [_instr("53"), _instr("56"), _instr("57"), _instr("c3")]
        name, val, note = detect_callee_saves(instrs)
        self.assertEqual(name, "forced_callee_saves")
        self.assertEqual(val, "ebx,esi,edi")


# ============================================================
# Epilogue / ret cleanup detection
# ============================================================


class TestDetectRetCleanup(unittest.TestCase):
    def test_bare_ret(self):
        instrs = [_instr("8bc1"), _instr("c3")]
        name, val, note = detect_ret_cleanup(instrs, b"\x8b\xc1\xc3")
        self.assertIsNone(name)

    def test_ret_4(self):
        instrs = [_instr("8bc1"), _instr("c20400")]
        name, val, note = detect_ret_cleanup(instrs, b"\x8b\xc1\xc2\x04\x00")
        self.assertEqual(name, "ret_cleanup_override")
        self.assertEqual(val, "4")

    def test_ret_8(self):
        instrs = [_instr("8bc1"), _instr("c20800")]
        name, val, note = detect_ret_cleanup(instrs, b"\x8b\xc1\xc2\x08\x00")
        self.assertEqual(name, "ret_cleanup_override")
        self.assertEqual(val, "8")

    def test_jmp_ending(self):
        instrs = [_instr("8bc1"), _instr("e9aabbccdd")]
        name, val, note = detect_ret_cleanup(instrs, b"\x8b\xc1\xe9\xaa\xbb\xcc\xdd")
        self.assertEqual(name, "no_ret")
        self.assertTrue(val)

    def test_indirect_jmp(self):
        instrs = [_instr("ff25aabbccdd")]
        raw = b"\xff\x25\xaa\xbb\xcc\xdd"
        name, val, note = detect_ret_cleanup(instrs, raw)
        self.assertEqual(name, "no_ret")

    def test_raw_bytes_fallback_ret(self):
        name, val, note = detect_ret_cleanup([], b"\x8b\xc1\xc3")
        self.assertIsNone(name)

    def test_raw_bytes_fallback_ret_n(self):
        name, val, note = detect_ret_cleanup([], b"\x8b\xc1\xc2\x04\x00")
        self.assertEqual(name, "ret_cleanup_override")
        self.assertEqual(val, "4")


# ============================================================
# expand_movzx detection
# ============================================================


class TestDetectExpandMovzx(unittest.TestCase):
    def test_xor_eax_mov_al(self):
        # xor eax,eax (33 C0) + mov al,[ecx+8] (8A 41 08)
        raw = bytes.fromhex("33c0 8a4108 c3")
        count, note = detect_expand_movzx(raw)
        self.assertEqual(count, 1)

    def test_reversed_xor_mov(self):
        # xor eax,eax reversed (31 C0) + mov al,[ecx+8]
        raw = bytes.fromhex("31c0 8a4108 c3")
        count, note = detect_expand_movzx(raw)
        self.assertEqual(count, 1)

    def test_no_false_positive(self):
        # xor eax,eax followed by non-byte-mov
        raw = bytes.fromhex("33c0 8bc1 c3")
        count, note = detect_expand_movzx(raw)
        self.assertEqual(count, 0)

    def test_multiple_patterns(self):
        # Two xor+mov patterns
        raw = bytes.fromhex("33c0 8a4108 33c9 8a5110 c3")
        count, note = detect_expand_movzx(raw)
        self.assertEqual(count, 2)


# ============================================================
# or -1 detection
# ============================================================


class TestDetectOrMinusOne(unittest.TestCase):
    def test_or_eax_minus_one(self):
        raw = bytes.fromhex("83c8ff c3")
        count, note = detect_or_minus_one(raw)
        self.assertEqual(count, 1)

    def test_or_ecx_minus_one(self):
        raw = bytes.fromhex("83c9ff c3")
        count, note = detect_or_minus_one(raw)
        self.assertEqual(count, 1)

    def test_no_match(self):
        raw = bytes.fromhex("b8ffffffff c3")
        count, note = detect_or_minus_one(raw)
        self.assertEqual(count, 0)


# ============================================================
# pop cleanup detection
# ============================================================


class TestDetectPopCleanup(unittest.TestCase):
    def test_pop_ecx_in_body(self):
        instrs = [_instr("8bc1"), _instr("e800000000"), _instr("59"), _instr("c3")]
        count, note = detect_pop_cleanup(instrs, b"")
        self.assertEqual(count, 1)

    def test_pop_ecx_not_in_prologue(self):
        # push esi at start, pop ecx later = cleanup
        instrs = [_instr("56"), _instr("e800000000"), _instr("59"), _instr("c3")]
        count, note = detect_pop_cleanup(instrs, b"")
        self.assertEqual(count, 1)

    def test_no_pop(self):
        instrs = [_instr("8bc1"), _instr("c3")]
        count, note = detect_pop_cleanup(instrs, b"")
        self.assertEqual(count, 0)


# ============================================================
# suppress_fp_imm detection
# ============================================================


class TestDetectSuppressFpImm(unittest.TestCase):
    def test_fld_mem_no_fldz(self):
        # fld dword ptr [addr] = D9 05 XX XX XX XX
        raw = bytes.fromhex("d905 aabbccdd c3")
        found, note = detect_suppress_fp_imm(raw)
        self.assertTrue(found)

    def test_has_fldz_no_suggestion(self):
        # fldz (D9 EE) present — no suggestion
        raw = bytes.fromhex("d9ee c3")
        found, note = detect_suppress_fp_imm(raw)
        self.assertFalse(found)


# ============================================================
# no_test_sete_fold detection
# ============================================================


class TestDetectNoTestSeteFold(unittest.TestCase):
    def test_not_shr_pattern(self):
        # not al (F6 D0) + shr eax, 3 (C1 E8 03)
        raw = bytes.fromhex("f6d0 c1e803 83e001 c3")
        count, note = detect_no_test_sete_fold(raw)
        self.assertEqual(count, 1)

    def test_no_match(self):
        raw = bytes.fromhex("33c0 c3")
        count, note = detect_no_test_sete_fold(raw)
        self.assertEqual(count, 0)


# ============================================================
# neg_sbb detection
# ============================================================


class TestDetectPreferNegSbb(unittest.TestCase):
    def test_neg_sbb_inc(self):
        # neg eax (F7 D8) + sbb eax,eax (1B C0) + inc eax (40)
        raw = bytes.fromhex("f7d8 1bc0 40 c3")
        count, note = detect_prefer_neg_sbb(raw)
        self.assertEqual(count, 1)

    def test_neg_ecx(self):
        # neg ecx (F7 D9) + sbb ecx,ecx (1B C9)
        raw = bytes.fromhex("f7d9 1bc9 41 c3")
        count, note = detect_prefer_neg_sbb(raw)
        self.assertEqual(count, 1)

    def test_only_neg_no_sbb(self):
        raw = bytes.fromhex("f7d8 33c0 c3")
        count, note = detect_prefer_neg_sbb(raw)
        self.assertEqual(count, 0)


# ============================================================
# sete ecx detection
# ============================================================


class TestDetectPreferSeteEcx(unittest.TestCase):
    def test_sete_cl(self):
        raw = bytes.fromhex("0f94c1 c3")
        count, note = detect_prefer_sete_ecx(raw)
        self.assertEqual(count, 1)

    def test_sete_al_no_match(self):
        raw = bytes.fromhex("0f94c0 c3")
        count, note = detect_prefer_sete_ecx(raw)
        self.assertEqual(count, 0)


# ============================================================
# xor8 detection
# ============================================================


class TestDetectPreferXor8(unittest.TestCase):
    def test_xor_al_al(self):
        raw = bytes.fromhex("32c0 c3")
        count, note = detect_prefer_xor8(raw)
        self.assertEqual(count, 1)

    def test_xor_eax_eax_no_match(self):
        raw = bytes.fromhex("33c0 c3")
        count, note = detect_prefer_xor8(raw)
        self.assertEqual(count, 0)


# ============================================================
# inc/dec byte detection
# ============================================================


class TestDetectPreferIncDecByte(unittest.TestCase):
    def test_inc_byte_mem(self):
        # inc byte ptr [ecx+0xb6] = FE 81 B6 00 00 00
        raw = bytes.fromhex("fe81 b6000000 c3")
        count, note = detect_prefer_inc_dec_byte(raw)
        self.assertEqual(count, 1)

    def test_inc_reg_no_match(self):
        # FE C0 = inc al (register, mod=3, should NOT match)
        raw = bytes.fromhex("fec0 c3")
        count, note = detect_prefer_inc_dec_byte(raw)
        self.assertEqual(count, 0)


# ============================================================
# fmul mem detection
# ============================================================


class TestDetectPreferFmulMem(unittest.TestCase):
    def test_fmul_mem(self):
        # fmul dword ptr [addr] = D8 0D XX XX XX XX
        raw = bytes.fromhex("d80d aabbccdd c3")
        count, note = detect_prefer_fmul_mem(raw)
        self.assertEqual(count, 1)

    def test_fmulp_no_match(self):
        # fmulp = DE C9 (not a memory-form fmul)
        raw = bytes.fromhex("dec9 c3")
        count, note = detect_prefer_fmul_mem(raw)
        self.assertEqual(count, 0)


# ============================================================
# Reversed encoding detection
# ============================================================


class TestDetectReversedEncoding(unittest.TestCase):
    def test_all_reversed(self):
        # mov eax, ecx (8B C1, MSVC direction) x3
        instrs = [_instr("8bc1"), _instr("8bc2"), _instr("8bc3"), _instr("c3")]
        rev, total, note = detect_reversed_encoding(instrs)
        self.assertEqual(rev, 3)
        self.assertEqual(total, 3)

    def test_all_modern(self):
        # mov ecx, eax (89 C1, modern direction) x3
        instrs = [_instr("89c1"), _instr("89c2"), _instr("89c3"), _instr("c3")]
        rev, total, note = detect_reversed_encoding(instrs)
        self.assertEqual(rev, 0)
        self.assertEqual(total, 3)

    def test_mixed(self):
        instrs = [_instr("8bc1"), _instr("89c2"), _instr("c3")]
        rev, total, note = detect_reversed_encoding(instrs)
        self.assertEqual(rev, 1)
        self.assertEqual(total, 2)

    def test_no_reg_reg_ops(self):
        instrs = [_instr("8b4108"), _instr("c3")]  # mov eax, [ecx+8] (memory, not reg-reg)
        rev, total, note = detect_reversed_encoding(instrs)
        self.assertEqual(total, 0)

    def test_xor_reversed(self):
        # xor eax,ecx using 33 (MSVC direction)
        instrs = [_instr("33c1"), _instr("c3")]
        rev, total, note = detect_reversed_encoding(instrs)
        self.assertEqual(rev, 1)


# ============================================================
# Trailing bytes detection
# ============================================================


class TestDetectTrailingBytes(unittest.TestCase):
    def test_trailing_after_ret(self):
        # ret + 2 bytes of dead code (not nop/int3) + CC padding
        raw = bytes.fromhex("c3 33c0 cccc")
        has_tb, has_ta, note = detect_trailing_bytes(raw, [])
        self.assertTrue(has_tb)
        self.assertFalse(has_ta)

    def test_no_trailing(self):
        # ret + only CC padding
        raw = bytes.fromhex("c3 cccc")
        has_tb, has_ta, note = detect_trailing_bytes(raw, [])
        self.assertFalse(has_tb)
        self.assertFalse(has_ta)

    def test_trailing_with_call(self):
        # ret + call instruction (E8) + padding
        raw = bytes.fromhex("c3 e8aabbccdd cccc")
        has_tb, has_ta, note = detect_trailing_bytes(raw, [])
        self.assertFalse(has_tb)
        self.assertTrue(has_ta)
        self.assertIn("trailing_asm", note)

    def test_no_ret_no_trailing(self):
        raw = bytes.fromhex("8bc1 e9aabbccdd")
        has_tb, has_ta, note = detect_trailing_bytes(raw, [])
        self.assertFalse(has_tb)
        self.assertFalse(has_ta)


# ============================================================
# FPU comparison detection
# ============================================================


class TestDetectFpuComparison(unittest.TestCase):
    def test_fnstsw_test_ah(self):
        # fnstsw ax (DF E0) + test ah, 0x41 (F6 C4 41)
        raw = bytes.fromhex("dfe0 f6c441 c3")
        count, note = detect_fpu_comparison(raw)
        self.assertEqual(count, 1)

    def test_fnstsw_without_test(self):
        raw = bytes.fromhex("dfe0 9e c3")  # fnstsw ax + sahf (modern, not MSVC)
        count, note = detect_fpu_comparison(raw)
        self.assertEqual(count, 0)


# ============================================================
# Opt-level classification
# ============================================================


class TestOptLevelClassifier(unittest.TestCase):
    def test_o1(self):
        attrs = {"prefer_or_minus_one": True, "prefer_pop_cleanup": True}
        self.assertEqual(classify_opt_level(attrs), "O1")

    def test_o2(self):
        attrs = {"expand_movzx": True, "prefer_xor8": True}
        self.assertEqual(classify_opt_level(attrs), "O2")

    def test_mixed(self):
        attrs = {"prefer_or_minus_one": True, "expand_movzx": True}
        self.assertEqual(classify_opt_level(attrs), "mixed")

    def test_unknown(self):
        attrs = {"no_callee_saves": True}  # neutral attribute
        self.assertEqual(classify_opt_level(attrs), "unknown")

    def test_empty(self):
        self.assertEqual(classify_opt_level({}), "unknown")


# ============================================================
# Function list parsing
# ============================================================


class TestParseFunctionList(unittest.TestCase):
    def test_csv_with_header(self):
        content = "name,rva,size\n_func1,0x1000,48\n_func2,0x1030,32\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(content)
            f.flush()
            entries = parse_function_list(Path(f.name))
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].name, "_func1")
        self.assertEqual(entries[0].rva, 0x1000)
        self.assertEqual(entries[0].size, 48)

    def test_csv_address_with_image_base(self):
        content = "name,address,size\n_func1,0x401000,48\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(content)
            f.flush()
            entries = parse_function_list(Path(f.name), image_base=0x400000)
        self.assertEqual(entries[0].rva, 0x1000)

    def test_csv_start_end(self):
        content = "name,rva,end\n_func1,0x1000,0x1030\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(content)
            f.flush()
            entries = parse_function_list(Path(f.name))
        self.assertEqual(entries[0].size, 0x30)

    def test_empty_csv(self):
        content = "name,rva,size\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(content)
            f.flush()
            entries = parse_function_list(Path(f.name))
        self.assertEqual(len(entries), 0)


# ============================================================
# Prologue scanner
# ============================================================


class TestPrologueScanner(unittest.TestCase):
    def test_push_ebp_prologue(self):
        # CC padding + push ebp; mov ebp, esp; ret + CC padding
        data = b"\xcc\xcc" + b"\x55\x8b\xec\xc3" + b"\xcc\xcc"
        entries = scan_for_functions(data, 0x1000)
        self.assertGreaterEqual(len(entries), 1)
        self.assertEqual(entries[0].rva, 0x1002)

    def test_push_esi_prologue(self):
        # CC padding + push esi; mov eax,ecx; ret + CC padding
        data = b"\xcc\xcc" + b"\x56\x8b\xc1\xc3" + b"\xcc\xcc"
        entries = scan_for_functions(data, 0x1000)
        self.assertGreaterEqual(len(entries), 1)

    def test_multiple_functions(self):
        func1 = b"\x55\x8b\xec\xc3"
        func2 = b"\x56\x8b\xc1\xc3"
        data = b"\xcc\xcc" + func1 + b"\xcc\xcc" + func2 + b"\xcc\xcc"
        entries = scan_for_functions(data, 0x1000)
        self.assertGreaterEqual(len(entries), 2)


# ============================================================
# Output formatters
# ============================================================


class TestFormatAttrC(unittest.TestCase):
    def test_boolean_attrs(self):
        attrs = {"no_callee_saves": True, "expand_movzx": True}
        result = format_attr_c(attrs)
        self.assertIn("no_callee_saves", result)
        self.assertIn("expand_movzx", result)
        self.assertTrue(result.startswith("__attribute__(("))

    def test_parameterized_attrs(self):
        attrs = {"forced_callee_saves": "esi,edi", "ret_cleanup_override": "4"}
        result = format_attr_c(attrs)
        self.assertIn('forced_callee_saves("esi,edi")', result)
        self.assertIn('ret_cleanup_override("4")', result)

    def test_empty(self):
        self.assertEqual(format_attr_c({}), "")


class TestFormatReport(unittest.TestCase):
    def test_basic_report(self):
        results = [
            DiscoveryResult(
                name="_func1", rva=0x1000, size=48,
                attributes={"no_callee_saves": True, "expand_movzx": True},
                opt_level="O2", notes=["1x xor+mov byte expansion"],
            ),
        ]
        output = format_report(results, "test.exe")
        self.assertIn("_func1", output)
        self.assertIn("O2", output)
        self.assertIn("no_callee_saves", output)
        self.assertIn("Summary", output)

    def test_summary_stats(self):
        results = [
            DiscoveryResult("f1", 0x1000, 10, {"expand_movzx": True}, "O2", []),
            DiscoveryResult("f2", 0x1100, 10, {"prefer_or_minus_one": True}, "O1", []),
        ]
        output = format_report(results)
        self.assertIn("O1: 1", output)
        self.assertIn("O2: 1", output)


class TestFormatCSV(unittest.TestCase):
    def test_header_row(self):
        results = []
        output = format_csv_output(results)
        reader = csv.reader(io.StringIO(output))
        header = next(reader)
        self.assertIn("name", header)
        self.assertIn("no_callee_saves", header)

    def test_data_row(self):
        results = [
            DiscoveryResult(
                name="_func1", rva=0x1000, size=48,
                attributes={"no_callee_saves": True, "forced_callee_saves": "esi"},
                opt_level="O2", notes=[],
            ),
        ]
        output = format_csv_output(results)
        reader = csv.reader(io.StringIO(output))
        next(reader)  # skip header
        row = next(reader)
        self.assertEqual(row[0], "_func1")
        # Find no_callee_saves column
        header = list(csv.reader(io.StringIO(output)))[0]
        ncs_idx = header.index("no_callee_saves")
        self.assertEqual(row[ncs_idx], "true")


class TestFormatCHeader(unittest.TestCase):
    def test_basic(self):
        results = [
            DiscoveryResult(
                name="_func1", rva=0x1000, size=48,
                attributes={"no_callee_saves": True},
                opt_level="O2", notes=[],
            ),
        ]
        output = format_c_header(results, "test.exe")
        self.assertIn("__attribute__((no_callee_saves))", output)
        self.assertIn("void", output)
        self.assertIn("Generated by msvc6_discover.py", output)


# ============================================================
# Integration: analyze_function
# ============================================================


class TestAnalyzeFunction(unittest.TestCase):
    def test_simple_leaf(self):
        entry = FunctionEntry("_leaf", 0x1000, 6)
        # xor eax,eax; mov al,[ecx+8]; ret
        instrs = [_instr("33c0"), _instr("8a4108"), _instr("c3")]
        raw = bytes.fromhex("33c0 8a4108 c3")
        result = analyze_function(entry, instrs, raw)
        self.assertEqual(result.name, "_leaf")
        self.assertIn("no_callee_saves", result.attributes)
        self.assertIn("expand_movzx", result.attributes)
        self.assertEqual(result.opt_level, "O2")

    def test_frame_pointer_function(self):
        entry = FunctionEntry("_framed", 0x1000, 5)
        instrs = [_instr("55"), _instr("8bec"), _instr("5d"), _instr("c3")]
        raw = bytes.fromhex("55 8bec 5d c3")
        result = analyze_function(entry, instrs, raw)
        # Should NOT have no_callee_saves (frame pointer detected)
        self.assertNotIn("no_callee_saves", result.attributes)

    def test_or_minus_one_return(self):
        entry = FunctionEntry("_ret_neg1", 0x1000, 4)
        instrs = [_instr("83c8ff"), _instr("c3")]
        raw = bytes.fromhex("83c8ff c3")
        result = analyze_function(entry, instrs, raw)
        self.assertIn("prefer_or_minus_one", result.attributes)
        self.assertEqual(result.opt_level, "O1")

    def test_empty_function(self):
        entry = FunctionEntry("_empty", 0x1000, 1)
        instrs = [_instr("c3")]
        raw = bytes.fromhex("c3")
        result = analyze_function(entry, instrs, raw)
        self.assertEqual(result.size, 1)


if __name__ == "__main__":
    unittest.main()

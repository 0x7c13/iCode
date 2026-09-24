# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for decode_subprocess_output — CJK code-page decoding on Windows.

The core bug: on Windows with a legacy console code page (e.g. GBK/cp936),
subprocess output is encoded in the system code page.  If the output is
mostly ASCII with a few CJK characters, the surrogate-ratio heuristic
(< 10 %) returned ``raw.decode("utf-8", errors="replace")`` — garbling
the CJK — instead of falling through to the full ``decode_bytes()``
detection pipeline.

These tests reproduce the bug by monkeypatching ``sys.platform`` and the
``_windows_uses_utf8`` check to simulate a Windows + legacy code page
environment, then verify that GBK/Shift-JIS/EUC-KR bytes are decoded
correctly.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from chrys.foundation.platform.process import decode_subprocess_output

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WIN32 = "win32"
_MODULE = "chrys.foundation.platform.process"


def _patch_windows_legacy_cp(console_encoding: str = "cp936"):
    """Context manager: pretend we're on Windows with a non-UTF-8 code page.

    *console_encoding* is the Python codec name for the simulated console
    code page (default ``"cp936"`` for GBK / Simplified Chinese).
    """
    return _patch_platform(_WIN32, utf8=False, console_encoding=console_encoding)


def _patch_windows_utf8_cp():
    """Context manager: pretend we're on Windows with UTF-8 (cp65001)."""
    return _patch_platform(_WIN32, utf8=True, console_encoding=None)


def _patch_unix():
    """Context manager: pretend we're on macOS/Linux."""
    return _patch_platform("darwin", utf8=True, console_encoding=None)


class _patch_platform:
    """Stack patches: ``sys.platform``, ``_windows_uses_utf8``, and
    ``_windows_console_encoding`` to simulate a target platform.
    """

    def __init__(self, platform: str, *, utf8: bool, console_encoding: str | None) -> None:
        self._p1 = patch(f"{_MODULE}.sys.platform", platform)
        self._p2 = patch(f"{_MODULE}._windows_uses_utf8", return_value=utf8)
        self._p3 = patch(f"{_MODULE}._windows_console_encoding", return_value=console_encoding)

    def __enter__(self):
        self._p1.__enter__()
        self._p2.__enter__()
        self._p3.__enter__()
        return self

    def __exit__(self, *args):
        self._p3.__exit__(*args)
        self._p2.__exit__(*args)
        self._p1.__exit__(*args)


# ---------------------------------------------------------------------------
# Encoding test data
# ---------------------------------------------------------------------------

# "测试结果: 成功" (Test result: success)
_CHINESE_TEXT = "测试结果: 成功"
_CHINESE_GBK = _CHINESE_TEXT.encode("gbk")
_CHINESE_UTF8 = _CHINESE_TEXT.encode("utf-8")

# "テスト結果" (Test result in Japanese)
_JAPANESE_TEXT = "テスト結果"
_JAPANESE_SJIS = _JAPANESE_TEXT.encode("shift_jis")

# "테스트 결과" (Test result in Korean)
_KOREAN_TEXT = "테스트 결과"
_KOREAN_EUCKR = _KOREAN_TEXT.encode("euc-kr")


def _mostly_ascii_with_gbk_chinese() -> bytes:
    """Simulate typical command output: mostly English, a few Chinese lines.

    This is the pattern that triggered the original bug — the surrogate
    ratio falls below 10 %, causing the code to return garbled UTF-8
    instead of falling through to GBK detection.
    """
    lines = [
        b"Starting process...",
        b"Loading configuration from C:\\Users\\test\\config.yaml",
        b"Connected to database successfully",
        b"Processing records: 100/100",
        b"Final result: " + "成功".encode("gbk"),
        b"Total time: 2.3s",
        b"",
    ]
    return b"\r\n".join(lines)


def _mostly_chinese_gbk() -> bytes:
    """Simulate output that is mostly Chinese (GBK-encoded)."""
    lines = [
        "开始处理数据...".encode("gbk"),
        "加载文件: data.csv".encode("gbk"),
        "处理了 100 条记录".encode("gbk"),
        "结果保存到: output.txt".encode("gbk"),
        "完成!".encode("gbk"),
        b"",
    ]
    return b"\r\n".join(lines)


def _single_line_gbk_chinese() -> bytes:
    """Single line of GBK-encoded Chinese — tests per-line progress decoding."""
    return "结果: 成功".encode("gbk")


# ---------------------------------------------------------------------------
# UTF-8 fast path (all platforms)
# ---------------------------------------------------------------------------


class TestUtf8FastPath:
    """When output is valid UTF-8, it should decode correctly everywhere."""

    def test_pure_ascii(self) -> None:
        assert decode_subprocess_output(b"hello world\n") == "hello world\n"

    def test_utf8_chinese(self) -> None:
        assert decode_subprocess_output(_CHINESE_UTF8) == _CHINESE_TEXT

    def test_utf8_mixed_cjk(self) -> None:
        text = "Hello 你好 World 世界\n"
        assert decode_subprocess_output(text.encode("utf-8")) == text

    def test_utf8_emoji(self) -> None:
        text = "Status: ✓ Done 🎉\n"
        assert decode_subprocess_output(text.encode("utf-8")) == text

    def test_empty(self) -> None:
        assert decode_subprocess_output(b"") == ""

    def test_bytearray(self) -> None:
        assert decode_subprocess_output(bytearray(b"hello")) == "hello"


# ---------------------------------------------------------------------------
# Windows + legacy code page (GBK, Shift-JIS, EUC-KR)
# ---------------------------------------------------------------------------


class TestWindowsLegacyCodePage:
    """On Windows with a non-UTF-8 console CP, non-UTF-8 bytes must go
    through the full detection pipeline (``decode_bytes``) instead of
    the surrogate heuristic.
    """

    def test_gbk_mostly_ascii_repro(self) -> None:
        """Reproduce the original bug: mostly-ASCII + a few GBK CJK chars.

        Before the fix, the surrogate ratio was < 10 %, so the function
        returned ``raw.decode("utf-8", errors="replace")`` — replacing
        the GBK bytes with U+FFFD or decoding them as wrong characters.
        """
        raw = _mostly_ascii_with_gbk_chinese()
        with _patch_windows_legacy_cp():
            result = decode_subprocess_output(raw)
        assert "成功" in result
        assert "\ufffd" not in result  # no replacement chars for the Chinese

    def test_gbk_mostly_chinese(self) -> None:
        """Mostly-Chinese GBK output should decode correctly."""
        raw = _mostly_chinese_gbk()
        with _patch_windows_legacy_cp():
            result = decode_subprocess_output(raw)
        assert "处理" in result
        assert "完成" in result
        assert "\ufffd" not in result

    def test_gbk_pure_chinese(self) -> None:
        """Pure GBK Chinese text."""
        with _patch_windows_legacy_cp():
            result = decode_subprocess_output(_CHINESE_GBK)
        assert "成功" in result

    def test_gbk_single_line(self) -> None:
        """Single line of GBK — simulates per-line progress decoding."""
        raw = _single_line_gbk_chinese()
        with _patch_windows_legacy_cp():
            result = decode_subprocess_output(raw)
        assert "成功" in result

    def test_shift_jis_japanese(self) -> None:
        """Shift-JIS output on Windows (cp932) should decode correctly."""
        with _patch_windows_legacy_cp(console_encoding="cp932"):
            result = decode_subprocess_output(_JAPANESE_SJIS)
        assert "テスト" in result or "結果" in result

    def test_euc_kr_korean(self) -> None:
        """EUC-KR output on Windows (cp949) should decode correctly."""
        with _patch_windows_legacy_cp(console_encoding="cp949"):
            result = decode_subprocess_output(_KOREAN_EUCKR)
        assert "테스트" in result or "결과" in result

    def test_utf8_still_works_on_legacy_cp(self) -> None:
        """UTF-8 output on Windows with legacy CP still takes the fast path.

        Even when the console CP is GBK, a program that explicitly outputs
        UTF-8 (e.g. ``chcp 65001`` or ``-Encoding UTF8``) should still
        decode correctly via the strict UTF-8 try.
        """
        with _patch_windows_legacy_cp():
            result = decode_subprocess_output(_CHINESE_UTF8)
        assert result == _CHINESE_TEXT

    def test_ascii_on_legacy_cp(self) -> None:
        """Pure ASCII is valid in all encodings — should always work."""
        with _patch_windows_legacy_cp():
            result = decode_subprocess_output(b"hello world\r\n")
        assert result == "hello world\r\n"

    def test_gbk_multiline_with_exit_code(self) -> None:
        """Simulates realistic shell tool output with [exit_code: 0] suffix."""
        body = _mostly_ascii_with_gbk_chinese()
        raw = body + b"\r\n[exit_code: 0]"
        with _patch_windows_legacy_cp():
            result = decode_subprocess_output(raw)
        assert "成功" in result
        assert "[exit_code: 0]" in result

    def test_utf16le_output_before_legacy_code_page(self) -> None:
        """PowerShell can emit UTF-16 text; do not decode that as cp936."""
        raw = "结果: 成功\r\n".encode("utf-16-le")
        with _patch_windows_legacy_cp():
            result = decode_subprocess_output(raw)
        assert result == "结果: 成功\r\n"

    def test_utf16_bom_output_before_legacy_code_page(self) -> None:
        raw = "结果: 成功\r\n".encode("utf-16")
        with _patch_windows_legacy_cp():
            result = decode_subprocess_output(raw)
        assert result == "结果: 成功\r\n"

    def test_nul_delimited_ascii_is_not_treated_as_utf16(self) -> None:
        """No-BOM UTF-16 probing must not erase NUL-delimited command output."""
        raw = b"file-a\x00file-b\x00"
        with _patch_windows_legacy_cp():
            result = decode_subprocess_output(raw)
        assert result == "file-a\x00file-b\x00"

    def test_nul_delimited_gbk_uses_legacy_code_page(self) -> None:
        """NUL-delimited legacy output should still decode with the console CP."""
        expected = "文件一\x00文件二\x00"
        raw = expected.encode("gbk")
        with _patch_windows_legacy_cp():
            result = decode_subprocess_output(raw)
        assert result == expected

    def test_fallback_to_detect_when_cp_cannot_decode(self) -> None:
        """When the console CP can't decode the bytes, fall back to detection."""
        # UTF-8 encoded Chinese passed through a simulated cp1252 console
        # — cp1252 can decode almost any byte but produces wrong characters.
        # This tests that the cp → decode_bytes fallback chain works.
        raw = _CHINESE_GBK
        with _patch_windows_legacy_cp(console_encoding="__nonexistent_codec__"):
            # LookupError for unknown codec → falls through to decode_bytes
            result = decode_subprocess_output(raw)
        # Should not crash; charset_normalizer handles it as a last resort
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Windows + UTF-8 code page (cp65001)
# ---------------------------------------------------------------------------


class TestWindowsUtf8CodePage:
    """On Windows with UTF-8 console CP, behaviour should match Unix."""

    def test_utf8_chinese(self) -> None:
        with _patch_windows_utf8_cp():
            result = decode_subprocess_output(_CHINESE_UTF8)
        assert result == _CHINESE_TEXT

    def test_corrupted_utf8_few_bad_bytes(self) -> None:
        """Mostly-valid UTF-8 with a few bad bytes → replace, not crash."""
        # Valid UTF-8 + 2 invalid bytes in 100 total → < 10 % surrogates
        raw = b"A" * 98 + b"\xff\xfe"
        with _patch_windows_utf8_cp():
            result = decode_subprocess_output(raw)
        assert "A" in result
        assert len(result) == 100  # bad bytes replaced 1:1


# ---------------------------------------------------------------------------
# Unix / macOS (surrogate heuristic path)
# ---------------------------------------------------------------------------


class TestUnix:
    """On Unix, the locale is always UTF-8, so the surrogate heuristic
    handles buffer-boundary byte splits.
    """

    def test_utf8_chinese(self) -> None:
        with _patch_unix():
            result = decode_subprocess_output(_CHINESE_UTF8)
        assert result == _CHINESE_TEXT

    def test_few_bad_bytes_replaced(self) -> None:
        """< 10 % surrogates → utf-8 with replace (buffer-boundary split)."""
        # "Hello" + one split multi-byte char + "World"
        raw = b"Hello \xe4 World"  # 0xE4 starts a 3-byte UTF-8 sequence
        with _patch_unix():
            result = decode_subprocess_output(raw)
        assert "Hello" in result
        assert "World" in result
        assert "\ufffd" in result  # the broken byte is replaced

    def test_many_bad_bytes_replaced(self) -> None:
        """>= 10 % surrogates → still utf-8 with replace on Unix."""
        # 50 ASCII + 50 invalid bytes → 50 % surrogates
        raw = b"A" * 50 + b"\xff" * 50
        with _patch_unix():
            result = decode_subprocess_output(raw)
        assert "A" in result
        # On Unix, even high surrogate ratio falls through to utf-8 replace
        assert "\ufffd" in result

    def test_empty(self) -> None:
        with _patch_unix():
            assert decode_subprocess_output(b"") == ""


# ---------------------------------------------------------------------------
# Regression: surrogate ratio threshold
# ---------------------------------------------------------------------------


class TestSurrogateThresholdRegression:
    """Verify the fix prevents the surrogate-ratio heuristic from
    short-circuiting on Windows with legacy code pages.
    """

    def test_old_behaviour_would_garble_mostly_ascii_gbk(self) -> None:
        """Demonstrate the bug: naive UTF-8 with replace garbles GBK Chinese.

        This test decodes the same bytes using the OLD logic
        (``raw.decode("utf-8", errors="replace")``) and verifies that the
        Chinese characters ARE garbled, confirming the test data actually
        reproduces the original bug.
        """
        raw = _mostly_ascii_with_gbk_chinese()
        # The old code path: strict UTF-8 fails, surrogate < 10% → replace
        naive = raw.decode("utf-8", errors="replace")
        # "成功" in GBK (0xB3C9 0xB9A6) is not valid UTF-8 → replaced
        assert "成功" not in naive, "Test data must NOT be valid UTF-8 for Chinese chars"

    @pytest.mark.parametrize(
        "chinese_ratio",
        [0.02, 0.05, 0.08, 0.15, 0.30, 0.50, 0.80],
        ids=lambda r: f"{int(r * 100)}pct_chinese",
    )
    def test_gbk_at_various_chinese_ratios(self, chinese_ratio: float) -> None:
        """GBK Chinese must decode correctly regardless of the CJK:ASCII ratio.

        The original bug only manifested when the CJK proportion was small
        (< ~10 %), but the fix should work at all ratios.
        """
        total_chars = 200
        n_chinese = max(1, int(total_chars * chinese_ratio))
        n_ascii = total_chars - n_chinese

        # Build mixed output: ASCII lines + Chinese lines
        ascii_part = b"A" * n_ascii
        # "中" in GBK = 0xD6D0 (2 bytes per char)
        chinese_part = ("中" * n_chinese).encode("gbk")
        raw = ascii_part + chinese_part

        with _patch_windows_legacy_cp():
            result = decode_subprocess_output(raw)

        assert "中" in result
        assert "\ufffd" not in result

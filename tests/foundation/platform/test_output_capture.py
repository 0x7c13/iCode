# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for bounded subprocess output capture and decoding of its kept parts."""

from __future__ import annotations

import asyncio
import codecs
import sys
import types
from pathlib import Path

import pytest

from chrys.foundation.platform import output_capture
from chrys.foundation.platform import process as process_mod
from chrys.foundation.platform.output_capture import (
    BoundedCapture,
    CapturedOutput,
    capture_limit_footer,
    drain_process_pipes,
)
from chrys.foundation.platform.process import decode_split_output, decode_subprocess_output, managed_subprocess
from tests.support.waiting import ENGINE_TURN_TIMEOUT

# ---------------------------------------------------------------------------
# BoundedCapture
# ---------------------------------------------------------------------------


def _capture(data: bytes, limit: int, chunk_size: int) -> CapturedOutput:
    capture = BoundedCapture(limit)
    for start in range(0, len(data), chunk_size):
        capture.feed(data[start : start + chunk_size])
    return capture.snapshot()


@pytest.mark.parametrize("chunk_size", [1, 2, 5, 100])
def test_capture_keeps_a_third_of_the_limit_as_head_and_the_rest_as_tail(chunk_size: int) -> None:
    data = b"abcdefghijklmnopqrstuvwxyz"

    captured = _capture(data, 9, chunk_size)

    assert captured == CapturedOutput(b"abc", b"uvwxyz", 26)
    assert captured.dropped == 26 - 3 - 6
    # The tail is the stream's last bytes: it starts where seen - len(tail) says.
    assert data[captured.seen - len(captured.tail) :] == captured.tail


@pytest.mark.parametrize(
    ("data", "head", "tail"), [(b"", b"", b""), (b"ab", b"ab", b""), (b"abcdefghi", b"abc", b"defghi")]
)
def test_capture_within_its_limit_keeps_everything(data: bytes, head: bytes, tail: bytes) -> None:
    captured = _capture(data, 9, 4)

    assert captured == CapturedOutput(head, tail, len(data))
    assert captured.dropped == 0


def test_capture_splits_a_limit_not_divisible_by_three() -> None:
    captured = _capture(bytes(range(20)), 10, 20)

    assert captured.head == bytes(range(3))
    assert captured.tail == bytes(range(13, 20))


def test_capture_default_limit_is_read_when_the_capture_is_made(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(output_capture, "OUTPUT_CAPTURE_LIMIT_BYTES", 6)
    capture = BoundedCapture()
    capture.feed(b"0123456789")

    assert capture.snapshot() == CapturedOutput(b"01", b"6789", 10)


def test_capture_limit_footer_counts_every_dropped_byte() -> None:
    kept = CapturedOutput(b"ab", b"", 2)
    dropped_a = CapturedOutput(b"a", b"z", 10)
    dropped_b = CapturedOutput(b"", b"", 5)

    assert capture_limit_footer(()) == ""
    assert capture_limit_footer((kept,)) == ""
    assert capture_limit_footer((kept, dropped_a, dropped_b)) == (
        "[Output capture limit reached: 13 bytes from the middle were not kept.]"
    )


# ---------------------------------------------------------------------------
# CapturedOutput.text
# ---------------------------------------------------------------------------


def test_text_without_a_gap_decodes_head_and_tail_as_one_stream() -> None:
    # A character split between head and tail is still one character.
    captured = CapturedOutput(b"caf" + "é".encode()[:1], "é".encode()[1:] + b" ok", 8)

    assert captured.text() == "café ok"


def test_text_marks_the_gap_with_the_dropped_byte_count() -> None:
    captured = CapturedOutput(b"first\n", b"last\n", 100)

    assert captured.text() == "first\n\n[... 89 bytes omitted ...]\nlast\n"


def test_text_cleans_each_kept_part_on_its_own() -> None:
    cleaned: list[str] = []

    def clean(text: str) -> str:
        cleaned.append(text)
        return text.upper()

    assert CapturedOutput(b"head", b"tail", 20).text(clean) == "HEAD\n[... 12 bytes omitted ...]\nTAIL"
    assert cleaned == ["head", "tail"]
    cleaned.clear()
    assert CapturedOutput(b"head", b"tail", 8).text(clean) == "HEADTAIL"
    assert cleaned == ["headtail"]


@pytest.mark.parametrize(
    ("head", "head_text"),
    [
        (b"line1\nmore\x1b]0;tit", "line1\nmore"),
        # An unfinished sequence on an earlier line is not the cut one.
        (b"line1\n\x1b]junk\nmore", "line1\n\x1b]junk\nmore"),
    ],
    ids=["last-line", "earlier-line"],
)
def test_text_drops_only_an_escape_sequence_the_gap_cut_short_on_the_last_line(head: bytes, head_text: str) -> None:
    captured = CapturedOutput(head, b"tail", 100)

    text = captured.text(lambda part: part)

    assert text == f"{head_text}\n[... {captured.dropped} bytes omitted ...]\ntail"


# ---------------------------------------------------------------------------
# decode_split_output
# ---------------------------------------------------------------------------


@pytest.fixture
def posix(monkeypatch: pytest.MonkeyPatch) -> None:
    shadow = types.ModuleType("sys")
    shadow.platform = "linux"
    monkeypatch.setattr(process_mod, "sys", shadow)


def _windows(monkeypatch: pytest.MonkeyPatch, code_page: str | None) -> None:
    shadow = types.ModuleType("sys")
    shadow.platform = "win32"
    monkeypatch.setattr(process_mod, "sys", shadow)
    monkeypatch.setattr(process_mod, "_windows_uses_utf8", lambda: code_page is None)
    monkeypatch.setattr(process_mod, "_windows_console_encoding", lambda: code_page)


def _utf8_char_starts(text: str) -> list[int]:
    starts = [0]
    for char in text:
        starts.append(starts[-1] + len(char.encode()))
    return starts


@pytest.mark.usefixtures("posix")
def test_utf8_parts_keep_every_whole_character_at_every_cut() -> None:
    text = "naïve 日本語 😀 end"
    data = text.encode()
    starts = _utf8_char_starts(text)

    for head_end in range(len(data) + 1):
        for tail_start in range(head_end, len(data) + 1):
            head, tail = decode_split_output(data[:head_end], data[tail_start:], tail_start)

            whole_in_head = max(i for i, start in enumerate(starts) if start <= head_end)
            first_in_tail = min(i for i, start in enumerate(starts) if start >= tail_start)
            assert head == text[:whole_in_head], (head_end, tail_start)
            assert tail == text[first_in_tail:], (head_end, tail_start)


@pytest.mark.usefixtures("posix")
def test_posix_replaces_bytes_that_are_not_utf8() -> None:
    assert decode_split_output(b"ok \xff\n", b"\x80\x80\x80\x80end", 50) == ("ok �\n", "�end")


_UTF16_TEXT = "日志 log 一\n第二行 second\n" * 4


@pytest.mark.parametrize("codec", ["utf-16-le", "utf-16-be"])
@pytest.mark.parametrize("bom", [False, True], ids=["no-bom", "bom"])
@pytest.mark.parametrize("head_end", [21, 22], ids=["odd-head", "even-head"])
@pytest.mark.parametrize("tail_start", [41, 42], ids=["odd-tail", "even-tail"])
def test_windows_utf16_parts_align_to_whole_code_units(
    monkeypatch: pytest.MonkeyPatch, codec: str, bom: bool, head_end: int, tail_start: int
) -> None:
    _windows(monkeypatch, "cp936")
    prefix = (codecs.BOM_UTF16_LE if codec == "utf-16-le" else codecs.BOM_UTF16_BE) if bom else b""
    data = prefix + _UTF16_TEXT.encode(codec)

    head, tail = decode_split_output(data[:head_end], data[tail_start:], tail_start)

    assert head == data[len(prefix) : head_end - head_end % 2].decode(codec)
    assert tail == data[tail_start + tail_start % 2 :].decode(codec)


@pytest.mark.parametrize("codec", ["utf-16-le", "utf-16-be"])
def test_windows_utf16_tail_skips_the_low_half_of_a_cut_surrogate_pair(
    monkeypatch: pytest.MonkeyPatch, codec: str
) -> None:
    _windows(monkeypatch, "cp936")
    data = "ok 😀 done 日本".encode(codec)
    # "ok " is three code units; the emoji's high half follows, then its low half.
    head, tail = decode_split_output(data[:8], data[8:], 8)

    assert (head, tail) == ("ok ", " done 日本")


def test_windows_ascii_utf16_without_bom_keeps_its_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """ASCII-only UTF-16 looks just like NUL-delimited ASCII, as for whole output."""
    _windows(monkeypatch, "cp936")
    head = "abc".encode("utf-16-le")
    tail = "xyz".encode("utf-16-le")

    assert decode_split_output(head, tail, 40) == (decode_subprocess_output(head), decode_subprocess_output(tail))
    assert decode_split_output(head, tail, 40) == ("a\x00b\x00c\x00", "x\x00y\x00z\x00")


@pytest.mark.parametrize(
    ("code_page", "tail_text"),
    [("cp1252", "€ payment accepted"), ("cp936", "操作失败请重试")],
)
def test_windows_ascii_head_does_not_outvote_a_code_page_tail(
    monkeypatch: pytest.MonkeyPatch, code_page: str, tail_text: str
) -> None:
    _windows(monkeypatch, code_page)

    head, tail = decode_split_output(b"processing records\n", tail_text.encode(code_page), 500)

    assert (head, tail) == ("processing records\n", tail_text)


def test_windows_code_page_tail_starting_with_a_trail_byte_starts_at_the_next_character(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _windows(monkeypatch, "cp936")
    tail = "操作失败请重试".encode("gbk")[1:]

    assert decode_split_output(b"ok\n", tail, 101) == ("ok\n", "作失败请重试")


def test_windows_code_page_head_drops_only_its_cut_last_character(monkeypatch: pytest.MonkeyPatch) -> None:
    _windows(monkeypatch, "cp936")
    head = "第一行日志\n第二行".encode("gbk")[:-1]

    assert decode_split_output(head, "结束\n".encode("gbk"), 999) == ("第一行日志\n第二", "结束\n")


def test_windows_utf8_evidence_beats_the_code_page(monkeypatch: pytest.MonkeyPatch) -> None:
    _windows(monkeypatch, "cp1252")
    tail = "€ total".encode()[1:]

    assert decode_split_output("构建日志\n".encode(), tail, 300) == ("构建日志\n", " total")


def test_windows_bytes_that_only_look_like_a_cut_utf8_character_are_no_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _windows(monkeypatch, "cp1252")
    # "é" reads as the start of a UTF-8 character and "€" as the rest of one.
    head, tail = "café".encode("cp1252"), "€ total!".encode("cp1252")

    assert decode_split_output(head, tail, 300) == ("café", "€ total!")


def test_windows_utf8_console_decodes_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    _windows(monkeypatch, None)
    tail = "€ total".encode()[2:]

    assert decode_split_output(b"ok\n", tail, 300) == ("ok\n", " total")


def test_windows_parts_no_one_codec_fits_decode_on_their_own(monkeypatch: pytest.MonkeyPatch) -> None:
    _windows(monkeypatch, "cp936")
    head = "日志".encode()
    tail = b"\x80\xff bad \xfe"

    assert decode_split_output(head, tail, 300) == (decode_subprocess_output(head), decode_subprocess_output(tail))


# ---------------------------------------------------------------------------
# drain_process_pipes
# ---------------------------------------------------------------------------

_BOTH_STREAMS_SCRIPT = """\
import sys
out, err = sys.stdout.buffer, sys.stderr.buffer
# More than a pipe buffer on stderr before any stdout: a reader that finishes
# stdout first would never see the end of either.
for i in range(2000):
    err.write(b"err-%04d " % i + b"e" * 90 + b"\\n")
err.write(b"ERR-END\\n")
err.flush()
for i in range(2000):
    out.write(b"out-%04d " % i + b"o" * 90 + b"\\n")
out.write(b"OUT-END\\n")
out.flush()
"""


async def test_drain_reads_both_pipes_to_their_end_within_the_limit(tmp_path: Path) -> None:
    script = tmp_path / "both.py"
    script.write_text(_BOTH_STREAMS_SCRIPT, encoding="utf-8")
    stdout = BoundedCapture(3000)
    stderr = BoundedCapture(3000)

    async with managed_subprocess(
        sys.executable,
        str(script),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    ) as proc:
        await asyncio.wait_for(drain_process_pipes(proc, stdout, stderr), ENGINE_TURN_TIMEOUT)
        assert proc.returncode == 0

    out, err = stdout.snapshot(), stderr.snapshot()
    assert out.seen == err.seen == 2000 * 100 + 8
    assert (len(out.head), len(out.tail), len(err.head), len(err.tail)) == (1000, 2000, 1000, 2000)
    assert out.head.startswith(b"out-0000 ") and out.tail.endswith(b"out-1999 " + b"o" * 90 + b"\nOUT-END\n")
    assert err.head.startswith(b"err-0000 ") and err.tail.endswith(b"err-1999 " + b"e" * 90 + b"\nERR-END\n")

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Bounded capture of subprocess output streams.

A capture keeps the first third and the last two thirds of its byte limit and
counts what it dropped from the middle; the dropped bytes cannot be recovered.
It holds raw bytes and counts only: decoding happens once, at the end.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from chrys.foundation.platform.process import decode_split_output, decode_subprocess_output

OUTPUT_CAPTURE_LIMIT_BYTES = 32 * 1024 * 1024
"""Bytes a capture keeps of one stream unless its caller gives another limit."""

_READ_CHUNK_BYTES = 64 * 1024

_CUT_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*|\][^\x07\x1b\n]*\x1b?|[\x20-\x2F])?\Z")
"""The start of a CSI, OSC or two-byte escape sequence, on the last line, with nothing after it."""


@dataclass(frozen=True, slots=True)
class CapturedOutput:
    """The bytes a capture kept of one stream, and how many it saw."""

    head: bytes
    tail: bytes
    seen: int

    @property
    def dropped(self) -> int:
        return self.seen - len(self.head) - len(self.tail)

    def text(self, clean: Callable[[str], str] | None = None) -> str:
        """Decode the kept bytes, marking the gap when the middle was dropped.

        *clean* runs on each kept part on its own: the two parts are not one
        terminal stream, so no escape sequence or carriage return reaches
        across the gap. Before it runs, an escape sequence the gap cut short
        at the end of the head is dropped, since *clean* finds no end for it.
        """
        if not self.dropped:
            text = decode_subprocess_output(self.head + self.tail)
            return clean(text) if clean is not None else text
        head, tail = decode_split_output(self.head, self.tail, self.seen - len(self.tail))
        if clean is not None:
            head, tail = clean(_CUT_ESCAPE.sub("", head)), clean(tail)
        return f"{head}\n[... {self.dropped} bytes omitted ...]\n{tail}"


class StreamSink(Protocol):
    """Takes an output stream's bytes as they arrive."""

    def feed(self, chunk: bytes, /) -> None: ...


class BoundedCapture:
    """Keep the head and tail of one output stream within a byte limit."""

    def __init__(self, limit_bytes: int | None = None) -> None:
        limit = OUTPUT_CAPTURE_LIMIT_BYTES if limit_bytes is None else limit_bytes
        self._head_limit = limit // 3
        self._tail_limit = limit - self._head_limit
        self._head = bytearray()
        self._tail = bytearray()
        self._seen = 0

    def feed(self, chunk: bytes) -> None:
        self._seen += len(chunk)
        room = self._head_limit - len(self._head)
        if room > 0:
            self._head += chunk[:room]
            chunk = chunk[room:]
        if chunk:
            self._tail += chunk
            excess = len(self._tail) - self._tail_limit
            if excess > 0:
                del self._tail[:excess]

    def snapshot(self) -> CapturedOutput:
        return CapturedOutput(bytes(self._head), bytes(self._tail), self._seen)


def capture_limit_footer(captures: Sequence[CapturedOutput]) -> str:
    """Return the note for bytes *captures* dropped, or ``""`` when they kept everything."""
    dropped = sum(capture.dropped for capture in captures)
    if not dropped:
        return ""
    return f"[Output capture limit reached: {dropped} bytes from the middle were not kept.]"


async def drain_process_pipes(
    proc: asyncio.subprocess.Process,
    stdout: StreamSink,
    stderr: StreamSink,
) -> None:
    """Read *proc*'s output pipes to their end in parallel, then wait for it to exit."""

    async def pump(stream: asyncio.StreamReader | None, capture: StreamSink) -> None:
        if stream is None:
            return
        while chunk := await stream.read(_READ_CHUNK_BYTES):
            capture.feed(chunk)

    await asyncio.gather(pump(proc.stdout, stdout), pump(proc.stderr, stderr))
    await proc.wait()

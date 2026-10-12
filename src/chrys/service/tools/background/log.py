# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The log file a background command writes to once its tool call has returned."""

from __future__ import annotations

import asyncio
import codecs
import contextvars
import logging
import os
from contextlib import suppress
from typing import TYPE_CHECKING

from chrys.foundation.platform.files import replace_with_retry, secure_open_owner_only_append
from chrys.foundation.platform.process import decode_subprocess_output, subprocess_output_utf16_codec
from chrys.foundation.text.tool_output import process_carriage_returns, strip_ansi
from chrys.foundation.trajectory.keys import ensure_owner_only_directory

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from chrys.foundation.platform.output_capture import CapturedOutput

logger = logging.getLogger(__name__)

LOG_LIMIT_BYTES = 20 * 1024 * 1024
"""Size past which a log rotates, keeping the part before as ``<name>.1``."""

_FLUSH_DELAY = 0.25
"""Seconds output queues before it is written, so a chatty program costs a few writes a second."""

_QUEUE_LIMIT = 8 * 1024 * 1024
"""Bytes waiting for a write past which new output is dropped."""

_LINE_LIMIT = 64 * 1024
"""Bytes of an unfinished line held before it is written as a line of its own."""

_NOTES = -1
"""The stream icode's own lines arrive on, apart from the command's."""


def _clean_text(text: str) -> str:
    return process_carriage_returns(strip_ansi(text))


def _clean_line(raw: bytes) -> str:
    return _clean_text(decode_subprocess_output(raw.rstrip(b"\r")))


class TaskLog:
    """Cleaned output of one background command, appended off the event loop.

    ``feed`` only queues bytes, so the reader calling it never waits on the
    disk: one write at a time decodes, cleans and appends what queued. What
    cannot be queued or written is dropped — reading must never stop, or the
    program blocks writing to a full pipe. Each stream is cut into lines on
    its own, so a multibyte character is never split, and the streams merge
    line by line in the order they arrived. A stream that starts out in UTF-16
    (some Windows programs) is turned into UTF-8 before it is cut.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._captured: tuple[CapturedOutput, ...] = ()
        self._queue: list[tuple[int, bytes]] = []
        self._queued = 0
        self._dropped = 0
        self._timer: asyncio.TimerHandle | None = None
        self._write: asyncio.Task[None] | None = None
        self._closed = False
        # Touched only by the write in flight.
        self._partial: dict[int, bytearray] = {}
        self._utf16: dict[int, codecs.IncrementalDecoder | None] = {}
        self._fd: int | None = None
        self._size = 0

    def feed(self, stream: int, chunk: bytes) -> None:
        """Queue *chunk* of output stream *stream* for the next write."""
        if self._closed:
            return
        if self._queued + len(chunk) > _QUEUE_LIMIT:
            self._dropped += len(chunk)
        else:
            self._queue.append((stream, chunk))
            self._queued += len(chunk)
        if self._timer is None and self._write is None:
            self._timer = asyncio.get_running_loop().call_later(_FLUSH_DELAY, self._start_write)

    def note(self, line: str) -> None:
        """Queue *line*, one of icode's own, after the output that arrived until now."""
        self.feed(_NOTES, f"{line}\n".encode())

    def open(self) -> int:
        """Create the log and its directory, so both exist before the command prints anything."""
        ensure_owner_only_directory(self.path.parent)
        handle = secure_open_owner_only_append(self.path)
        self._fd = handle.fd
        self._size = os.fstat(handle.fd).st_size
        return handle.fd

    def start(self, captured: Sequence[CapturedOutput]) -> None:
        """Write *captured*, what the command printed before it moved to the background, first."""
        self._captured = tuple(captured)
        if self._timer is None and self._write is None:
            self._start_write()

    async def close(self, last_line: str) -> None:
        """Write what is still queued, then *last_line*, and close the file."""
        if self._closed:
            return
        self._closed = True
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._write is not None:
            await asyncio.wait({self._write})
        batch, dropped = self._take()
        await asyncio.to_thread(self._append, batch, dropped, last_line)

    def _take(self) -> tuple[list[tuple[int, bytes]], int]:
        batch, dropped = self._queue, self._dropped
        self._queue, self._queued, self._dropped = [], 0, 0
        return batch, dropped

    def _start_write(self) -> None:
        self._timer = None
        batch, dropped = self._take()
        # A fresh context: the write belongs to the log, not to whatever task fed it.
        write = asyncio.get_running_loop().create_task(
            asyncio.to_thread(self._append, batch, dropped, None), context=contextvars.Context()
        )
        self._write = write
        # Fresh again, or the next timer would carry whichever context started the first write.
        write.add_done_callback(self._written, context=contextvars.Context())

    def _written(self, write: asyncio.Task[None]) -> None:
        self._write = None
        if not write.cancelled() and write.exception() is not None:
            logger.debug("Background task log write failed", exc_info=write.exception())
        if not self._closed and (self._queue or self._dropped):
            self._timer = asyncio.get_running_loop().call_later(_FLUSH_DELAY, self._start_write)

    def _append(self, batch: list[tuple[int, bytes]], dropped: int, last_line: str | None) -> None:
        texts = [capture.text(_clean_text) for capture in self._captured]
        parts = [text if text.endswith("\n") else f"{text}\n" for text in texts if text]
        self._captured = ()
        for stream, chunk in batch:
            if stream not in self._utf16:
                codec = subprocess_output_utf16_codec(chunk)
                self._utf16[stream] = codecs.getincrementaldecoder(codec)(errors="replace") if codec else None
            if (decoder := self._utf16[stream]) is not None:
                chunk = decoder.decode(chunk).encode("utf-8")
            pending = self._partial.setdefault(stream, bytearray())
            pending += chunk
            *lines, rest = bytes(pending).split(b"\n")
            parts.extend(f"{_clean_line(line)}\n" for line in lines)
            if len(rest) > _LINE_LIMIT:
                parts.append(f"{_clean_line(rest)}\n")
                rest = b""
            pending[:] = rest
        if dropped:
            parts.append(f"[... {dropped} bytes of output dropped ...]\n")
        if last_line is not None:
            parts.extend(f"{_clean_line(bytes(line))}\n" for line in self._partial.values() if line.strip())
            self._partial.clear()
            parts.append(f"{last_line}\n")
        try:
            if parts:
                self._write_text("".join(parts))
        except OSError:
            logger.debug("Background task log %s could not be written", self.path, exc_info=True)
        finally:
            if last_line is not None and self._fd is not None:
                with suppress(OSError):
                    os.close(self._fd)
                self._fd = None

    def _write_text(self, text: str) -> None:
        data = text.encode("utf-8", errors="replace")
        if self._fd is not None and self._size + len(data) > LOG_LIMIT_BYTES:
            os.close(self._fd)
            self._fd = None
            replace_with_retry(self.path, self.path.with_name(f"{self.path.name}.1"))
        fd = self._fd if self._fd is not None else self.open()
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
            self._size += written

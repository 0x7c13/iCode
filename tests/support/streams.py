# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Text stream doubles for code that writes to ``sys.stdout`` or ``sys.stderr``."""

from __future__ import annotations

import errno
import io


class FailingTextStream(io.TextIOBase):
    """A text stream whose every write fails: ``BrokenPipeError`` (the reader has gone) or an ``EIO`` ``OSError``.

    ``fileno()`` reports ``descriptor`` (a pipe end the test owns and closes); without one it raises, as an
    in-memory stream's does. ``writes`` counts the attempts.
    """

    def __init__(self, descriptor: int | None = None, *, broken_pipe: bool = True) -> None:
        super().__init__()
        self.descriptor = descriptor
        self.broken_pipe = broken_pipe
        self.writes = 0

    def fileno(self) -> int:
        if self.descriptor is None:
            raise io.UnsupportedOperation("fileno")
        return self.descriptor

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def write(self, s: str, /) -> int:
        self.writes += 1
        if self.broken_pipe:
            raise BrokenPipeError(errno.EPIPE, "The reader has gone.")
        raise OSError(errno.EIO, "Input/output error")

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A pseudo-terminal master that ends the way Linux has it end, on every platform."""

from __future__ import annotations

import asyncio
import errno
import os

from chrys.foundation.platform.pty_output import PtyOutputProtocol


def linux_style_master() -> tuple[type[PtyOutputProtocol], asyncio.Event]:
    """A protocol to put in place of `PtyOutputProtocol`, and the event set once it was told of the end.

    Linux reports a program's exit as an I/O error on the master and never as end of file; macOS
    reports end of file. The difference only shows to a reader that is busy when the end comes, which
    the event lets a test arrange: keep the reader away until it is set.
    """
    ended = asyncio.Event()

    class LinuxStyleMaster(PtyOutputProtocol):
        def eof_received(self) -> bool:
            # Linux has no such thing to say. The transport goes on to close whatever we answer.
            return True

        def connection_lost(self, exc: Exception | None) -> None:
            super().connection_lost(exc if exc is not None else OSError(errno.EIO, os.strerror(errno.EIO)))
            ended.set()

    return LinuxStyleMaster, ended

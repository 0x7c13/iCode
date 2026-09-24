# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""`PtyOutputProtocol`: a master that fails ends the output, after what it had already delivered."""

from __future__ import annotations

import asyncio
import errno
import os

import pytest

from chrys.foundation.platform.pty_output import PtyOutputProtocol


async def test_master_failing_ends_the_output_after_what_was_already_read() -> None:
    # Linux reports the program's exit as an I/O error on the master. It may come while whoever reads
    # is still busy with an earlier read, and must not cost what arrived in between.
    reader = asyncio.StreamReader()
    protocol = PtyOutputProtocol(reader)

    protocol.data_received(b"last words")
    protocol.connection_lost(OSError(errno.EIO, os.strerror(errno.EIO)))

    assert await reader.read(64) == b"last words"
    assert await reader.read(64) == b""


async def test_master_closing_in_order_ends_the_output_too() -> None:
    reader = asyncio.StreamReader()
    protocol = PtyOutputProtocol(reader)

    protocol.data_received(b"last words")
    protocol.connection_lost(None)

    assert await reader.read(64) == b"last words"
    assert await reader.read(64) == b""


async def test_error_that_is_not_the_masters_is_still_raised() -> None:
    reader = asyncio.StreamReader()
    protocol = PtyOutputProtocol(reader)

    protocol.connection_lost(RuntimeError("not an I/O error"))

    with pytest.raises(RuntimeError, match="not an I/O error"):
        await reader.read(64)

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Reading the master of a pseudo-terminal to the end of what its program wrote."""

from __future__ import annotations

import asyncio


class PtyOutputProtocol(asyncio.StreamReaderProtocol):
    """Ends a pseudo-terminal's output the same way however its master does.

    Linux reports the program's exit as an I/O error on the master, not as end of file, and a
    `StreamReader` raises an error it was given ahead of handing over what it still holds: the
    program's last words, whenever whoever reads was busy with the ones before.
    """

    def connection_lost(self, exc: Exception | None) -> None:
        super().connection_lost(None if isinstance(exc, OSError) else exc)

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A background command whose end the test decides, for the session's background-task registry."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from chrys.foundation.platform.output_capture import CapturedOutput


class HeldProcess:
    """A command the registry adopts; it runs until the test ends it or the registry stops it.

    Build it inside the running event loop.
    """

    def __init__(self, *, ignores: tuple[bool, ...] = (), kept: bytes = b"before\n") -> None:
        """*ignores* lists the stop requests (False: terminate, True: kill) it survives."""
        self.pid = 4242
        self.returncode = -1
        self.sink: Callable[[int, bytes], None] | None = None
        self.signals: list[bool] = []
        self.closed = False
        self._ignores = ignores
        self._kept = kept
        loop = asyncio.get_running_loop()
        self._outlived: asyncio.Future[tuple[int, ...]] = loop.create_future()
        self._code: asyncio.Future[int] = loop.create_future()

    def hand_off(self, sink: Callable[[int, bytes], None]) -> tuple[CapturedOutput, ...]:
        self.sink = sink
        return (CapturedOutput(self._kept, b"", len(self._kept)),)

    async def shell_exited(self) -> tuple[int, ...]:
        await asyncio.wait({self._outlived, self._code}, return_when=asyncio.FIRST_COMPLETED)
        return self._outlived.result() if self._outlived.done() else ()

    async def wait(self) -> int:
        return await asyncio.shield(self._code)

    async def terminate(self, *, force: bool) -> None:
        self.signals.append(force)
        if force not in self._ignores:
            self.exit(-9 if force else -15)

    async def close(self) -> None:
        self.closed = True
        self.exit(-9)

    def outlive(self, code: int, pids: tuple[int, ...]) -> None:
        """Let its shell exit with *code* while the processes *pids* run on."""
        self.returncode = code
        self._outlived.set_result(pids)

    def exit(self, code: int) -> None:
        """End it, its shell exiting with *code* unless it already exited."""
        if not self._outlived.done():
            self.returncode = code
        if not self._code.done():
            self._code.set_result(self.returncode)

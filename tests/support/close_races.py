# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Event handshakes for close and rollback tests, independent of ready-queue order."""

from __future__ import annotations

import asyncio

import pytest


async def assert_entered_before_completion(entered: asyncio.Event, closing: asyncio.Task[None]) -> None:
    """Require the real waiting entry to be reached before close can return."""
    witness = asyncio.create_task(entered.wait())
    try:
        await asyncio.wait_for(asyncio.wait([closing, witness], return_when=asyncio.FIRST_COMPLETED), 5)
        assert entered.is_set(), "close returned without entering the required drain"
    finally:
        witness.cancel()
        await asyncio.gather(witness, return_exceptions=True)


class ReleaseGate:
    """A resource release that records completion only after its gate opens."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.proceed = asyncio.Event()
        self.completed = asyncio.Event()

    async def __call__(self, *_args: object) -> None:
        self.entered.set()
        await self.proceed.wait()
        self.completed.set()


async def assert_cancel_during_rollback(task: asyncio.Task[object], release: ReleaseGate) -> None:
    """Cancel after rollback starts; cleanup must finish without uncancelling."""
    try:
        await asyncio.wait_for(release.entered.wait(), 5)
        assert task.cancelling() == 0
        assert task.cancel() is True
        assert task.cancelling() == 1
        release.proceed.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(task), 5)
        assert release.completed.is_set()
        assert task.cancelled() is True
        assert task.cancelling() == 1
    finally:
        release.proceed.set()
        await asyncio.gather(task, return_exceptions=True)

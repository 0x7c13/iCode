# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Cancellation and stale-result fences for one workflow UI flow."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chrys.app.tui.screens.main.workflow_controller import WorkflowController


type FlowToken = tuple[int, int]


class WorkflowFlow:
    def __init__(self, host: WorkflowController) -> None:
        self.host = host
        self.revision = 0
        self.task: asyncio.Task[None] | None = None

    def current(self, token: FlowToken) -> bool:
        return not self.host.closed and token == (self.host.generation, self.revision)

    def invalidate(self) -> None:
        self.revision += 1
        if self.task is not None:
            self.host.cancel_task(self.task)
            self.task = None

    def start(self, operation: Callable[[FlowToken], Coroutine[Any, Any, None]]) -> asyncio.Task[None]:
        previous = self.task
        self.invalidate()
        token = self.host.generation, self.revision

        async def run() -> None:
            if previous is not None:
                await asyncio.gather(previous, return_exceptions=True)
            if self.current(token):
                await operation(token)

        self.task = self.host.spawn(run())
        return self.task

    @property
    def active(self) -> bool:
        return self.task is not None and not self.task.done()

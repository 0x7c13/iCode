# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Progress bridge bound by existing invocation event middleware."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextvars import ContextVar

web_progress_callback: ContextVar[Callable[[list[str]], Awaitable[None]] | None] = ContextVar(
    "web_progress", default=None
)


async def report_web_progress(message: str) -> None:
    callback = web_progress_callback.get()
    if callback is not None:
        await callback([message])

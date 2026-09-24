# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for InterruptMiddleware: the interrupt flag gates the tool call chain."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from chrys.service.agent_middleware.control.interrupt import InterruptMiddleware


async def test_interrupt_not_set_passes_through() -> None:
    mw = InterruptMiddleware()
    called = False

    async def call_next() -> None:
        nonlocal called
        called = True

    # Minimal FunctionInvocationContext mock
    ctx = SimpleNamespace()
    await mw.process(ctx, call_next)
    assert called


async def test_interrupt_set_raises() -> None:
    mw = InterruptMiddleware()
    mw.set_interrupted()

    async def call_next() -> None:
        pass

    ctx = SimpleNamespace()
    with pytest.raises(Exception, match="interrupted"):
        await mw.process(ctx, call_next)


async def test_interrupt_reset_clears_flag() -> None:
    mw = InterruptMiddleware()
    mw.set_interrupted()
    mw.reset()

    called = False

    async def call_next() -> None:
        nonlocal called
        called = True

    ctx = SimpleNamespace()
    await mw.process(ctx, call_next)
    assert called

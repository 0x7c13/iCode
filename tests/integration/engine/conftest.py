# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared fixtures for the engine pipeline integration tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.support.pipeline_helpers import PipelineTestContext, create_test_engine, restore_mock_client

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable
    from pathlib import Path

    from chrys.service.llm.mock import MockResponse


@pytest.fixture
async def make_pipeline_ctx(
    tmp_path: Path,
) -> AsyncIterator[Callable[..., Awaitable[PipelineTestContext]]]:
    """Return a factory building engine contexts that are torn down with the test.

    Replaces the per-test ``try: ... finally: await ctx.cleanup()`` boilerplate.
    Every context the factory hands out is shut down here, newest first, and any
    patch a test installed over the mock client's inner call is dropped before
    the shutdown so it always runs against the unpatched client — the ordering
    the hand-written ``finally`` blocks used.
    """
    contexts: list[PipelineTestContext] = []

    async def _make(responses: list[MockResponse], **kwargs: Any) -> PipelineTestContext:
        ctx = await create_test_engine(responses, tmp_path, **kwargs)
        contexts.append(ctx)
        return ctx

    yield _make

    for ctx in reversed(contexts):
        restore_mock_client(ctx)
        await ctx.cleanup()

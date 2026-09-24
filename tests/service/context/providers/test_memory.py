# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for MemoryProvider — the system-prompt injector."""

from __future__ import annotations

import pytest

from chrys.kernel import AgentSession, SessionContext
from chrys.service.context.providers.memory import MemoryProvider


@pytest.mark.asyncio
async def test_injects_instructions_when_text_non_empty() -> None:
    text = "# Auto-loaded memory\n\n## File: a.md\n\nbody"
    provider = MemoryProvider(text)
    session = AgentSession()
    context = SessionContext(input_messages=[])

    await provider.before_run(agent=None, session=session, context=context, state={})

    assert len(context.instructions) == 1
    assert context.instructions[0] == text


@pytest.mark.asyncio
async def test_no_op_when_text_empty() -> None:
    provider = MemoryProvider("")
    session = AgentSession()
    context = SessionContext(input_messages=[])

    await provider.before_run(agent=None, session=session, context=context, state={})

    # Provider must be silent when there is nothing to inject — otherwise
    # an empty section would clutter every system prompt.
    assert len(context.instructions) == 0


@pytest.mark.asyncio
async def test_injects_on_every_call() -> None:
    """``before_run`` is called per LLM invocation; injection must be repeated."""
    text = "memory text"
    provider = MemoryProvider(text)
    session = AgentSession()

    for _ in range(3):
        context = SessionContext(input_messages=[])
        await provider.before_run(agent=None, session=session, context=context, state={})
        assert context.instructions == [text]


def test_default_source_id() -> None:
    """Source ID is stable so callers can identify our provider in tooling."""
    assert MemoryProvider.DEFAULT_SOURCE_ID == "chrys_memory"
    p = MemoryProvider("x")
    assert p.source_id == "chrys_memory"


# ---------------------------------------------------------------------------
# ContextManager wiring
# ---------------------------------------------------------------------------


class TestContextManagerWiring:
    def test_provider_present_when_memory_text_non_empty(self) -> None:
        from chrys.service.context.manager import ContextManager
        from chrys.service.profiles.models.resolver import default_profile

        ctx = ContextManager(default_profile(), memory_text="some memory")

        assert any(isinstance(p, MemoryProvider) for p in ctx.providers)

    def test_provider_absent_when_memory_text_empty(self) -> None:
        from chrys.service.context.manager import ContextManager
        from chrys.service.profiles.models.resolver import default_profile

        ctx = ContextManager(default_profile(), memory_text="")

        assert not any(isinstance(p, MemoryProvider) for p in ctx.providers)

    def test_provider_order(self) -> None:
        """MemoryProvider sits between ContextManagementProvider and history."""
        from chrys.service.context.manager import ContextManager
        from chrys.service.context.providers.context_management import ContextManagementProvider
        from chrys.service.context.providers.history import CompressibleHistoryProvider
        from chrys.service.profiles.models.resolver import default_profile

        ctx = ContextManager(default_profile(), memory_text="memory")
        types = [type(p).__name__ for p in ctx.providers]

        mgmt_idx = types.index(ContextManagementProvider.__name__)
        mem_idx = types.index(MemoryProvider.__name__)
        hist_idx = types.index(CompressibleHistoryProvider.__name__)
        assert mgmt_idx < mem_idx < hist_idx

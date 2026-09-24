# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Lifecycle coverage for the shared agent_engine fixture's EngineTracker."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from typing import Any
from unittest.mock import create_autospec

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from tests.support.ci import CI_LINUX_ONLY
from tests.support.engines import EngineTracker
from tests.support.paths import REPO_ROOT

# Platform-independent test-infra self-checks: the Linux CI job covers them.
pytestmark = CI_LINUX_ONLY


def test_registering_engine_fixtures_does_not_import_the_backend() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import tests.support.engines; assert 'chrys.orchestration.engine.assembly' not in sys.modules",
        ],
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
    )


class _FakeLifecycle:
    def __init__(self, calls: list[dict[str, bool]]) -> None:
        self._calls = calls

    async def close_session(self) -> None:
        self._calls.append({"release_lock": True, "close_mcp": False})

    async def close_session_in_place(self) -> None:
        self._calls.append({"release_lock": False, "close_mcp": False})


class _FakeEngine:
    def __init__(self) -> None:
        self.shutdown_calls: list[dict[str, bool]] = []
        self.lifecycle = _FakeLifecycle(self.shutdown_calls)
        self.started = 0
        self.fail_next_shutdown = False
        self.cancel_next_shutdown = False

    async def shutdown(self) -> None:
        if self.cancel_next_shutdown:
            self.cancel_next_shutdown = False
            raise asyncio.CancelledError("mid-shutdown cancellation")
        if self.fail_next_shutdown:
            self.fail_next_shutdown = False
            raise RuntimeError("mid-shutdown failure")
        self.shutdown_calls.append(_FULL)

    async def start(self, profile: Any = None) -> None:
        self.started += 1


_FULL = {"release_lock": True, "close_mcp": True}


async def test_full_shutdown_is_terminal_and_not_repeated() -> None:
    tracker = EngineTracker()
    engine = tracker.register(_FakeEngine())

    await engine.shutdown()
    await tracker.finalize()

    assert engine.shutdown_calls == [_FULL]


@pytest.mark.parametrize("in_place", [False, True])
async def test_partial_shutdown_stays_pending_for_final_cleanup(in_place: bool) -> None:
    tracker = EngineTracker()
    engine = tracker.register(_FakeEngine())

    if in_place:
        await engine.lifecycle.close_session_in_place()
    else:
        await engine.lifecycle.close_session()
    await tracker.finalize()

    assert len(engine.shutdown_calls) == 2
    assert engine.shutdown_calls[-1] == _FULL


@pytest.mark.parametrize("keyword", ["release_session_lock", "close_mcp_cache"])
async def test_shutdown_rejects_resource_selection_keywords(keyword: str) -> None:
    tracker = EngineTracker()
    engine = tracker.register(_FakeEngine())

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        await engine.shutdown(**{keyword: False})
    await tracker.finalize()

    assert engine.shutdown_calls == [_FULL]


async def test_raised_shutdown_is_retried_at_teardown() -> None:
    tracker = EngineTracker()
    engine = tracker.register(_FakeEngine())
    engine.fail_next_shutdown = True

    with pytest.raises(RuntimeError, match="mid-shutdown failure"):
        await engine.shutdown()
    await tracker.finalize()

    assert engine.shutdown_calls == [_FULL]


async def test_restart_after_full_shutdown_rearms_cleanup() -> None:
    """The SessionNew/restore shape: shut down, then start the engine again."""
    tracker = EngineTracker()
    engine = tracker.register(_FakeEngine())

    await engine.shutdown()
    await engine.start()
    await tracker.finalize()

    assert engine.started == 1
    assert len(engine.shutdown_calls) == 2
    assert engine.shutdown_calls[-1] == _FULL


async def test_finalize_attempts_every_engine_despite_a_failure() -> None:
    tracker = EngineTracker()
    first = tracker.register(_FakeEngine())
    failing = tracker.register(_FakeEngine())
    last = tracker.register(_FakeEngine())
    failing.fail_next_shutdown = True

    with pytest.raises(RuntimeError, match="mid-shutdown failure"):
        await tracker.finalize()

    assert first.shutdown_calls == [_FULL], "engines after the failure must still be cleaned up"
    assert last.shutdown_calls == [_FULL]
    assert failing.shutdown_calls == []


async def test_finalize_aggregates_multiple_failures() -> None:
    tracker = EngineTracker()
    engines = [tracker.register(_FakeEngine()) for _ in range(3)]
    engines[0].fail_next_shutdown = True
    engines[2].fail_next_shutdown = True

    with pytest.raises(ExceptionGroup) as info:
        await tracker.finalize()

    assert len(info.value.exceptions) == 2
    assert engines[1].shutdown_calls == [_FULL]


async def test_finalize_continues_past_cancellation() -> None:
    """CancelledError is a BaseException — it must not truncate cleanup."""
    tracker = EngineTracker()
    first = tracker.register(_FakeEngine())
    cancelled = tracker.register(_FakeEngine())
    last = tracker.register(_FakeEngine())
    cancelled.cancel_next_shutdown = True

    with pytest.raises(asyncio.CancelledError):
        await tracker.finalize()

    assert first.shutdown_calls == [_FULL], "engines after the cancellation must still be cleaned up"
    assert last.shutdown_calls == [_FULL]
    assert cancelled.shutdown_calls == []


async def test_finalize_aggregates_cancellation_with_failures() -> None:
    tracker = EngineTracker()
    engines = [tracker.register(_FakeEngine()) for _ in range(3)]
    engines[0].cancel_next_shutdown = True
    engines[2].fail_next_shutdown = True

    with pytest.raises(BaseExceptionGroup) as info:
        await tracker.finalize()

    assert len(info.value.exceptions) == 2
    assert any(isinstance(exc, asyncio.CancelledError) for exc in info.value.exceptions)
    assert engines[1].shutdown_calls == [_FULL]


async def test_finalize_shuts_down_in_reverse_registration_order() -> None:
    tracker = EngineTracker()
    order: list[int] = []

    class _OrderedEngine(_FakeEngine):
        def __init__(self, tag: int) -> None:
            super().__init__()
            self._tag = tag

        async def shutdown(self) -> None:
            order.append(self._tag)
            await super().shutdown()

    tracker.register(_OrderedEngine(1))
    tracker.register(_OrderedEngine(2))
    await tracker.finalize()

    assert order == [2, 1]


async def test_agent_engine_factory_preserves_assembly_arguments(agent_engine, monkeypatch) -> None:
    from chrys.orchestration.engine import assembly

    bus = EventBus()
    settings = Settings()
    built = _FakeEngine()
    assemble = create_autospec(assembly.assemble_agent_engine, return_value=built)
    monkeypatch.setattr(assembly, "assemble_agent_engine", assemble)

    engine = agent_engine(bus, settings, allow_user_interaction=False)
    assert engine is built
    assemble.assert_called_once_with(bus, settings, allow_user_interaction=False)
    await engine.start()
    await engine.shutdown()
    assert engine.started == 1
    assert engine.shutdown_calls == [_FULL]

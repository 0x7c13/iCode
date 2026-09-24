# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Lifecycle-safe AgentEngine factory fixture."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol
from unittest.mock import create_autospec

import pytest

if TYPE_CHECKING:
    from chrys.orchestration.engine.engine import AgentEngine
    from chrys.orchestration.engine.state.current_agent import CurrentAgent
    from chrys.orchestration.engine.state.machine import EngineStateMachine
    from chrys.orchestration.engine.trajectory import TrajectoryRecorder
    from chrys.service.mutations.workspace_changes import WorkspaceChangeTracker
    from chrys.service.session.history import SessionHistoryManager
    from chrys.service.session.persistence import SessionPersistence


class AgentEngineFactory(Protocol):
    """Construct an engine that the fixture will shut down after the test."""

    def __call__(self, *args: Any, **kwargs: Any) -> AgentEngine: ...


class EngineTracker:
    """Track engines and whether each reached a FULL terminal shutdown.

    ``AgentEngine`` exposes no shutdown-completed signal (``_shutting_down``
    is set on entry and ``EngineState`` has no terminal state), so completion
    is tracked by wrapping ``shutdown``. Only a successful engine shutdown counts:
    the session lifecycle legitimately performs PARTIAL shutdowns before
    resetting or restarting an engine, and those must stay pending for final
    cleanup. A later ``start()`` re-arms cleanup after a full shutdown.
    """

    def __init__(self) -> None:
        self._engines: list[Any] = []
        self._fully_shut_down: set[int] = set()

    def register(self, engine: Any) -> Any:
        original_shutdown = engine.shutdown
        original_start = engine.start

        async def _tracked_shutdown() -> None:
            await original_shutdown()
            self._fully_shut_down.add(id(engine))

        async def _tracked_start(*args: Any, **kwargs: Any) -> Any:
            self._fully_shut_down.discard(id(engine))
            return await original_start(*args, **kwargs)

        engine.shutdown = _tracked_shutdown
        engine.start = _tracked_start
        self._engines.append(engine)
        return engine

    async def finalize(self) -> None:
        """Fully shut down, in reverse order, every engine still pending.

        Every pending engine is attempted even when one shutdown raises or is
        cancelled — stopping early would leave the remaining engines' session
        locks, background tasks, and MCP caches alive in the xdist worker.
        CancelledError is a BaseException, so the handler must catch
        BaseException; a sole cancellation is re-raised as-is so it still
        propagates as cancellation, and mixed failures become a
        BaseExceptionGroup.
        """
        failures: list[BaseException] = []
        for engine in reversed(self._engines):
            if id(engine) in self._fully_shut_down:
                continue
            try:
                await engine.shutdown()
            except BaseException as exc:
                failures.append(exc)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("engine teardown failures", failures)


@pytest.fixture
async def agent_engine() -> AsyncIterator[AgentEngineFactory]:
    """Return a factory and shut down every constructed engine in reverse order."""
    tracker = EngineTracker()

    def _factory(*args: Any, **kwargs: Any) -> AgentEngine:
        # Registering this plugin is global; constructing an engine is opt-in.
        from chrys.orchestration.engine.assembly import assemble_agent_engine

        return tracker.register(assemble_agent_engine(*args, **kwargs))

    yield _factory

    await tracker.finalize()


@dataclass
class AssembledServices:
    """Shared services retained by an integration test at the composition boundary."""

    history: SessionHistoryManager
    fsm: EngineStateMachine
    persistence: SessionPersistence
    trajectory_recorder: TrajectoryRecorder
    workspace_change_tracker: WorkspaceChangeTracker
    on_successful_turn: Callable[[], None]
    on_turn_started: Callable[[], None]


@pytest.fixture
def engine_services(monkeypatch: pytest.MonkeyPatch) -> Callable[[AgentEngine], AssembledServices]:
    """Capture real constructor dependencies without inspecting component internals."""
    from chrys.orchestration.engine import assembly

    captured: dict[CurrentAgent, AssembledServices] = {}
    construct = assembly.TurnCoordinator

    def retain(**dependencies):
        captured[dependencies["current"]] = AssembledServices(
            history=dependencies["history"],
            fsm=dependencies["fsm"],
            persistence=dependencies["persistence"],
            trajectory_recorder=dependencies["trajectory_recorder"],
            workspace_change_tracker=dependencies["workspace_change_tracker"],
            on_successful_turn=dependencies["on_successful_turn"],
            on_turn_started=dependencies["on_turn_started"],
        )
        return construct(**dependencies)

    monkeypatch.setattr(assembly, "TurnCoordinator", create_autospec(construct, side_effect=retain))

    def resolve(engine: AgentEngine) -> AssembledServices:
        return captured[engine.current]

    return resolve

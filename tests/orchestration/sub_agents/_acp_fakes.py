# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared doubles and builders for the ACP sub-agent translator, permission-broker, and controller tests."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar

import pytest
from acp.schema import SessionNotification, ToolCallProgress

from chrys.foundation.models.invocations import InvocationOrigin
from chrys.orchestration.invoker import acp as acp_module
from chrys.orchestration.invoker.acp_protocol import AcpPermissionBroker
from chrys.orchestration.sub_agents.acp_policy import AcpSubAgentPolicy
from chrys.orchestration.sub_agents.shell import SubAgentToolShell
from chrys.service.acp_client import AcpAgentSpec, AcpPromptOutcome
from chrys.service.approval.policy import ApprovalMode
from chrys.service.approval.turn_context import TurnContextHolder

if TYPE_CHECKING:
    from pathlib import Path

    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.trajectory.context import TrajectoryContext


def session_notification(update: Any) -> SessionNotification:
    return SessionNotification(sessionId="remote", update=update)


def running_tool_call(call_id: str, title: str) -> SessionNotification:
    """An in-progress ``tool_call_update`` for ``call_id`` as the remote would stream it."""
    return session_notification(
        ToolCallProgress(sessionUpdate="tool_call_update", toolCallId=call_id, title=title, status="in_progress")
    )


def make_broker(
    bus: EventBus,
    mode: list[ApprovalMode],
    *,
    timeout: float | None = 1,
    allow_user_interaction: bool = True,
    trajectory_context: TrajectoryContext | None = None,
    trajectory_boundary_operation_id: str | None = None,
) -> AcpPermissionBroker:
    context = TurnContextHolder()
    context.replace(["original prompt"])
    return AcpPermissionBroker(
        event_bus=bus,
        session_id="parent",
        caller_name="External",
        mode_getter=lambda: mode[0],
        turn_context=context,
        workspace_roots=["/workspace"],
        workspace_cwd="/workspace",
        approval_judge=None,
        ask_user_timeout_seconds=timeout,
        allow_user_interaction=allow_user_interaction,
        trajectory_context=trajectory_context,
        trajectory_boundary_operation_id=trajectory_boundary_operation_id,
    )


def make_spec(tmp_path: Path) -> AcpAgentSpec:
    return AcpAgentSpec(command="stub", cwd=str(tmp_path), stderr_log_path=tmp_path / "stderr.log")


class AcpClientDouble:
    """Root of the ACP client doubles: teardown bookkeeping and nothing else.

    ``force_closes`` counts teardowns of every double, subclasses included, so
    :func:`install_client` can reset it in one place. Everything a scenario actually
    drives — a script, an update sink, the stateful-phase flag — belongs to the
    subclass that needs it: a client whose scenario never had one must still fail
    loudly if production reaches for it.
    """

    force_closes: ClassVar[int] = 0

    async def cancel(self) -> None:
        return

    async def force_close(self) -> None:
        AcpClientDouble.force_closes += 1


def _handshake() -> Any:
    """The ``session/new`` result a remote returns."""
    return SimpleNamespace(session_id="remote", agent_info=None)


class FakeAcpClient(AcpClientDouble):
    """Scripted ``AcpAgentClient`` stand-in — one script entry per controller attempt.

    ``scripts`` is consumed with ``pop(0)``, and ``prompt`` requires its key: a test
    that installs this client owes it an entry per attempt, so an unscripted attempt
    fails instead of quietly succeeding. An entry maps ``connect`` / ``open`` to an
    exception to raise, and ``prompt`` to the outcome to return or the exception to
    raise. Install through :func:`install_client`, which resets the queue and the
    counter under ``monkeypatch`` for the duration of the test.
    """

    scripts: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, spec: AcpAgentSpec, callbacks: Any, **_kwargs: Any) -> None:
        self.spec = spec
        self.callbacks = callbacks
        self.script: dict[str, Any] = self.scripts.pop(0)
        # Mirrors the real client: flips only once session/new is sent, so
        # scripted connect/open failures stay in the stateless retry window.
        self.stateful_phase_started = False

    async def connect(self) -> None:
        outcome = self.script.get("connect")
        if isinstance(outcome, BaseException):
            raise outcome

    async def open_session(self) -> Any:
        outcome = self.script.get("open")
        if isinstance(outcome, BaseException):
            raise outcome
        self.stateful_phase_started = True
        return _handshake()

    async def prompt(self, _prompt: str) -> AcpPromptOutcome:
        outcome = self.script["prompt"]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class StreamingAcpClient(AcpClientDouble):
    """Base for the clients that stream remote updates through the translator.

    Connect and session/new always succeed and there is deliberately no script:
    these scenarios are about what the controller does with the updates, so a
    subclass overrides only the phase it streams from.
    """

    def __init__(self, spec: AcpAgentSpec, callbacks: Any, **kwargs: Any) -> None:
        self.spec = spec
        self.callbacks = callbacks
        self.stateful_phase_started = False
        # The translator the controller hands the client; subclasses stream
        # remote updates through it.
        self._sink: Any = kwargs["update_sink"]

    async def connect(self) -> None:
        return

    async def open_session(self) -> Any:
        self.stateful_phase_started = True
        return _handshake()


def install_client(
    monkeypatch: pytest.MonkeyPatch,
    client: type[AcpClientDouble] = FakeAcpClient,
    scripts: list[dict[str, Any]] | None = None,
) -> type[AcpClientDouble]:
    """Route the controller's ``AcpAgentClient`` to ``client`` with a fresh script queue and teardown counter.

    The queue lives on :class:`FakeAcpClient` and the counter on
    :class:`AcpClientDouble`, so every double shares one counter; ``monkeypatch``
    restores both when the test ends, so nothing leaks into the next test even when
    xdist runs several modules in one process.
    """
    monkeypatch.setattr(FakeAcpClient, "scripts", list(scripts or []))
    monkeypatch.setattr(AcpClientDouble, "force_closes", 0)
    monkeypatch.setattr(acp_module, "AcpAgentClient", client)
    return client


def hanging_client() -> type[AcpClientDouble]:
    """A client whose ``prompt`` blocks until ``cancel`` is called.

    ``entered`` is set on entering ``prompt``; ``cancel`` sets ``cancelled``,
    which releases it. The events are created inside the calling test's loop
    rather than at import time, which is why the class is built per call. The
    scenario never streams and never fails a phase, so the client carries
    neither an update sink nor the stateful-phase flag.
    """

    class HangingClient(AcpClientDouble):
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        def __init__(self, spec: AcpAgentSpec, callbacks: Any, **_kwargs: Any) -> None:
            self.spec = spec
            self.callbacks = callbacks

        async def connect(self) -> None:
            return

        async def open_session(self) -> Any:
            return _handshake()

        async def prompt(self, _prompt: str) -> AcpPromptOutcome:
            type(self).entered.set()
            await type(self).cancelled.wait()
            return AcpPromptOutcome(stop_reason="cancelled", usage=None)

        async def cancel(self) -> None:
            type(self).cancelled.set()

    return HangingClient


class LateCloseClient(StreamingAcpClient):
    """Ends the turn normally, then streams a tool-call start from inside ``force_close``."""

    async def prompt(self, _prompt: str) -> AcpPromptOutcome:
        return AcpPromptOutcome(stop_reason="end_turn", usage=None)

    async def force_close(self) -> None:
        await super().force_close()
        await self._sink.put(1, running_tool_call("late", "Late"))


class PrePromptClient(StreamingAcpClient):
    """Streams a tool-call start during ``open_session``; ``prompt`` must never run."""

    async def open_session(self) -> Any:
        handshake = await super().open_session()
        await self._sink.put(1, running_tool_call("pre", "Pre"))
        return handshake

    async def prompt(self, _prompt: str) -> AcpPromptOutcome:
        raise AssertionError("prompt must not run — cancelled first")


async def cancel_before_prompt(_attempt: int, _handshake: Any, _translator: Any) -> None:
    """An ``attempt_callback`` that cancels the attempt between session/new and the prompt."""
    raise asyncio.CancelledError


def make_controller(bus: EventBus, tmp_path: Path, **overrides: Any) -> SubAgentToolShell:
    """A controller over ``bus`` with a BYPASS broker and a stub spec under ``tmp_path``.

    ``overrides`` replace or extend the constructor arguments.
    """
    kwargs: dict[str, Any] = {
        "invocation_id": "inv",
        "tool_name": "external",
        "agent_name": "External",
        "prompt": "work",
        "spec_factory": lambda _attempt: make_spec(tmp_path),
        "event_bus": bus,
    }
    if "broker" not in overrides:
        kwargs["broker"] = make_broker(bus, [ApprovalMode.BYPASS])
    kwargs.update(overrides)
    kwargs.setdefault(
        "origin", InvocationOrigin("sub_agent", kwargs.get("session_id") or "", kwargs["invocation_id"], None)
    )
    origin = kwargs.pop("origin")
    kwargs.pop("invocation_id")
    kwargs.pop("session_id", None)
    shell = SubAgentToolShell(
        origin=origin,
        tool_name=kwargs.pop("tool_name"),
        agent_name=kwargs.pop("agent_name"),
        event_bus=kwargs.pop("event_bus"),
    )
    commit = kwargs.pop("parent_interrupted_result_commit", None)
    if commit is not None:
        shell.bind_parent_interrupt_commit(commit)
    shell.attach_policy(AcpSubAgentPolicy(shell=shell, **kwargs))
    return shell

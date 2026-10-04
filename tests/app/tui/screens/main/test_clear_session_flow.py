# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the /clear flow: confirm, then delete the current session and start fresh."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chrys.app.tui.screens.main.navigation import MainNavigationController
from chrys.app.tui.screens.main.state import (
    MainScreenServices,
    MainScreenState,
    RunState,
    SessionViewState,
    SubmitCoordinator,
)
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import Error, SessionClear, SessionDeleted
from chrys.foundation.i18n import MessageRef
from chrys.foundation.i18n.formatting import format_message
from chrys.service.state.store import JsonFileStateStore
from tests.support.tui_helpers import make_backend_handler, make_session_handler

_SESSION_ID = "4201eebc-ca45-4328-8882-272f3d7c41cb"


class _FakeNavigationView:
    def __init__(self, session_id: str = _SESSION_ID) -> None:
        self.session_id = session_id
        self.dialogs: list[dict[str, Any]] = []
        self.notifications: list[tuple[str, str, str]] = []

    def current_session_id(self) -> str:
        return self.session_id

    def open_confirm_dialog(self, **kwargs: Any) -> None:
        self.dialogs.append(kwargs)

    def notify(
        self,
        message: MessageRef | str,
        *,
        title: MessageRef | str,
        severity: str = "information",
        timeout: float | None = 3,
    ) -> None:
        self.notifications.append((_render(message), _render(title), severity))


def _render(value: MessageRef | str) -> str:
    return value if isinstance(value, str) else format_message(value)


class _Harness:
    """Navigation controller wired to a fake view; workers are collected, not scheduled."""

    def __init__(
        self,
        *,
        session_id: str = _SESSION_ID,
        agent_running: bool = False,
        agent_loading: bool = False,
        submit_pending: bool = False,
        has_messages: bool = True,
        state_store: JsonFileStateStore | None = None,
    ) -> None:
        self.view = _FakeNavigationView(session_id)
        self.submit_pending = submit_pending
        self.deleted: list[str] = []
        self.workers: list[Callable[[], Awaitable[object]]] = []

        async def _delete_current_and_new(session_id: str) -> None:
            self.deleted.append(session_id)

        async def _unused(_session_id: str) -> None:
            return None

        async def _flush() -> None:
            return None

        def _start_worker(work: Callable[[], Awaitable[object]]) -> object:
            self.workers.append(work)
            return work

        self.navigation = MainNavigationController(
            services=MainScreenServices(bus=EventBus(), state_store=state_store),
            view=self.view,  # type: ignore[arg-type]
            is_agent_loading=lambda: agent_loading,
            is_agent_running=lambda: agent_running,
            is_submit_pending=lambda: self.submit_pending,
            is_workflow_mode=lambda: False,
            has_messages=lambda: has_messages,
            is_dashboard_visible=lambda: False,
            set_interrupt_confirm_active=lambda _active: None,
            publish_interrupt=lambda: None,
            dismiss_suggestions=lambda: False,
            cancel_pending_injection=lambda: False,
            delete_current_and_new=_delete_current_and_new,
            restore_session=_unused,
            flush_notifications=_flush,
            start_worker=_start_worker,
            debug=lambda _key, _msg: None,
        )

    async def run_workers(self) -> None:
        while self.workers:
            await self.workers.pop(0)()

    def confirm(self, result: bool) -> None:
        self.view.dialogs[-1]["on_result"](result)


def test_clear_opens_confirmation_and_deletes_nothing_until_confirmed() -> None:
    harness = _Harness()

    harness.navigation.clear_session()

    assert len(harness.view.dialogs) == 1
    dialog = harness.view.dialogs[0]
    assert _render(dialog["title"]) == "Clear Session"
    assert _render(dialog["confirm_label"]) == "Delete"
    assert dialog["confirm_variant"] == "error"
    message = _render(dialog["message"])
    assert '"4201eebcca45"' in message
    assert "cannot be recovered" in message
    assert harness.workers == []
    assert harness.deleted == []


@pytest.mark.asyncio
async def test_clear_confirmed_deletes_current_session_and_starts_new() -> None:
    harness = _Harness()

    harness.navigation.clear_session()
    harness.confirm(True)
    await harness.run_workers()

    assert harness.deleted == [_SESSION_ID]


def test_clear_cancelled_keeps_current_session() -> None:
    harness = _Harness()

    harness.navigation.clear_session()
    harness.confirm(False)

    assert harness.workers == []
    assert harness.deleted == []


@pytest.mark.parametrize(("agent_running", "agent_loading"), [(True, False), (False, True)])
def test_clear_is_ignored_while_agent_running_or_loading(agent_running: bool, agent_loading: bool) -> None:
    harness = _Harness(agent_running=agent_running, agent_loading=agent_loading)

    harness.navigation.clear_session()

    assert harness.view.dialogs == []
    assert harness.workers == []
    assert harness.deleted == []


def test_clear_is_ignored_while_a_submit_is_pending() -> None:
    """agent_running flips only after admission; a prompt still being prepared blocks /clear."""
    harness = _Harness(submit_pending=True)

    harness.navigation.clear_session()

    assert harness.view.dialogs == []
    assert harness.workers == []
    assert harness.deleted == []


def test_clear_confirmed_while_a_submit_is_pending_does_not_delete() -> None:
    """A submit that started while the dialog was up must veto the confirmed delete."""
    harness = _Harness()

    harness.navigation.clear_session()
    harness.submit_pending = True
    harness.confirm(True)

    assert harness.workers == []
    assert harness.deleted == []


def test_clear_without_active_session_warns_and_never_publishes_delete() -> None:
    """An empty session id addresses the sessions root; it must never reach SessionDelete."""
    harness = _Harness(session_id="")

    harness.navigation.clear_session()

    assert harness.view.dialogs == []
    assert harness.workers == []
    assert harness.deleted == []
    assert harness.view.notifications == [("No active session to clear", "Clear Session", "warning")]


def test_clear_on_empty_session_warns_without_confirmation(tmp_path: Path) -> None:
    """A never-used session has nothing to delete; /clear warns instead of prompting (mirrors /fork)."""
    harness = _Harness(has_messages=False, state_store=JsonFileStateStore(tmp_path))

    harness.navigation.clear_session()

    assert harness.view.dialogs == []
    assert harness.workers == []
    assert harness.deleted == []
    assert harness.view.notifications == [
        ("Nothing to clear: the current session is empty", "Clear Session", "warning")
    ]


def test_clear_on_empty_chat_with_session_files_still_prompts(tmp_path: Path) -> None:
    """Empty transcript but files on disk (rollback-to-welcome keeps sub-agent artifacts): /clear proceeds."""
    store = JsonFileStateStore(tmp_path)
    artifact = store.session_dir(_SESSION_ID) / "sub_agents" / "child.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")
    harness = _Harness(has_messages=False, state_store=store)

    harness.navigation.clear_session()

    assert harness.view.notifications == []
    assert len(harness.view.dialogs) == 1


def test_clear_confirmed_after_session_changed_does_not_delete() -> None:
    harness = _Harness()

    harness.navigation.clear_session()
    harness.view.session_id = "other-session"
    harness.confirm(True)

    assert harness.workers == []
    assert harness.deleted == []


def _make_clear_screen(bus: EventBus, loading: list[bool], creating: list[bool]) -> SimpleNamespace:
    return SimpleNamespace(
        _state=MainScreenState(run=RunState(has_messages=True)),
        _services=MainScreenServices(bus=bus),
        notify=lambda *_args, **_kwargs: None,
        _set_agent_loading=loading.append,
        _set_creating_new_session=creating.append,
        _debug=lambda *_args: None,
    )


def test_delete_current_and_new_publishes_session_clear_with_input_blocked() -> None:
    """/clear is ONE backend op: input is blocked and the new-session flag set before it is published."""
    bus = EventBus()
    loading: list[bool] = []
    creating: list[bool] = []
    seen: list[tuple[str, bool, bool]] = []
    screen = _make_clear_screen(bus, loading, creating)

    async def fake_backend(event: SessionClear) -> None:
        seen.append((event.session_id, screen._state.run.agent_loading, screen._state.session.creating_new_session))
        await bus.publish(SessionDeleted(session_id=event.session_id))

    async def run() -> None:
        await bus.subscribe(SessionClear, fake_backend)
        await make_session_handler(screen).delete_current_and_new("session-1")

    asyncio.run(run())

    assert seen == [("session-1", True, True)]
    # Acknowledged delete: the fresh session's own load/ready events own the
    # flags from here, so nothing is reset optimistically.
    assert loading == [True]
    assert creating == [True]
    assert screen._state.session.creating_new_session is True


def test_delete_current_and_new_without_ack_resets_new_session_state() -> None:
    """No ``SessionDeleted`` acknowledgement means nothing was deleted: undo the optimistic UI state."""
    bus = EventBus()
    loading: list[bool] = []
    creating: list[bool] = []
    screen = _make_clear_screen(bus, loading, creating)

    async def failing_backend(event: SessionClear) -> None:
        await bus.publish(Error(code="session_clear_failed", message="Failed to delete session: boom"))

    async def run() -> None:
        await bus.subscribe(SessionClear, failing_backend)
        await make_session_handler(screen).delete_current_and_new("session-1")

    asyncio.run(run())

    assert loading == [True, False]
    assert creating == [True, False]
    assert screen._state.session.creating_new_session is False


@pytest.mark.parametrize(
    ("running", "loading", "submitting"),
    [
        pytest.param(True, False, False, id="agent-running"),
        pytest.param(False, True, False, id="agent-loading"),
        pytest.param(False, False, True, id="submit-being-admitted"),
    ],
)
def test_delete_current_and_new_is_ignored_while_busy(running: bool, loading: bool, submitting: bool) -> None:
    """Running, loading, or a submit still being admitted all veto the worker (re-checked, not just preflight)."""
    published: list[object] = []
    loading_calls: list[bool] = []

    async def publish(event: object) -> None:
        published.append(event)

    screen = SimpleNamespace(
        _state=MainScreenState(
            run=RunState(agent_running=running, agent_loading=loading, has_messages=True),
            submit=SubmitCoordinator(active=submitting),
        ),
        _services=MainScreenServices(bus=SimpleNamespace(publish=publish)),
        _set_agent_loading=loading_calls.append,
    )
    handler = make_session_handler(screen)

    asyncio.run(handler.delete_current_and_new("session-1"))

    assert handler.creating_new_session is False
    assert published == []
    assert loading_calls == []


def test_session_clear_error_toasts_and_keeps_session_without_retry_mode() -> None:
    """``session_clear_failed`` is a toast + state reset, never the generic chat error / retry mode."""
    flashes: list[tuple[str, bool]] = []
    notifications: list[tuple[str, str, str]] = []
    unlocked: list[None] = []
    running: list[bool] = []
    loading: list[bool] = []
    creating: list[bool] = []
    errors_added: list[str] = []

    class _FakeStatusBar:
        def flash(self, message: str, *, error: bool = False) -> None:
            flashes.append((message, error))

    class _FakeInputBar:
        locked = True
        retry_mode = False

        def unlock_and_keep(self) -> None:
            self.locked = False
            unlocked.append(None)

    class _FakeChatPanel:
        async def add_error(self, message: str) -> None:
            errors_added.append(message)

    input_bar = _FakeInputBar()

    def query_one(cls: type) -> object:
        if cls.__name__ == "StatusBar":
            return _FakeStatusBar()
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "ChatPanel":
            return _FakeChatPanel()
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    def notify(message: str, *, title: str, severity: str, **_kwargs: object) -> None:
        notifications.append((title, severity, message))

    state = MainScreenState(run=RunState(agent_loading=True), session=SessionViewState(creating_new_session=True))
    screen = SimpleNamespace(
        _state=state,
        _set_agent_running=running.append,
        _set_agent_loading=loading.append,
        _set_creating_new_session=creating.append,
        query_one=query_one,
        notify=notify,
        _debug=lambda *_args: None,
    )
    screen._sessions = make_session_handler(screen)
    handler = make_backend_handler(screen)
    handler._agent_load_dialog = None

    asyncio.run(handler.on_error(Error(code="session_clear_failed", message="Failed to delete session: boom")))

    assert notifications == [("Clear Session", "error", "The current session was kept: Failed to delete session: boom")]
    assert loading == [False]
    assert creating == [False]
    assert state.session.creating_new_session is False
    assert unlocked == [None]
    assert running == []
    assert flashes == []
    assert errors_added == []
    assert input_bar.retry_mode is False

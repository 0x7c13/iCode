# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for InputFlowController retry, interrupt publication and user-message sending."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from types import SimpleNamespace
from typing import Any, cast

import pytest

from chrys.app.tui.screens.main.input_flow import InputFlowController, PreparedSend, SendIntent
from chrys.app.tui.screens.main.ports import InputRestore
from chrys.app.tui.screens.main.screen import MainScreen
from chrys.app.tui.screens.main.state import MainScreenServices, MainScreenState
from chrys.app.tui.support.gc_freeze import (
    GcAbsorbReason,
    GcAbsorbRequested,
)
from chrys.app.tui.widgets.chrome.status_bar import STATUS_RESUMING
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    Error,
    InvocationContentDelta,
    InvocationContentFinished,
    InvocationMessage,
    UserInterrupt,
    UserMessage,
    UserRetry,
    Warning,
)
from chrys.foundation.i18n import MessageRef
from chrys.foundation.models.invocations import InvocationOrigin
from tests.support.tui_helpers import (
    fake_session_title,
    make_backend_handler,
    status_text,
)


class _FakeInputFlowView:
    def __init__(self) -> None:
        self.calls: list[object] = []
        self.user_contents: list[object] = []
        self.gc_messages: list[object] = []
        self.session_id = "panel-session"
        # The composer's draft: text handed back lands only in an empty one.
        self.draft = ""

    def set_terminal_title_for_user_message(self, text: str) -> None:
        self.calls.append(("title", text))

    def clear_terminal_title_result(self) -> None:
        self.calls.append("clear_title_result")

    def current_chat_session_id(self) -> str:
        return self.session_id

    def add_input_history(self, text: str, *, session_id: str | None) -> None:
        self.calls.append(f"history:{text}:{session_id}")

    def set_retry_mode(self, enabled: bool, *, label: str = "Retry") -> None:
        self.calls.append(("retry_mode", enabled, label))

    def start_run_status(self, label: MessageRef | str) -> None:
        self.calls.append(f"status:{status_text(label)}")

    async def render_user_message(self, text: str, *, created_at: object, contents: object) -> None:
        self.calls.append(f"user:{text}")
        self.user_contents.append(contents)

    async def render_user_retry_note(self, text: str, *, created_at: object) -> None:
        self.calls.append(f"retry-note:{text}")

    def hide_trailing_status_action(self) -> None:
        self.calls.append("hide_status_action")

    def set_retry_pending(self, pending: bool) -> None:
        self.calls.append(("retry_pending", pending))

    def clear_inline_questions(self) -> None:
        self.calls.append("clear_questions")

    def clear_ask_user_inline_prompts(self) -> None:
        self.calls.append("clear_inline")

    def lock_input_with_text(self) -> None:
        self.calls.append("lock_input")

    def unlock_input_keep_if_locked(self) -> None:
        self.calls.append("unlock_keep")

    def flash_interrupted(self) -> None:
        self.calls.append("flash_interrupted")

    async def render_interrupted(self) -> None:
        self.calls.append("interrupted")

    def update_toc(self) -> None:
        self.calls.append("toc")

    def restore_input_draft(self, text: str) -> bool:
        self.calls.append(("restore", text))
        if not text or self.draft:
            return False
        self.draft = text
        return True

    def flash_submit_still_sending(self) -> None:
        self.calls.append("still_sending")


def _make_input_flow_controller(
    bus: EventBus,
    *,
    state: MainScreenState | None = None,
    view: _FakeInputFlowView | None = None,
    handle_agent_message: object | None = None,
    handle_agent_content: object | None = None,
    handle_error: object | None = None,
    running: list[bool] | None = None,
    has_messages: list[bool] | None = None,
    start_worker: object | None = None,
) -> tuple[InputFlowController, MainScreenState, _FakeInputFlowView]:
    state = state or MainScreenState()
    view = view or _FakeInputFlowView()
    running = running if running is not None else []
    has_messages = has_messages if has_messages is not None else []

    async def _default_handle_agent_message(event: InvocationMessage) -> None:
        view.calls.append(f"agent:{event.text}")

    async def _default_handle_agent_content(event: InvocationContentDelta | InvocationContentFinished) -> None:
        view.calls.append(f"content:{event.segment_id}")

    async def _default_handle_error(event: Error) -> None:
        view.calls.append(f"error:{event.message}")

    def _unexpected_start_worker(work: object) -> None:
        raise AssertionError(f"unexpected worker scheduling: {work!r}")

    def _set_agent_running(value: bool) -> None:
        state.run.agent_running = value
        running.append(value)

    def _set_has_messages(value: bool) -> None:
        state.run.has_messages = value
        has_messages.append(value)

    controller = InputFlowController(
        state=state,
        services=MainScreenServices(bus=bus),
        view=view,
        start_worker=start_worker or _unexpected_start_worker,
        handle_agent_message=handle_agent_message or _default_handle_agent_message,
        handle_agent_content=handle_agent_content or _default_handle_agent_content,
        handle_error=handle_error or _default_handle_error,
        set_agent_running=_set_agent_running,
        set_has_messages=_set_has_messages,
        clear_workspace_marker=lambda: setattr(state.workspace_marker, "original_cwd", None),
        clear_pending_questions=view.clear_inline_questions,
        commit_profile_switch_marker=lambda: setattr(state.profile_marker, "from_profile", None),
        post_gc_message=view.gc_messages.append,
        debug=lambda *_args: None,
    )
    return controller, state, view


async def _run_started(scheduled: list[Callable[[], Awaitable[object]]]) -> None:
    """Run the work the controller handed its worker hook, as the started workers would.

    Each on a task of its own, as a Textual worker runs, apart from the handler
    that scheduled it.
    """
    while scheduled:
        await asyncio.ensure_future(scheduled.pop(0)())


def test_input_bar_retry_leaves_status_action_until_admission() -> None:
    from chrys.app.tui.widgets.chrome.input_bar import InputBar

    calls: list[object] = []

    class _FakeChatPanel:
        def hide_trailing_status_action(self) -> None:
            calls.append("hide_status_action")

    def query_one(cls: type) -> object:
        if cls.__name__ == "ChatPanel":
            return _FakeChatPanel()
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    flow = SimpleNamespace(request_retry=lambda text: calls.append(("retry", text)))
    screen = SimpleNamespace(
        query_one=query_one,
        _input_flow=flow,
        # No prompt shortcut card on this draft: the retry goes out as typed.
        _prompt_card=SimpleNamespace(submit=lambda _text, *, intent="message": False),
        _route_shortcut_draft=lambda _text, *, intent="message": False,
        _model_unconfigured=lambda: False,
        _workflow=SimpleNamespace(workflow_mode=False),
        _suggestions=SimpleNamespace(suggestion_mode=None),
        _services=MainScreenServices(bus=EventBus()),
    )

    MainScreen._on_retry_requested(screen, InputBar.RetryRequested("please continue"))

    assert calls == [("retry", "please continue")]


def test_inline_status_retry_resumes_without_consuming_input_draft() -> None:
    from chrys.app.tui.widgets.chat.messages import ConversationStatusAction
    from chrys.app.tui.widgets.chrome.input_bar import InputBar

    calls: list[object] = []

    class _FakeInputBar:
        retry_mode = True

        def consume_retry_text(self) -> str:
            calls.append("consume_retry_text")
            return "typed note"

    def query_one(cls: type) -> object:
        if cls is InputBar:
            return _FakeInputBar()
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    flow = SimpleNamespace(request_retry=lambda text: calls.append(("retry", text)))
    screen = SimpleNamespace(
        _state=MainScreenState(),
        query_one=query_one,
        _input_flow=flow,
        # No prompt shortcut card on this draft: the retry goes out as typed.
        _prompt_card=SimpleNamespace(submit=lambda _text, *, intent="message": False),
        _route_shortcut_draft=lambda _text, *, intent="message": False,
        _model_unconfigured=lambda: False,
    )

    MainScreen._on_conversation_status_action_pressed(screen, ConversationStatusAction.Pressed())

    # The inline card resumes plainly: the typed draft is neither consumed
    # nor sent as the continuation prompt (that is the input bar's button).
    assert calls == [("retry", "")]


def test_request_retry_ignores_duplicate_while_retry_submit_is_pending() -> None:
    state = MainScreenState()
    state.submit.active = True
    controller, _state, view = _make_input_flow_controller(EventBus(), state=state)

    controller.request_retry("duplicate note")

    # Nothing is scheduled; the typed note goes back to the composer, which says a send is still in flight.
    assert view.calls == [("restore", "duplicate note"), "still_sending"]
    assert view.draft == "duplicate note"


def test_request_retry_without_a_note_while_a_submit_is_pending_changes_nothing() -> None:
    state = MainScreenState()
    state.submit.active = True
    controller, _state, view = _make_input_flow_controller(EventBus(), state=state)

    controller.request_retry("")

    assert view.calls == []


@pytest.mark.parametrize("outcome", ["accepted", "rejected", "cancelled"])
async def test_retry_reserves_before_worker_start_and_releases_after_admission(outcome: str) -> None:
    bus = EventBus()
    scheduled = []
    controller, state, view = _make_input_flow_controller(bus, start_worker=scheduled.append)
    entered = asyncio.Event()
    release = asyncio.Event()
    published = []

    async def admit(event: UserRetry) -> None:
        published.append(event.text)
        entered.set()
        await release.wait()
        state.submit.blocked = outcome == "rejected"

    await bus.subscribe(UserRetry, admit)
    controller.request_retry("first note")
    controller.request_retry("same-tick duplicate")
    assert state.submit.active
    assert state.submit.text == "first note"
    # The same-tick duplicate is refused: its note goes back to the composer.
    assert view.calls == [("retry_pending", True), ("restore", "same-tick duplicate"), "still_sending"]
    assert len(scheduled) == 1
    task = asyncio.create_task(scheduled.pop()())
    try:
        async with asyncio.timeout(5):
            await entered.wait()
            controller.request_retry("duplicate click")
            assert not scheduled
            assert published == ["first note"]
            if outcome == "cancelled":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                release.set()
                await task
    finally:
        release.set()
        task.cancel()
        async with asyncio.timeout(5):
            with suppress(asyncio.CancelledError):
                await task

    assert not state.submit.active
    assert not state.render_gate.active
    assert view.calls.count(("retry_pending", False)) == 1
    assert state.run.agent_running is (outcome == "accepted")
    if outcome == "accepted":
        assert "hide_status_action" in view.calls
        controller.request_retry("already running")
        assert not scheduled
    else:
        assert "hide_status_action" not in view.calls
        # A later intentional attempt is available again after rejection/cancel.
        controller.request_retry("try again")
        await _run_started(scheduled)
        assert published == ["first note", "try again"]


@pytest.mark.parametrize(
    "error_code",
    [
        "executor_error",
        "retry_missing_user_anchor",
        "hook_blocked",
        "not_ready",
        "prompt_admission_conflict",
        "sub_agent_paused",
    ],
)
@pytest.mark.parametrize("text", ["", "https://example.com/document"], ids=["plain", "with-note"])
@pytest.mark.parametrize("initial_retry_mode", [False, True], ids=["ordinary-input", "continue-input"])
def test_retry_distinguishes_admission_rejections_from_fast_run_errors(
    error_code: str, text: str, initial_retry_mode: bool
) -> None:
    bus = EventBus()
    state = MainScreenState()
    order: list[str] = []

    class _FlowView(_FakeInputFlowView):
        def start_run_status(self, label: MessageRef | str) -> None:
            super().start_run_status(label)
            order.append(f"status:{status_text(label)}")

        def hide_trailing_status_action(self) -> None:
            order.append("hide_status_action")

    class _FakeStatusBar:
        def flash(self, message: str, *, error: bool = False) -> None:
            order.append(f"flash:{message}:{error}")

    class _FakeInputBar:
        def __init__(self) -> None:
            self.locked = True
            self._retry_label = "Continue" if initial_retry_mode else ""
            self.retry_mode = initial_retry_mode
            self.value = ""

        def restore_draft(self, text: str, *, keep_card: bool = False) -> bool:
            self.value = text
            return True

        def unlock_and_keep(self) -> None:
            self.locked = False
            order.append("unlock")

    class _FakeChatPanel:
        async def add_error(self, message: str, *, action_label: str | None = "Retry") -> None:
            order.append(f"error:{message}:{action_label}")

    def query_one(cls: object) -> object:
        if cls.__name__ == "StatusBar":
            return _FakeStatusBar()
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "ChatPanel":
            return _FakeChatPanel()
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    input_bar = _FakeInputBar()
    view = _FlowView()
    screen = SimpleNamespace(
        _state=state,
        _session_title=fake_session_title(),
        query_one=query_one,
        _debug=lambda *_args: None,
        notify=lambda message, **kwargs: order.append(
            f"notify:{status_text(kwargs['title'])}:{status_text(message)}:{kwargs['severity']}"
        ),
    )
    handler = make_backend_handler(screen)
    handler._agent_load_dialog = None
    scheduled: list[Callable[[], Awaitable[object]]] = []
    controller, _state, _view = _make_input_flow_controller(
        bus,
        state=state,
        view=view,
        handle_error=handler.on_error,
        start_worker=scheduled.append,
    )

    async def _fail_retry(_event: UserRetry) -> None:
        event_type = Warning if error_code == "sub_agent_paused" else Error
        await bus.publish(event_type(code=error_code, message="retry failed"))

    async def _run() -> None:
        await bus.subscribe(UserRetry, _fail_retry)
        await bus.subscribe(Error, handler.on_error)
        await bus.subscribe(Warning, handler.on_warning)
        controller.request_retry(text)
        await _run_started(scheduled)

    asyncio.run(_run())

    assert state.run.agent_running is False
    assert state.render_gate.active is False
    if error_code != "executor_error":
        severity = "warning" if error_code == "sub_agent_paused" else "error"
        title = "Warning" if error_code == "sub_agent_paused" else "Retry not started"
        assert f"notify:{title}:retry failed:{severity}" in order
        assert not any(item.startswith("error:") for item in order)
        assert "hide_status_action" not in order
        assert "status:Resuming" not in order
        assert f"retry-note:{text}" not in view.calls
        assert input_bar.value == text
        assert input_bar.locked is False
        assert input_bar.retry_mode is initial_retry_mode
        assert input_bar._retry_label == ("Continue" if initial_retry_mode else "")
    else:
        assert order.index("hide_status_action") < order.index("status:Resuming")
        assert order.index("status:Resuming") < order.index("error:retry failed:Retry")
        assert state.submit.blocked is False
        assert input_bar.retry_mode is True
        if text:
            assert f"retry-note:{text}" in view.calls


@pytest.mark.parametrize("status", ["error", "interrupted"])
@pytest.mark.parametrize("rejection", ["retry_missing_user_anchor", "sub_agent_paused"])
@pytest.mark.parametrize("context_changed", [False, True], ids=["trailing-status", "trailing-system-messages"])
async def test_rejected_retry_keeps_mounted_action_until_next_retry_is_accepted(
    status: str, rejection: str, context_changed: bool
) -> None:
    from textual.widgets import Button

    from chrys.app.tui.screens.main.view_adapter import MainScreenViewAdapter
    from chrys.app.tui.widgets.chat.messages import ErrorMessage, InterruptedMessage
    from chrys.app.tui.widgets.chat.panel import ChatPanel
    from tests.support.tui_helpers import ChatPanelApp

    state = MainScreenState()
    bus = EventBus()
    notifications: list[str] = []

    class Input:
        retry_mode = True
        _retry_label = "Continue"
        value = ""
        locked = True

        def restore_draft(self, text: str, *, keep_card: bool = False) -> bool:
            self.value = text
            return True

        def unlock_and_keep(self) -> None:
            self.locked = False

    input_bar = Input()
    async with ChatPanelApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        if status == "error":
            await panel.add_error("Previous run failed")
            card = panel.query_one(ErrorMessage)
        else:
            await panel.add_interrupted()
            card = panel.query_one(InterruptedMessage)
        if context_changed:
            await panel.add_system("Workspace changed: /old → /new")
            await panel.add_system("Agent profile switched: Code → QA")
        await pilot.pause()
        row = card.query_one(".status-action-row")
        button = card.query_one(Button)

        class View(_FakeInputFlowView):
            def hide_trailing_status_action(self) -> None:
                panel.hide_trailing_status_action()

            def set_retry_pending(self, pending: bool) -> None:
                MainScreenViewAdapter(screen, state=MainScreenState()).set_retry_pending(pending)

        def query_one(cls):
            if cls is ChatPanel:
                return panel
            if cls.__name__ == "InputBar":
                return input_bar
            if cls.__name__ == "StatusBar":
                return SimpleNamespace(flash=lambda *_args, **_kwargs: None)
            raise AssertionError(f"Unexpected widget: {cls}")

        screen = SimpleNamespace(
            _state=state,
            query_one=query_one,
            notify=lambda message, **_kwargs: notifications.append(status_text(message)),
            _debug=lambda *_args: None,
        )
        handler = make_backend_handler(screen)
        scheduled: list[Callable[[], Awaitable[object]]] = []
        controller, _, _ = _make_input_flow_controller(
            bus, state=state, view=View(), handle_error=handler.on_error, start_worker=scheduled.append
        )
        attempts = 0

        async def retry(_event: UserRetry) -> None:
            nonlocal attempts
            attempts += 1
            assert button.disabled and row.display
            assert input_bar.retry_pending
            if attempts <= 2:
                event_type = Warning if rejection == "sub_agent_paused" else Error
                await bus.publish(event_type(code=rejection, message="Cannot start yet"), raise_handler_errors=True)

        await bus.subscribe(UserRetry, retry)
        await bus.subscribe(Error, handler.on_error)
        await bus.subscribe(Warning, handler.on_warning)
        for attempt in range(1, 3):
            controller.request_retry("https://example.com/document")
            await _run_started(scheduled)
            assert notifications == ["Cannot start yet"] * attempt
            assert card.is_mounted and row.display
            assert not button.disabled and not input_bar.retry_pending
            assert input_bar.value == "https://example.com/document"
            assert not state.run.agent_running
            assert not state.submit.active and state.submit.kind == "turn"

        controller.request_retry("https://example.com/document")
        await _run_started(scheduled)
        assert attempts == 3
        assert card.is_mounted and not row.display
        assert not input_bar.retry_pending
        assert state.run.agent_running
        assert not state.submit.active and state.submit.kind == "turn"


def test_publish_interrupt_publishes_before_marking_agent_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sleep ToolCallResult handlers must still see the screen as running."""
    from textual import _time, events

    bus = EventBus()
    observed_running: list[bool] = []
    order: list[str] = []
    input_events: list[events.Key] = []

    class _RunningStates(list[bool]):
        def append(self, value: bool) -> None:
            order.append("idle")
            super().append(value)

    class _GcMessages(list[object]):
        def append(self, message: object) -> None:
            order.append("gc")
            super().append(message)

    class _OrderedView(_FakeInputFlowView):
        def clear_terminal_title_result(self) -> None:
            order.append("clear-title")
            super().clear_terminal_title_result()

        async def render_interrupted(self) -> None:
            assert state.run.agent_running is True
            input_events.append(events.Key("x", "x"))
            order.append("render")
            await super().render_interrupted()

    running = _RunningStates()
    state = MainScreenState()
    state.run.agent_running = True
    view = _OrderedView()
    view.gc_messages = _GcMessages()
    controller, state, view = _make_input_flow_controller(bus, state=state, view=view, running=running)

    async def _on_interrupt(_event: UserInterrupt) -> None:
        observed_running.append(state.run.agent_running)

    async def _run() -> None:
        await bus.subscribe(UserInterrupt, _on_interrupt)
        await controller.publish_interrupt()

    source_times = iter([10.0, 20.0])
    monkeypatch.setattr(_time, "get_time", lambda: next(source_times))
    asyncio.run(_run())

    assert observed_running == [True]
    assert "interrupted" in view.calls
    assert "clear_questions" in view.calls
    assert running == [False]
    assert state.run.agent_running is False
    assert order == ["render", "clear-title", "idle", "gc"]
    assert "clear_title_result" in view.calls
    assert len(view.gc_messages) == 1
    assert isinstance(view.gc_messages[0], GcAbsorbRequested)
    assert view.gc_messages[0].reason is GcAbsorbReason.TURN_TERMINAL
    assert view.gc_messages[0].terminal_boundary is True
    assert view.gc_messages[0].time == 10.0
    assert input_events[0].time == 20.0


def test_publish_interrupt_reasserts_flash_after_idle_gate_closes() -> None:
    """A teardown event that flips the status bar back to run mode while the
    interrupt awaits (``agent_running`` still True) must be overwritten by a
    second interrupted flash after the gate closes."""
    bus = EventBus()
    order: list[str] = []

    class _RunningStates(list[bool]):
        def append(self, value: bool) -> None:
            order.append("idle")
            super().append(value)

    class _RaceView(_FakeInputFlowView):
        def flash_interrupted(self) -> None:
            order.append("flash")
            super().flash_interrupted()

        async def render_interrupted(self) -> None:
            # Simulate a CompactionFinished/late ToolCallResult handler racing
            # the teardown window and re-showing the run-mode status while
            # ``agent_running`` is still True.
            order.append("raced-show-status")
            await super().render_interrupted()

    state = MainScreenState()
    state.run.agent_running = True
    controller, state, view = _make_input_flow_controller(bus, state=state, view=_RaceView(), running=_RunningStates())

    asyncio.run(controller.publish_interrupt())

    assert order == ["flash", "raced-show-status", "idle", "flash"]
    assert "clear_title_result" in view.calls
    assert view.calls.count("flash_interrupted") == 2


def test_publish_interrupt_render_failure_releases_turn_without_absorb() -> None:
    bus = EventBus()
    running: list[bool] = []

    class _FailingView(_FakeInputFlowView):
        async def render_interrupted(self) -> None:
            self.calls.append("interrupted")
            raise RuntimeError("render failed")

    state = MainScreenState()
    state.run.agent_running = True
    state.pending_injection.begin("pending-id", "queued")
    view = _FailingView()
    controller, state, view = _make_input_flow_controller(bus, state=state, view=view, running=running)

    with pytest.raises(RuntimeError, match="render failed"):
        asyncio.run(controller.publish_interrupt())

    assert running == [False]
    assert state.run.agent_running is False
    assert state.pending_injection.active is False
    assert "unlock_keep" in view.calls
    assert "clear_title_result" in view.calls
    assert view.gc_messages == []
    assert ("retry_mode", True, "Continue") not in view.calls


def test_send_user_message_real_bus_rejection_sets_pending_blocked_before_render() -> None:
    bus = EventBus()
    running: list[bool] = []
    published: list[str] = []
    controller, state, view = _make_input_flow_controller(bus, running=running)

    async def _reject_image_prompt(event: UserMessage) -> None:
        published.append(event.text)
        state.submit.block()

    async def _run() -> None:
        await bus.subscribe(UserMessage, _reject_image_prompt)
        await controller.send_user_message("describe @shot.png")

    asyncio.run(_run())

    assert published == ["describe @shot.png"]
    assert state.submit.active is False
    assert state.submit.blocked is False
    assert state.render_gate.active is False
    assert running == []
    assert "user:describe @shot.png" not in view.calls


def test_send_user_message_defers_fast_agent_message_until_user_bubble_is_rendered() -> None:
    bus = EventBus()
    state = MainScreenState()
    view = _FakeInputFlowView()

    async def _on_agent_message(event: InvocationMessage) -> None:
        if state.render_gate.defers(event):
            state.render_gate.defer(event)
            return
        view.calls.append(f"agent:{event.text}")

    async def _on_agent_content(event: InvocationContentDelta | InvocationContentFinished) -> None:
        if state.render_gate.defers(event):
            state.render_gate.defer(event)
            return
        view.calls.append(f"delta:{event.text}" if isinstance(event, InvocationContentDelta) else "finished")

    controller, _state, view = _make_input_flow_controller(
        bus,
        state=state,
        view=view,
        handle_agent_message=_on_agent_message,
        handle_agent_content=_on_agent_content,
    )

    async def _accept_and_finish(event: UserMessage) -> None:
        event.prepared_contents = ["hello", "prepared-image-content"]
        event.started_turn = True
        origin = InvocationOrigin("turn", event.session_id or "", "turn-test", None)
        await bus.publish(
            InvocationContentDelta(
                origin=origin, session_id=event.session_id, segment_id="s1", channel="answer", text="done"
            )
        )
        await bus.publish(
            InvocationContentFinished(origin=origin, session_id=event.session_id, segment_id="s1", channel="answer")
        )
        await bus.publish(
            InvocationMessage(text=f"done:{event.text}", is_final=True, session_id=event.session_id, origin=origin)
        )

    async def _run() -> None:
        await bus.subscribe(UserMessage, _accept_and_finish)
        await bus.subscribe(InvocationContentDelta, _on_agent_content)
        await bus.subscribe(InvocationContentFinished, _on_agent_content)
        await bus.subscribe(InvocationMessage, _on_agent_message)
        await controller.send_user_message("hello")

    asyncio.run(_run())

    # Streamed text and the final both wait for the user bubble, then replay in order.
    user_index = view.calls.index("user:hello")
    assert view.calls[user_index + 1 : user_index + 4] == ["delta:done", "finished", "agent:done:hello"]
    assert view.user_contents == [["hello", "prepared-image-content"]]


def test_text_arriving_while_deferred_text_replays_waits_behind_it() -> None:
    """A replayed delta can await a mount; a live one arriving meanwhile must not overtake the backlog."""
    bus = EventBus()
    state = MainScreenState()
    view = _FakeInputFlowView()
    origin = InvocationOrigin("turn", "", "turn-test", None)

    def _delta(text: str) -> InvocationContentDelta:
        return InvocationContentDelta(origin=origin, segment_id="s1", channel="answer", text=text)

    live = _delta("C")

    async def _on_agent_content(event: InvocationContentDelta | InvocationContentFinished) -> None:
        if state.render_gate.defers(event):
            state.render_gate.defer(event)
            return
        if isinstance(event, InvocationContentDelta):
            view.calls.append(f"delta:{event.text}")
            if event.text == "A":
                # The engine streams on while this delta's mount is awaited.
                await bus.publish(live)

    controller, _state, view = _make_input_flow_controller(
        bus, state=state, view=view, handle_agent_content=_on_agent_content
    )

    async def _accept_and_stream(event: UserMessage) -> None:
        event.started_turn = True
        await bus.publish(_delta("A"))
        await bus.publish(_delta("B"))

    async def _run() -> None:
        await bus.subscribe(UserMessage, _accept_and_stream)
        await bus.subscribe(InvocationContentDelta, _on_agent_content)
        await controller.send_user_message("hello")

    asyncio.run(_run())

    assert [call for call in view.calls if str(call).startswith("delta:")] == ["delta:A", "delta:B", "delta:C"]
    assert state.render_gate.active is False


def test_send_user_message_defers_fast_error_until_user_bubble_is_rendered() -> None:
    bus = EventBus()
    state = MainScreenState()
    order: list[str] = []

    class _FlowView(_FakeInputFlowView):
        async def render_user_message(self, text: str, *, created_at: object, contents: object) -> None:
            await super().render_user_message(text, created_at=created_at, contents=contents)
            order.append(f"user:{text}")

    class _FakeStatusBar:
        def flash(self, message: str, *, error: bool = False) -> None:
            order.append(f"status:{message}:{error}")

    class _FakeInputBar:
        def __init__(self) -> None:
            self.locked = True
            self._retry_label = ""
            self.retry_mode = False

        def unlock_and_keep(self) -> None:
            self.locked = False
            order.append("unlock")

    class _FakeChatPanel:
        async def add_error(self, message: str, *, action_label: str | None = "Retry") -> None:
            order.append(f"error:{message}:{action_label}")

    def query_one(cls: object) -> object:
        if cls.__name__ == "StatusBar":
            return _FakeStatusBar()
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "ChatPanel":
            return _FakeChatPanel()
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    input_bar = _FakeInputBar()
    view = _FlowView()
    screen = SimpleNamespace(
        _state=state,
        _session_title=fake_session_title(),
        query_one=query_one,
        _debug=lambda *_args: None,
    )
    handler = make_backend_handler(screen)
    handler._agent_load_dialog = None

    controller, _state, view = _make_input_flow_controller(
        bus,
        state=state,
        view=view,
        handle_error=handler.on_error,
    )

    async def _accept_and_fail(event: UserMessage) -> None:
        event.prepared_contents = ["hello", "prepared-image-content"]
        event.started_turn = True
        await bus.publish(Error(code="executor_error", message=f"failed:{event.text}", session_id=event.session_id))

    async def _run() -> None:
        await bus.subscribe(UserMessage, _accept_and_fail)
        await bus.subscribe(Error, handler.on_error)
        await controller.send_user_message("hello")

    asyncio.run(_run())

    user_index = order.index("user:hello")
    error_index = order.index("error:failed:hello:Retry")
    assert user_index < error_index
    assert view.user_contents == [["hello", "prepared-image-content"]]
    assert state.submit.blocked is False
    assert input_bar.retry_mode is True


# -- prepared sends: one claim from Enter to publish ---------------------------

_PREPARED_DRAFT = "check !`git status`"
_PREPARED_FINAL = "check clean"


class _Recorded:
    def __init__(self) -> None:
        self.events: list[UserMessage | UserRetry] = []
        self.in_publish: list[tuple[bool, str, str]] = []


async def _record_publishes(bus: EventBus, state: MainScreenState, *, start_turn: bool = True) -> _Recorded:
    recorded = _Recorded()

    async def record(event: UserMessage | UserRetry) -> None:
        recorded.events.append(event)
        # Who the frontend would attribute a refusal published now to.
        recorded.in_publish.append((state.submit.in_publish(), state.submit.text, state.submit.kind))
        if isinstance(event, UserMessage) and start_turn and event.injection_id is None:
            event.started_turn = True

    await bus.subscribe(UserMessage, record)
    await bus.subscribe(UserRetry, record)
    return recorded


class _Send:
    """A prepared send's own give-back and validity check, as the card builds them.

    The give-back hands the draft to the composer at most once; *valid* is
    what the send's check answers when asked.
    """

    def __init__(self, view: _FakeInputFlowView, text: str = _PREPARED_DRAFT) -> None:
        self._view = view
        self.text = text
        self.valid = True
        self.give_backs = 0
        self.validity_checks = 0

    def give_back(self) -> InputRestore | None:
        # As the card's give-back: the proof of its restore the first time, None after or when a draft wins.
        self.give_backs += 1
        if self.give_backs > 1 or not self._view.restore_input_draft(self.text):
            return None
        return InputRestore(generation=self.give_backs, text=self.text)

    def still_valid(self) -> bool:
        self.validity_checks += 1
        return self.valid


def _begin(controller: InputFlowController, send: _Send, intent: SendIntent = "message") -> PreparedSend | None:
    return controller.begin_prepared(send.text, intent, give_back=send.give_back, still_valid=send.still_valid)


@pytest.mark.parametrize("intent", ["message", "continue"])
async def test_begin_prepared_claims_at_once_and_opens_nothing_else(intent: SendIntent) -> None:
    controller, state, view = _make_input_flow_controller(EventBus())
    send = _Send(view)

    prepared = _begin(controller, send, intent)

    assert prepared is not None
    assert (prepared.text, prepared.intent) == (_PREPARED_DRAFT, intent)
    assert controller.holds(prepared)
    assert state.submit.holds(prepared.claim)
    assert state.submit.text == _PREPARED_DRAFT
    assert state.submit.kind == ("retry" if intent == "continue" else "turn")
    # The send's own give-back and check are the claim's.
    assert state.submit.give_back == send.give_back
    assert state.submit.still_valid == send.still_valid
    # No render gate, running or retry-pending state until it publishes; nothing given back or checked yet.
    assert not state.render_gate.active
    assert not state.run.agent_running
    assert view.calls == []
    assert (send.give_backs, send.validity_checks) == (0, 0)


@pytest.mark.parametrize("intent", ["message", "continue"])
async def test_begin_prepared_is_refused_while_another_send_holds_the_claim(intent: SendIntent) -> None:
    controller, state, view = _make_input_flow_controller(EventBus())
    first_send = _Send(view, "first")
    first = _begin(controller, first_send)
    assert first is not None
    second = _Send(view)

    assert _begin(controller, second, intent) is None

    # Given back through its own give-back, and the composer says a send is in flight; the first keeps its claim.
    assert second.give_backs == 1
    assert view.calls == [("restore", _PREPARED_DRAFT), "still_sending"]
    assert controller.holds(first)
    assert state.submit.text == "first"
    assert state.submit.give_back == first_send.give_back
    assert first_send.give_backs == 0


async def test_refused_begin_prepared_says_nothing_when_its_text_did_not_come_back() -> None:
    controller, _state, view = _make_input_flow_controller(EventBus())
    first = _begin(controller, _Send(view, "first"))
    assert first is not None
    view.draft = "typed meanwhile"
    second = _Send(view)

    assert _begin(controller, second) is None

    assert second.give_backs == 1
    assert "still_sending" not in view.calls
    assert view.draft == "typed meanwhile"
    assert controller.holds(first)


async def test_begin_prepared_continue_is_refused_while_a_turn_runs_and_a_message_is_not() -> None:
    state = MainScreenState()
    state.run.agent_running = True
    controller, state, view = _make_input_flow_controller(EventBus(), state=state)
    send = _Send(view)

    assert _begin(controller, send, "continue") is None
    assert send.give_backs == 1
    assert view.calls == [("restore", _PREPARED_DRAFT), "still_sending"]
    assert not state.submit.active

    view.draft = ""
    message = _begin(controller, _Send(view, "note for the turn"))
    assert message is not None
    assert controller.holds(message)


async def test_publish_prepared_message_opens_a_turn_with_the_final_text_under_the_same_claim() -> None:
    bus = EventBus()
    scheduled: list[Callable[[], Awaitable[object]]] = []
    controller, state, view = _make_input_flow_controller(bus, start_worker=scheduled.append)
    recorded = await _record_publishes(bus, state)
    send = _Send(view)
    prepared = _begin(controller, send)
    assert prepared is not None
    claims_before = state.submit.generation

    assert controller.publish_prepared(prepared, _PREPARED_FINAL) is True
    # Nothing is published, and nothing checked, until the worker runs.
    assert recorded.events == []
    assert send.validity_checks == 0
    await _run_started(scheduled)

    assert [(type(event).__name__, event.text) for event in recorded.events] == [("UserMessage", _PREPARED_FINAL)]
    # Published under the claim taken at Enter, attributed to it, holding the draft as written.
    assert state.submit.generation == claims_before
    assert recorded.in_publish == [(True, _PREPARED_DRAFT, "turn")]
    assert send.validity_checks == 1
    assert f"user:{_PREPARED_FINAL}" in view.calls
    assert state.run.agent_running
    assert not state.submit.active
    assert send.give_backs == 0


async def test_publish_prepared_message_while_a_turn_runs_is_an_injection() -> None:
    bus = EventBus()
    scheduled: list[Callable[[], Awaitable[object]]] = []
    controller, state, view = _make_input_flow_controller(bus, start_worker=scheduled.append)
    recorded = await _record_publishes(bus, state)
    prepared = _begin(controller, _Send(view))
    assert prepared is not None

    assert controller.publish_prepared(prepared, _PREPARED_FINAL) is True
    # A turn started after the publish was scheduled: decided when the worker publishes.
    state.run.agent_running = True
    await _run_started(scheduled)

    assert len(recorded.events) == 1
    event = recorded.events[0]
    assert isinstance(event, UserMessage)
    assert event.text == _PREPARED_FINAL
    assert event.injection_id is not None
    assert recorded.in_publish == [(True, _PREPARED_DRAFT, "injection")]
    assert state.pending_injection.matches(event.injection_id)
    assert "lock_input" in view.calls
    assert not any(isinstance(call, str) and call.startswith("user:") for call in view.calls)
    assert not state.submit.active


async def test_publish_prepared_continue_shows_retry_pending_then_the_resumed_run() -> None:
    bus = EventBus()
    scheduled: list[Callable[[], Awaitable[object]]] = []
    controller, state, view = _make_input_flow_controller(bus, start_worker=scheduled.append)
    recorded = await _record_publishes(bus, state)
    prepared = _begin(controller, _Send(view), "continue")
    assert prepared is not None

    assert controller.publish_prepared(prepared, _PREPARED_FINAL) is True
    # The retry-pending state and the render gate open in the worker, right before it publishes.
    assert view.calls == []
    assert not state.render_gate.active
    await _run_started(scheduled)

    assert [(type(event).__name__, event.text) for event in recorded.events] == [("UserRetry", _PREPARED_FINAL)]
    assert recorded.in_publish == [(True, _PREPARED_DRAFT, "retry")]
    assert view.calls[0] == ("retry_pending", True)
    assert view.calls.count(("retry_pending", False)) == 1
    assert "hide_status_action" in view.calls
    assert f"status:{status_text(STATUS_RESUMING.bind())}" in view.calls
    assert f"retry-note:{_PREPARED_FINAL}" in view.calls
    assert state.run.agent_running
    assert not state.submit.active
    assert not state.render_gate.active


@pytest.mark.parametrize("intent", ["message", "continue"])
@pytest.mark.parametrize("stale", ["no_longer_valid", "agent_loading", "claim_lost"])
async def test_publish_worker_checks_again_and_gives_the_draft_back_once(intent: SendIntent, stale: str) -> None:
    bus = EventBus()
    scheduled: list[Callable[[], Awaitable[object]]] = []
    controller, state, view = _make_input_flow_controller(bus, start_worker=scheduled.append)
    recorded = await _record_publishes(bus, state)
    send = _Send(view)
    prepared = _begin(controller, send, intent)
    assert prepared is not None
    assert controller.publish_prepared(prepared, _PREPARED_FINAL) is True
    newer: int | None = None

    # What changes between the scheduling and the worker's run.
    if stale == "no_longer_valid":
        send.valid = False
    elif stale == "agent_loading":
        state.run.agent_loading = True
    else:
        state.submit.clear()
        newer = state.submit.begin("someone else's", kind="turn")
    await _run_started(scheduled)

    assert recorded.events == []
    assert not state.render_gate.active
    assert ("retry_pending", True) not in view.calls
    assert not state.run.agent_running
    if newer is None:
        # Its own claim ends, and its draft comes back through its give-back, once.
        assert not state.submit.active
        assert send.give_backs == 1
        assert view.calls == [("restore", _PREPARED_DRAFT)]
    else:
        # Not its claim any more: it neither ends the newer one nor gives anything back.
        assert state.submit.holds(newer)
        assert send.give_backs == 0
        assert view.calls == []


async def test_publish_prepared_when_the_claim_is_not_its_own_schedules_nothing() -> None:
    bus = EventBus()
    scheduled: list[Callable[[], Awaitable[object]]] = []
    controller, state, view = _make_input_flow_controller(bus, start_worker=scheduled.append)
    send = _Send(view)
    prepared = _begin(controller, send)
    assert prepared is not None
    state.submit.clear()

    assert controller.publish_prepared(prepared, _PREPARED_FINAL) is False

    assert scheduled == []
    assert send.give_backs == 0


async def test_abandon_prepared_releases_its_claim_and_returns_what_its_give_back_says() -> None:
    controller, state, view = _make_input_flow_controller(EventBus())
    send = _Send(view)
    prepared = _begin(controller, send)
    assert prepared is not None

    assert controller.abandon_prepared(prepared) is True

    assert not state.submit.active
    assert state.submit.give_back is None
    assert send.give_backs == 1
    assert view.calls == [("restore", _PREPARED_DRAFT)]
    assert view.draft == _PREPARED_DRAFT


async def test_abandon_prepared_keeps_a_draft_typed_meanwhile() -> None:
    controller, state, view = _make_input_flow_controller(EventBus())
    send = _Send(view)
    prepared = _begin(controller, send)
    assert prepared is not None
    view.draft = "typed meanwhile"

    assert controller.abandon_prepared(prepared) is False

    assert send.give_backs == 1
    assert view.draft == "typed meanwhile"
    # The claim is released all the same: a later send is not stuck behind it.
    assert not state.submit.active
    assert _begin(controller, _Send(view, "next")) is not None


async def test_an_old_token_never_touches_a_newer_claim() -> None:
    bus = EventBus()
    scheduled: list[Callable[[], Awaitable[object]]] = []
    controller, state, view = _make_input_flow_controller(bus, start_worker=scheduled.append)
    recorded = await _record_publishes(bus, state)
    old_send = _Send(view, "old draft")
    old = _begin(controller, old_send)
    assert old is not None
    assert controller.abandon_prepared(old) is True
    view.draft = ""
    view.calls.clear()
    newer_send = _Send(view, "newer draft")
    newer = _begin(controller, newer_send)
    assert newer is not None

    # The old send's late cleanup and late publish both find the claim is not theirs.
    assert controller.abandon_prepared(old) is False
    assert controller.publish_prepared(old, "old final") is False
    assert not controller.holds(old)

    assert scheduled == []
    assert recorded.events == []
    assert view.calls == []
    assert (old_send.give_backs, newer_send.give_backs) == (1, 0)
    assert controller.holds(newer)
    assert state.submit.text == "newer draft"

    assert controller.publish_prepared(newer, "newer final") is True
    await _run_started(scheduled)
    assert [event.text for event in recorded.events] == ["newer final"]


async def test_a_plain_send_cannot_take_the_claim_from_a_prepared_one() -> None:
    bus = EventBus()
    scheduled: list[Callable[[], Awaitable[object]]] = []
    controller, state, view = _make_input_flow_controller(bus, start_worker=scheduled.append)
    recorded = await _record_publishes(bus, state)
    send = _Send(view)
    prepared = _begin(controller, send)
    assert prepared is not None

    controller.submit_user_text("typed while it prepares")
    controller.request_retry("a continue note")
    await _run_started(scheduled)

    assert recorded.events == []
    assert controller.holds(prepared)
    # The continue note, refused at once, is back in the composer; the message's worker, refused later, finds
    # that draft there and leaves it.
    assert view.draft == "a continue note"
    assert ("restore", "typed while it prepares") in view.calls
    assert view.calls.count("still_sending") == 1
    assert send.give_backs == 0

    assert controller.publish_prepared(prepared, _PREPARED_FINAL) is True
    await _run_started(scheduled)
    assert [event.text for event in recorded.events] == [_PREPARED_FINAL]


async def test_a_prepared_message_that_opens_no_turn_comes_back_through_its_give_back() -> None:
    bus = EventBus()
    scheduled: list[Callable[[], Awaitable[object]]] = []
    controller, state, view = _make_input_flow_controller(bus, start_worker=scheduled.append)
    recorded = await _record_publishes(bus, state, start_turn=False)
    send = _Send(view)
    prepared = _begin(controller, send)
    assert prepared is not None

    assert controller.publish_prepared(prepared, _PREPARED_FINAL) is True
    await _run_started(scheduled)

    assert [event.text for event in recorded.events] == [_PREPARED_FINAL]
    # The draft as written comes back through the send's own give-back, not the published text.
    assert send.give_backs == 1
    assert view.calls == [("restore", _PREPARED_DRAFT)]
    assert not state.submit.active
    assert not state.render_gate.active
    assert not state.run.agent_running


# -- plain sends release only their own claim ----------------------------------


class _TakeoverBus:
    """A bus whose publish hands the claim to a newer send meanwhile, then accepts, refuses or raises."""

    def __init__(self, state: MainScreenState, outcome: str) -> None:
        self._state = state
        self._outcome = outcome
        self.newer: int | None = None
        self.published: list[object] = []

    async def publish(self, event: object) -> None:
        self.published.append(event)
        # The submit in flight was let go meanwhile, and a newer send took the claim.
        self._state.submit.clear()
        self.newer = self._state.submit.begin("newer", kind="turn")
        if self._outcome == "refused":
            self._state.submit.block()
        elif self._outcome == "raised":
            raise RuntimeError("publish failed")
        elif isinstance(event, UserMessage) and event.injection_id is None:
            event.started_turn = True


@pytest.mark.parametrize("outcome", ["accepted", "refused", "raised"])
@pytest.mark.parametrize("entry", ["message", "retry", "injection"])
async def test_a_plain_sends_cleanup_never_ends_a_newer_claim(entry: str, outcome: str) -> None:
    state = MainScreenState()
    state.run.agent_running = entry == "injection"
    bus = _TakeoverBus(state, outcome)
    scheduled: list[Callable[[], Awaitable[object]]] = []
    controller = _make_input_flow_controller(cast(Any, bus), state=state, start_worker=scheduled.append)[0]

    if entry == "retry":
        controller.request_retry("a note")
    else:
        controller.submit_user_text("hello")
    assert len(scheduled) == 1
    if outcome == "raised":
        with pytest.raises(RuntimeError, match="publish failed"):
            await _run_started(scheduled)
    else:
        await _run_started(scheduled)

    assert len(bus.published) == 1
    assert bus.newer is not None
    assert state.submit.holds(bus.newer)
    assert state.submit.text == "newer"

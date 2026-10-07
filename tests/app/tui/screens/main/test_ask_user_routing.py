# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for ask-user routing: dialog hosting, timeouts, queueing and inline fallback."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

import chrys.app.tui.screens.dialogs.ask_user as _ask_user_mod
from chrys.app.tui.screens.main.event_handlers import (
    BackendEventHandler,
)
from chrys.app.tui.screens.main.state import MainScreenState, RunState
from chrys.app.tui.screens.main.view_adapter import MainScreenViewAdapter
from chrys.app.tui.widgets import PromptDraft
from chrys.foundation.events.types import (
    AskUserTimedOut,
    QuestionToUser,
)
from chrys.foundation.i18n import MessageRef
from chrys.foundation.models.ask_user import AskUserAnswer, AskUserOption, AskUserQuestion
from chrys.foundation.models.invocations import InvocationOrigin
from tests.support.tui_helpers import (
    FakeNotificationService,
    make_backend_handler,
    status_text,
)


class _FakeAskUserDialog:
    def __init__(
        self,
        request_id: str,
        questions: tuple[AskUserQuestion, ...],
        caller_name: str = "",
        draft: PromptDraft | None = None,
        allow_inline: bool = True,
    ) -> None:
        self.request_id = request_id
        self.questions = questions
        self.caller_name = caller_name
        self.draft = draft
        self.allow_inline = allow_inline
        self.dismissed = False
        self.callback = None

    def dismiss_due_to_timeout(self) -> None:
        self.dismissed = True
        if self.callback is not None:
            self.callback(None)

    def submit(self, text: str) -> None:
        self.dismissed = True
        if self.callback is not None:
            self.callback((self.request_id, (AskUserAnswer(values=(text,)),)))

    def answer_inline(self, draft_text: str = "") -> None:
        from chrys.app.tui.screens.dialogs.ask_user import AskUserInlineResult

        self.dismissed = True
        if self.callback is not None:
            self.callback(AskUserInlineResult(self.request_id, PromptDraft(drafts=(draft_text,))))


class _FakeAskUserApp:
    def __init__(self) -> None:
        self.notification_service = FakeNotificationService()
        self.pushed: list[tuple[_FakeAskUserDialog, object]] = []

    def push_screen(self, screen: _FakeAskUserDialog, callback) -> None:
        screen.callback = callback
        self.pushed.append((screen, callback))


def _ask_user_event(
    request_id: str,
    question: str,
    *,
    call_id: str = "",
    options: tuple[str, ...] = (),
) -> QuestionToUser:
    return QuestionToUser(
        request_id=request_id,
        call_id=call_id,
        questions=(
            AskUserQuestion(
                question=question,
                options=tuple(AskUserOption(label=label) for label in options),
            ),
        ),
    )


@pytest.mark.parametrize(("call_id", "allow_inline"), [("call-1", True), ("", False)])
def test_question_dialog_offers_inline_for_chat_calls_before_card_mount(
    monkeypatch: pytest.MonkeyPatch, call_id: str, allow_inline: bool
) -> None:
    monkeypatch.setattr(_ask_user_mod, "AskUserDialog", _FakeAskUserDialog)
    app = _FakeAskUserApp()
    screen = SimpleNamespace(app=app)
    adapter = MainScreenViewAdapter(screen, state=MainScreenState())  # type: ignore[arg-type]

    # External ACP questions carry no call id, so no card can take the
    # hand-off; the modal must not offer a button that only reopens itself.
    adapter.show_question_dialog(_ask_user_event("req-1", "Proceed?", call_id=call_id), None, lambda _result: None)

    dialog, _callback = app.pushed[0]
    assert dialog.allow_inline is allow_inline


def _make_ask_user_handler(
    monkeypatch,
) -> tuple[BackendEventHandler, _FakeAskUserApp, list[tuple[str, tuple[AskUserAnswer, ...]]], list]:
    monkeypatch.setattr(_ask_user_mod, "AskUserDialog", _FakeAskUserDialog)

    app = _FakeAskUserApp()
    responses: list[tuple[str, tuple[AskUserAnswer, ...]]] = []
    debug_log: list[tuple[str, str]] = []

    screen = SimpleNamespace(
        app=app,
        _handle_ask_user_response=lambda request_id, answers: responses.append((request_id, answers)),
        _debug=lambda event_type, detail="": debug_log.append((event_type, detail)),
    )
    handler = make_backend_handler(screen)
    handler._question_queue = deque()
    handler._question_dialog_open = False
    handler._open_question_dialogs = {}
    handler._inline_question_call_ids = {}
    handler._inline_question_request_ids = {}
    handler._question_drafts = {}
    return handler, app, responses, debug_log


def test_ask_user_timeout_dismisses_live_dialog(monkeypatch) -> None:
    handler, app, responses, debug_log = _make_ask_user_handler(monkeypatch)

    asyncio.run(handler.on_question_to_user(_ask_user_event("q1", "Need input?")))
    dialog, _callback = app.pushed[0]

    asyncio.run(handler.on_ask_user_timed_out(AskUserTimedOut(request_id="q1")))

    assert dialog.dismissed is True
    assert responses == []
    assert handler._open_question_dialogs == {}
    assert handler._question_dialog_open is False
    assert debug_log[-1] == ("AskUserTimedOut", "q1")


def test_ask_user_timeout_removes_queued_dialog(monkeypatch) -> None:
    handler, app, responses, _debug_log = _make_ask_user_handler(monkeypatch)

    asyncio.run(handler.on_question_to_user(_ask_user_event("q1", "First?")))
    asyncio.run(handler.on_question_to_user(_ask_user_event("q2", "Second?")))
    assert len(app.pushed) == 1

    asyncio.run(handler.on_ask_user_timed_out(AskUserTimedOut(request_id="q2")))
    first_dialog, _callback = app.pushed[0]
    first_dialog.submit("answer")

    assert responses == [("q1", (AskUserAnswer(values=("answer",)),))]
    assert len(app.pushed) == 1
    assert handler._question_dialog_open is False


def test_ask_user_inline_releases_dialog_slot_for_next_question(monkeypatch) -> None:
    monkeypatch.setattr(_ask_user_mod, "AskUserDialog", _FakeAskUserDialog)

    class _FakeChatPanel:
        def __init__(self) -> None:
            self.inline_calls: list[tuple[str, str, tuple[AskUserQuestion, ...], PromptDraft | None]] = []

        def is_tool_running(self, call_id: str) -> bool:
            return call_id in {"c1", "c2"}

        def show_ask_user_inline(
            self,
            call_id: str,
            request_id: str,
            questions: tuple[AskUserQuestion, ...],
            *,
            draft: PromptDraft | None = None,
        ) -> bool:
            self.inline_calls.append((call_id, request_id, questions, draft))
            return True

    app = _FakeAskUserApp()
    panel = _FakeChatPanel()
    debug_log: list[tuple[str, str]] = []
    screen = SimpleNamespace(
        app=app,
        _handle_ask_user_response=lambda *_args: None,
        _debug=lambda event_type, detail="": debug_log.append((event_type, detail)),
        query_one=lambda _cls: panel,
    )
    handler = make_backend_handler(screen)
    handler._question_queue = deque()
    handler._question_dialog_open = False
    handler._open_question_dialogs = {}
    handler._inline_question_call_ids = {}
    handler._inline_question_request_ids = {}
    handler._question_drafts = {}

    asyncio.run(handler.on_question_to_user(_ask_user_event("q1", "First?", call_id="c1", options=("A",))))
    asyncio.run(handler.on_question_to_user(_ask_user_event("q2", "Second?", call_id="c2")))
    first_dialog, _callback = app.pushed[0]

    first_dialog.answer_inline("draft")

    expected_questions = _ask_user_event("q1", "First?", call_id="c1", options=("A",)).questions
    assert panel.inline_calls == [("c1", "q1", expected_questions, PromptDraft(drafts=("draft",)))]
    assert handler._inline_question_call_ids == {"q1": "c1"}
    assert handler._inline_question_request_ids == {"c1": "q1"}
    assert len(app.pushed) == 2
    second_dialog, _callback = app.pushed[1]
    assert second_dialog.request_id == "q2"
    assert handler._question_dialog_open is True


def _answer_inline_ask_user_that_cannot_render(
    monkeypatch,
    *,
    tool_running: bool,
) -> tuple[BackendEventHandler, _FakeAskUserApp]:
    """Answer an inline ask_user whose chat panel refuses to re-render the draft.

    ``tool_running`` decides whether the tool that asked is still alive, which
    is what the handler consults before reopening a modal fallback.
    """
    monkeypatch.setattr(_ask_user_mod, "AskUserDialog", _FakeAskUserDialog)

    class _FakeChatPanel:
        def show_ask_user_inline(
            self,
            _call_id: str,
            _request_id: str,
            _questions: tuple[AskUserQuestion, ...],
            *,
            draft: PromptDraft | None = None,
        ) -> bool:
            assert draft == PromptDraft(drafts=("draft",))
            return False

        def is_tool_running(self, call_id: str) -> bool:
            return tool_running and call_id == "c1"

    app = _FakeAskUserApp()
    screen = SimpleNamespace(
        app=app,
        _handle_ask_user_response=lambda *_args: None,
        _debug=lambda *_args: None,
        query_one=lambda _cls: _FakeChatPanel(),
    )
    handler = make_backend_handler(screen)
    handler._question_queue = deque()
    handler._question_dialog_open = False
    handler._open_question_dialogs = {}
    handler._inline_question_call_ids = {}
    handler._inline_question_request_ids = {}
    handler._question_drafts = {}

    asyncio.run(handler.on_question_to_user(_ask_user_event("q1", "First?", call_id="c1")))
    first_dialog, _callback = app.pushed[0]

    first_dialog.answer_inline("draft")
    return handler, app


def test_ask_user_inline_fallback_preserves_draft_when_tool_is_still_running(monkeypatch) -> None:
    _handler, app = _answer_inline_ask_user_that_cannot_render(monkeypatch, tool_running=True)

    assert len(app.pushed) == 2
    fallback_dialog, _callback = app.pushed[1]
    assert fallback_dialog.request_id == "q1"
    assert fallback_dialog.draft == PromptDraft(drafts=("draft",))


def test_ask_user_inline_fallback_does_not_reopen_for_finished_tool(monkeypatch) -> None:
    handler, app = _answer_inline_ask_user_that_cannot_render(monkeypatch, tool_running=False)

    assert len(app.pushed) == 1
    assert handler._question_dialog_open is False
    assert handler._question_queue == deque()
    assert handler._question_drafts == {}


def test_inline_tool_result_restores_input_focus() -> None:
    from chrys.app.tui.widgets.chat.panel import ChatPanel
    from chrys.app.tui.widgets.chrome.input_bar import InputBar
    from chrys.app.tui.widgets.chrome.status_bar import StatusBar
    from chrys.foundation.events.types import InvocationToolCallResult

    calls: list[str] = []

    class _FakeChatPanel:
        session_id = "s1"

        async def add_tool_result(self, *_args, **_kwargs) -> None:
            calls.append("tool_result")

    class _FakeInputBar:
        def focus_input(self) -> None:
            calls.append("focus_input")

    class _FakeStatusBar:
        def show(self, value: MessageRef | str) -> None:
            calls.append(f"status:{status_text(value)}")

    panel = _FakeChatPanel()
    input_bar = _FakeInputBar()
    status_bar = _FakeStatusBar()

    def query_one(cls: type) -> object:
        if cls is ChatPanel:
            return panel
        if cls is InputBar:
            return input_bar
        if cls is StatusBar:
            return status_bar
        raise AssertionError(f"unexpected query_one({cls})")

    screen = SimpleNamespace(
        _state=MainScreenState(run=RunState(agent_running=True)),
        _debug=lambda *_args: None,
        query_one=query_one,
    )
    handler = make_backend_handler(screen)
    handler._inline_question_call_ids = {"q1": "c1"}
    handler._inline_question_request_ids = {"c1": "q1"}
    handler._question_drafts = {"q1": PromptDraft(drafts=("draft",))}
    handler._accumulate_shell_snapshots = lambda _metadata: None

    asyncio.run(
        handler.on_tool_result(
            InvocationToolCallResult(
                call_id="c1",
                tool_name="ask_user",
                result="User response: yes",
                origin=InvocationOrigin("turn", "", "turn-test", None),
            )
        )
    )

    assert calls == ["tool_result", "focus_input", "status:Thinking"]
    assert handler._inline_question_call_ids == {}
    assert handler._inline_question_request_ids == {}
    assert handler._question_drafts == {}


def test_ask_user_timeout_clears_inline_pending_map(monkeypatch) -> None:
    handler, _app, _responses, debug_log = _make_ask_user_handler(monkeypatch)
    handler._inline_question_call_ids = {"q1": "c1"}
    handler._inline_question_request_ids = {"c1": "q1"}
    handler._question_drafts = {"q1": PromptDraft(drafts=("draft",))}

    asyncio.run(handler.on_ask_user_timed_out(AskUserTimedOut(request_id="q1")))

    assert handler._inline_question_call_ids == {}
    assert handler._inline_question_request_ids == {}
    assert handler._question_drafts == {}
    assert debug_log[-1] == ("AskUserTimedOut", "q1 (inline)")


def test_clear_pending_questions_dismisses_dialog_and_clears_queue_and_inline_state(monkeypatch) -> None:
    handler, app, responses, _debug_log = _make_ask_user_handler(monkeypatch)

    asyncio.run(handler.on_question_to_user(_ask_user_event("q1", "First?")))
    asyncio.run(handler.on_question_to_user(_ask_user_event("q2", "Second?")))
    dialog, _callback = app.pushed[0]
    handler._inline_question_call_ids = {"q-inline": "c-inline"}
    handler._inline_question_request_ids = {"c-inline": "q-inline"}
    handler._question_drafts = {"q-inline": PromptDraft(drafts=("draft",))}

    handler.clear_pending_questions()

    assert dialog.dismissed is True
    assert responses == []
    assert list(handler._question_queue) == []
    assert handler._open_question_dialogs == {}
    assert handler._inline_question_call_ids == {}
    assert handler._inline_question_request_ids == {}
    assert handler._question_drafts == {}
    assert handler._question_dialog_open is False
    assert len(app.pushed) == 1


@pytest.mark.parametrize("session_id, expected", [("chat", True), ("workflow", False)])
def test_question_dialog_routes_by_session_without_probing_a_tool_card(
    monkeypatch: pytest.MonkeyPatch, session_id: str, expected: bool
) -> None:
    monkeypatch.setattr(_ask_user_mod, "AskUserDialog", _FakeAskUserDialog)
    app = _FakeAskUserApp()
    adapter = MainScreenViewAdapter(SimpleNamespace(app=app), state=MainScreenState())  # type: ignore[arg-type]
    monkeypatch.setattr(
        adapter, "current_chat_session_id", create_autospec(adapter.current_chat_session_id, return_value="chat")
    )
    probe = create_autospec(adapter.question_can_reopen_modal, side_effect=AssertionError("card probe"))
    monkeypatch.setattr(adapter, "question_can_reopen_modal", probe)
    event = replace(_ask_user_event("question", "Continue?", call_id="call"), session_id=session_id)
    adapter.show_question_dialog(event, None, lambda _result: None)
    assert app.pushed[0][0].allow_inline is expected
    probe.assert_not_called()

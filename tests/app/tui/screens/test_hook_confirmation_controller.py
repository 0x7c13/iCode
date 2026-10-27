# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The TUI's hook confirmation queue: one dialog at a time, one answer per question, none for a withdrawn one."""

from __future__ import annotations

from collections.abc import Callable

from chrys.app.tui.screens.main.dialog_controllers import HookConfirmationQueueController
from chrys.foundation.events.types import HookConfirmationRequest, HookConfirmationWithdrawn
from chrys.foundation.models.hook_confirmations import HookConfirmationAnswer


class _Port:
    def __init__(self) -> None:
        self.shown: list[tuple[str, Callable[[bool | None], None]]] = []
        self.dismissed: list[str] = []
        self.answers: list[tuple[str | None, str, HookConfirmationAnswer]] = []
        self.notified = 0

    def show_hook_confirmation(
        self, event: HookConfirmationRequest, on_result: Callable[[bool | None], None]
    ) -> object:
        self.shown.append((event.request_id, on_result))
        return event.request_id

    def dismiss_hook_confirmation(self, dialog: object) -> None:
        assert isinstance(dialog, str)
        self.dismissed.append(dialog)

    def publish_hook_confirmation_answer(
        self, session_id: str | None, request_id: str, answer: HookConfirmationAnswer
    ) -> None:
        self.answers.append((session_id, request_id, answer))

    def notify_approval_required(self) -> None:
        self.notified += 1

    def close(self, request_id: str, result: bool | None) -> None:
        (on_result,) = [callback for shown, callback in self.shown if shown == request_id]
        on_result(result)


def _request(request_id: str) -> HookConfirmationRequest:
    return HookConfirmationRequest(
        session_id="s1",
        request_id=request_id,
        operation_id=f"op-{request_id}",
        revision=1,
        run_id=f"run-{request_id}",
        hook_id="guard",
        subject="tool",
        tool_name="edit_file",
    )


def test_questions_show_one_at_a_time_and_each_answer_goes_back_once() -> None:
    port = _Port()
    controller = HookConfirmationQueueController(port)
    first, second = _request("r1"), _request("r2")

    controller.on_request(first)
    controller.on_request(second)
    assert [shown for shown, _ in port.shown] == ["r1"]
    assert controller.awaiting_user is True
    port.close("r1", True)
    assert [shown for shown, _ in port.shown] == ["r1", "r2"]
    port.close("r2", False)

    assert port.answers == [("s1", "r1", "approved"), ("s1", "r2", "declined")]
    assert controller.awaiting_user is False
    assert port.notified == 2


def test_a_question_already_claimed_elsewhere_is_left_alone() -> None:
    port = _Port()
    controller = HookConfirmationQueueController(port)
    request = _request("r1")
    assert request.claim.try_claim() is True

    controller.on_request(request)

    assert port.shown == []
    assert controller.awaiting_user is False


def test_a_withdrawn_queued_question_never_shows() -> None:
    port = _Port()
    controller = HookConfirmationQueueController(port)
    controller.on_request(_request("r1"))
    controller.on_request(_request("r2"))

    controller.on_withdrawn(HookConfirmationWithdrawn(session_id="s1", request_id="r2", outcome="invalidated"))
    port.close("r1", True)

    assert [shown for shown, _ in port.shown] == ["r1"]
    assert port.answers == [("s1", "r1", "approved")]
    assert controller.awaiting_user is False


def test_a_withdrawn_open_question_closes_without_an_answer() -> None:
    port = _Port()
    controller = HookConfirmationQueueController(port)
    controller.on_request(_request("r1"))
    controller.on_request(_request("r2"))

    controller.on_withdrawn(HookConfirmationWithdrawn(session_id="s1", request_id="r1", outcome="timed_out"))
    assert port.dismissed == ["r1"]
    # The dialog closes the way a withdrawn one does: with no result.
    port.close("r1", None)

    assert port.answers == []
    assert [shown for shown, _ in port.shown] == ["r1", "r2"]
    port.close("r2", True)
    assert port.answers == [("s1", "r2", "approved")]


def test_a_question_that_fails_to_show_answers_unavailable_and_the_next_shows() -> None:
    class _BrokenPort(_Port):
        def show_hook_confirmation(
            self, event: HookConfirmationRequest, on_result: Callable[[bool | None], None]
        ) -> object:
            if event.request_id == "r1":
                raise RuntimeError("cannot mount")
            return super().show_hook_confirmation(event, on_result)

    port = _BrokenPort()
    controller = HookConfirmationQueueController(port)
    controller.on_request(_request("r1"))
    controller.on_request(_request("r2"))

    assert port.answers == [("s1", "r1", "frontend_unavailable")]
    assert [shown for shown, _ in port.shown] == ["r2"]
    port.close("r2", True)
    assert port.answers[-1] == ("s1", "r2", "approved")

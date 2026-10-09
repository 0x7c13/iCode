# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Human approval deadlines exclude queueing and model review time."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from types import ModuleType

import pytest

from chrys.app.tui.screens.main import dialog_controllers as controller_module
from chrys.app.tui.screens.main.dialog_controllers import ApprovalQueueController
from chrys.foundation.events.types import ApprovalCancelled, ApprovalReviewed
from tests.app.tui.screens.test_dialog_controllers import _approval_request, _ApprovalPort


class _Timer:
    def __init__(self, delay: float, callback: Callable[[str], None], request_id: str) -> None:
        self.delay = delay
        self.callback = callback
        self.request_id = request_id
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        if not self.cancelled:
            self.callback(self.request_id)


class _Scheduler:
    def __init__(self) -> None:
        self.timers: list[_Timer] = []

    def call_later(self, delay: float, callback: Callable[[str], None], request_id: str) -> _Timer:
        timer = _Timer(delay, callback, request_id)
        self.timers.append(timer)
        return timer


@pytest.fixture
def approval_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[ApprovalQueueController, _ApprovalPort, _Scheduler]]:
    scheduler = _Scheduler()
    shadow = ModuleType("asyncio")
    shadow.Lock = asyncio.Lock
    shadow.get_running_loop = lambda: scheduler
    monkeypatch.setattr(controller_module, "asyncio", shadow)
    port = _ApprovalPort()
    controller = ApprovalQueueController(port, timeout_seconds=lambda: 45)
    try:
        yield controller, port, scheduler
    finally:
        controller.close()


async def test_timeout_rejects_once_and_gives_the_next_request_its_own_budget(approval_wait) -> None:
    controller, port, scheduler = approval_wait
    await controller.on_request(_approval_request("first", judging=False))
    await controller.on_request(_approval_request("second", judging=False))
    assert [(timer.request_id, timer.delay) for timer in scheduler.timers] == [("first", 45)]

    scheduler.timers[0].fire()
    assert port.responses == [("first", False, "Human approval request timed out.", None)]
    assert port.cancelled_dialogs == ["first"]
    assert [(timer.request_id, timer.delay) for timer in scheduler.timers] == [("first", 45), ("second", 45)]
    assert set(controller.open_dialogs) == {"second"}
    scheduler.timers[0].fire()
    assert len(port.responses) == 1

    await controller.on_cancelled(ApprovalCancelled(request_id="second"))
    scheduler.timers[1].fire()
    assert len(port.responses) == 1
    assert controller._timeouts == {}


@pytest.mark.parametrize("defer", [False, True])
async def test_judge_time_is_excluded_and_flagged_request_gets_full_budget(approval_wait, defer: bool) -> None:
    controller, port, scheduler = approval_wait
    port.defer_while_judging = defer
    await controller.on_request(_approval_request("reviewing"))
    assert scheduler.timers == []
    assert port.responses == []

    await controller.on_reviewed(ApprovalReviewed(request_id="reviewing", approved=False, reason="Review this"))
    assert [(timer.request_id, timer.delay) for timer in scheduler.timers] == [("reviewing", 45)]
    # Repeated presentation updates do not extend an existing deadline.
    await controller.on_reviewed(ApprovalReviewed(request_id="reviewing", approved=False, reason="Review this"))
    assert len(scheduler.timers) == 1
    scheduler.timers[0].fire()
    assert port.responses == [("reviewing", False, "Human approval request timed out.", None)]


@pytest.mark.parametrize("defer", [False, True])
async def test_auto_approval_never_starts_a_human_timeout(approval_wait, defer: bool) -> None:
    controller, port, scheduler = approval_wait
    port.defer_while_judging = defer
    await controller.on_request(_approval_request("safe"))
    await controller.on_reviewed(ApprovalReviewed(request_id="safe", approved=True))
    assert scheduler.timers == []
    assert port.responses == []


async def test_cached_flag_starts_timing_only_when_its_dialog_is_shown(approval_wait) -> None:
    controller, port, scheduler = approval_wait
    await controller.on_request(_approval_request("first", judging=False))
    await controller.on_request(_approval_request("flagged"))
    await controller.on_reviewed(ApprovalReviewed(request_id="flagged", approved=False, reason="Review"))
    assert len(scheduler.timers) == 1
    port.dialogs[0].user_decision_submitted = True
    port.dialogs[0].callback((True, "", None))
    assert scheduler.timers[0].cancelled
    assert [(timer.request_id, timer.delay) for timer in scheduler.timers] == [("first", 45), ("flagged", 45)]


@pytest.mark.parametrize("approved", [False, True])
async def test_user_response_wins_and_cancels_the_deadline(approval_wait, approved: bool) -> None:
    controller, port, scheduler = approval_wait
    await controller.on_request(_approval_request("human", judging=False))
    port.dialogs[0].user_decision_submitted = True
    port.dialogs[0].callback((approved, "my choice", None))
    assert scheduler.timers[0].cancelled
    scheduler.timers[0].fire()
    assert port.responses == [("human", approved, "my choice", None)]
    assert controller._timeouts == {}


async def test_timeout_does_not_wait_for_a_covered_dialog_to_pop(
    approval_wait, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller, port, scheduler = approval_wait
    await controller.on_request(_approval_request("covered", judging=False))
    dialog = port.dialogs[0]

    def defer_dismissal(handle) -> None:
        handle.is_dismissed = True

    monkeypatch.setattr(port, "dismiss_approval_dialog", defer_dismissal)
    scheduler.timers[0].fire()
    assert port.responses == [("covered", False, "Human approval request timed out.", None)]
    assert "covered" in controller.cancelled_requests
    dialog.callback(None)
    await controller.on_reviewed(ApprovalReviewed(request_id="covered", approved=True))
    assert len(port.responses) == 1
    assert not controller.open_dialogs
    assert not controller.cancelled_requests


async def test_screen_teardown_cancels_pending_timers(approval_wait) -> None:
    controller, port, scheduler = approval_wait
    await controller.on_request(_approval_request("pending", judging=False))
    controller.close()
    assert scheduler.timers[0].cancelled
    scheduler.timers[0].fire()
    assert port.responses == []
    assert controller._timeouts == {}


@pytest.mark.parametrize("approved", [False, True])
async def test_zero_disables_timer_but_allows_user_response(approval_wait, approved: bool) -> None:
    controller, port, scheduler = approval_wait
    controller._timeout_seconds = lambda: 0
    await controller.on_request(_approval_request("unlimited", judging=False))
    assert scheduler.timers == []
    assert port.responses == []
    port.dialogs[0].user_decision_submitted = True
    port.dialogs[0].callback((approved, "human choice", None))
    assert port.responses == [("unlimited", approved, "human choice", None)]


async def test_default_unlimited_wait_can_be_cancelled(approval_wait) -> None:
    _, port, scheduler = approval_wait
    controller = ApprovalQueueController(port)
    await controller.on_request(_approval_request("unlimited", judging=False))
    assert scheduler.timers == []
    await controller.on_cancelled(ApprovalCancelled(request_id="unlimited"))
    assert port.cancelled_dialogs == ["unlimited"]
    assert port.responses == []
    assert not controller.open_dialogs
    controller.close()

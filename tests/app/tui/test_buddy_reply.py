# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The reply flow shared by every TUI surface that can pet the buddy."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from chrys.app.tui.buddy_reply import PetReplyFlow
from chrys.foundation.i18n.formatting import format_message
from tests.support.buddies import HeldPetReply, a_buddy

if TYPE_CHECKING:
    from chrys.foundation.i18n import MessageRef


class _Surface:
    def __init__(self) -> None:
        self.toasts: list[tuple[str, float]] = []
        self.answered = 0

    def toast(self, message: MessageRef | str, timeout: float) -> None:
        self.toasts.append((message if isinstance(message, str) else format_message(message), timeout))

    def on_answered(self) -> None:
        self.answered += 1

    def flow(self) -> PetReplyFlow:
        return PetReplyFlow(self.toast, on_answered=self.on_answered)


pytestmark = pytest.mark.usefixtures("buddy_reply_gate_left_open")


@pytest.mark.asyncio
async def test_a_pet_toasts_a_thinking_line_now_and_the_answer_when_it_comes(monkeypatch: pytest.MonkeyPatch) -> None:
    model = HeldPetReply(monkeypatch, "💛 hoot")
    surface = _Surface()
    flow = surface.flow()

    assert flow.start(a_buddy())

    assert len(surface.toasts) == 1
    assert "Pico" in surface.toasts[0][0]
    assert surface.answered == 0
    task = flow.task
    assert task is not None

    model.go.set()
    await task

    assert surface.toasts[1] == ("💛 hoot", 10)
    assert surface.toasts[0][1] < surface.toasts[1][1]
    assert surface.answered == 1
    assert flow.task is None


@pytest.mark.asyncio
async def test_one_answer_is_written_at_a_time_across_every_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    model = HeldPetReply(monkeypatch, "💛 hoot")
    sidebar, command = _Surface(), _Surface()
    first, second = sidebar.flow(), command.flow()

    assert first.start(a_buddy())
    assert not first.start(a_buddy())
    assert not second.start(a_buddy())
    assert command.toasts == []
    assert len(sidebar.toasts) == 1

    task = first.task
    assert task is not None
    model.go.set()
    await task

    assert second.start(a_buddy())
    await second.shutdown()


@pytest.mark.asyncio
async def test_shutdown_stops_waiting_for_the_model_and_frees_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    model = HeldPetReply(monkeypatch, "never sent")
    surface = _Surface()
    flow = surface.flow()
    assert flow.start(a_buddy())
    task = flow.task
    assert task is not None
    await model.asked.wait()

    await flow.shutdown()

    assert task.cancelled()
    assert flow.task is None
    assert len(surface.toasts) == 1
    assert surface.answered == 0


@pytest.mark.asyncio
async def test_shutdown_before_the_answer_task_ever_ran_still_frees_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A task cancelled before its first step never enters its body, so its own cleanup never runs."""
    HeldPetReply(monkeypatch, "never sent")
    flow = _Surface().flow()

    assert flow.start(a_buddy())
    await flow.shutdown()  # no await between start and here: the task has not had a turn yet


@pytest.mark.asyncio
async def test_an_answer_started_while_shutdown_is_still_waiting_keeps_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = HeldPetReply(monkeypatch, "💛 hoot")
    surface = _Surface()
    flow, other = surface.flow(), _Surface().flow()
    assert flow.start(a_buddy())
    await model.asked.wait()
    old = flow.task
    assert old is not None
    # Runs after the old answer has let go of the gate and before shutdown() gets to continue.
    restarted: list[bool] = []
    old.add_done_callback(lambda _task: restarted.append(flow.start(a_buddy())))

    await flow.shutdown()

    assert restarted == [True]
    assert not other.start(a_buddy())
    new = flow.task
    assert new is not None
    assert new is not old
    model.go.set()
    await new
    assert surface.toasts[-1] == ("💛 hoot", 10)


@pytest.mark.asyncio
async def test_shutdown_leaves_an_answer_another_surface_is_waiting_for_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    model = HeldPetReply(monkeypatch, "💛 hoot")
    busy, idle = _Surface().flow(), _Surface().flow()
    assert busy.start(a_buddy())

    await idle.shutdown()

    assert not idle.start(a_buddy())
    task = busy.task
    assert task is not None
    model.go.set()
    await task


@pytest.mark.asyncio
async def test_a_surface_that_cannot_toast_does_not_wedge_the_gate() -> None:
    def broken_toast(_message: MessageRef | str, _timeout: float) -> None:
        raise RuntimeError("screen is gone")

    flow = PetReplyFlow(broken_toast, on_answered=lambda: None)

    with pytest.raises(RuntimeError, match="screen is gone"):
        flow.start(a_buddy())

    assert flow.task is None


@pytest.mark.asyncio
async def test_a_surface_that_went_away_mid_answer_is_logged_not_raised(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="chrys.app.tui.buddy_reply")
    model = HeldPetReply(monkeypatch, "💛 hoot")
    toasts: list[str] = []

    def toast_once(message: MessageRef | str, _timeout: float) -> None:
        if toasts:
            raise RuntimeError("screen is gone")
        toasts.append(str(message))

    flow = PetReplyFlow(toast_once, on_answered=lambda: None)
    assert flow.start(a_buddy())
    task = flow.task
    assert task is not None

    model.go.set()
    await task

    assert "Buddy answer could not be shown" in caplog.text
    assert flow.task is None

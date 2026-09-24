# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The callback every frontend runs after a successful turn, on its event loop."""

from __future__ import annotations

import logging
import threading
from random import Random

import pytest

from chrys.app.features.buddy import actions, lifecycle
from chrys.app.features.buddy.store import BuddyStore
from tests.support.buddies import HeldSaveFile
from tests.support.waiting import wait_for

_FAILED = "Failed to credit the buddy with a turn"


def _turns() -> int:
    buddy = actions.current_buddy()
    assert buddy is not None
    return buddy.record.turns


@pytest.mark.asyncio
async def test_a_successful_turn_is_credited_to_the_buddy_without_the_turn_waiting_for_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions.hatch(Random(1))
    hold = HeldSaveFile(monkeypatch)

    lifecycle.on_successful_turn()
    lifecycle.on_successful_turn()

    # The credit sits on the save file on a thread of its own: the loop, and so the turn, has gone on.
    await wait_for(hold.entered.is_set, description="the credit has reached the save file")
    assert _turns() == 0
    hold.release()
    await wait_for(lambda: _turns() == 2, description="both turns are credited")


@pytest.mark.asyncio
async def test_a_successful_turn_without_a_buddy_does_nothing_at_all(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="chrys.app.features.buddy")
    credited = threading.Event()
    outcomes: list[BaseException | None] = []
    credit = lifecycle.record_turn

    def credit_and_tell() -> None:
        try:
            credit()
        except BaseException as failure:
            outcomes.append(failure)
            raise
        else:
            outcomes.append(None)
        finally:
            credited.set()

    monkeypatch.setattr(lifecycle, "record_turn", credit_and_tell)

    lifecycle.on_successful_turn()

    await wait_for(credited.is_set, description="the credit has run")
    assert outcomes == [None]
    assert actions.current_buddy() is None
    assert not BuddyStore().path.parent.exists()
    assert caplog.text == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [PermissionError("read-only"), TimeoutError("wedged"), ValueError("bug")])
async def test_a_buddy_failure_never_reaches_the_turn(
    failure: Exception, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="chrys.app.features.buddy.lifecycle")

    def fail() -> None:
        raise failure

    monkeypatch.setattr(lifecycle, "record_turn", fail)

    lifecycle.on_successful_turn()

    await wait_for(lambda: _FAILED in caplog.text, description="the failure is logged, and that is all")
    assert f"{type(failure).__name__}: {failure}" in caplog.text


def test_without_an_event_loop_the_turn_is_not_credited_and_that_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="chrys.app.features.buddy.lifecycle")
    actions.hatch(Random(1))

    lifecycle.on_successful_turn()

    assert _FAILED in caplog.text
    assert "no running event loop" in caplog.text
    assert _turns() == 0

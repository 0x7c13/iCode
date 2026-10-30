# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for main-screen submit lifecycle state."""

from __future__ import annotations

import asyncio

import pytest

from chrys.app.tui.screens.main.ports import InputRestore
from chrys.app.tui.screens.main.submit_state import SubmitState


def test_submit_state_begin_sets_active_text_and_clears_blocked() -> None:
    state = SubmitState(blocked=True)

    state.begin("describe @shot.png")

    assert state.active is True
    assert state.text == "describe @shot.png"
    assert state.blocked is False


def test_submit_state_block_records_backend_rejection() -> None:
    state = SubmitState(active=True, text="blocked prompt")

    state.block()

    assert state.active is True
    assert state.text == "blocked prompt"
    assert state.blocked is True


def test_submit_state_clear_resets_publish_phase() -> None:
    state = SubmitState(active=True, text="blocked prompt", blocked=True)

    state.clear()

    assert state == SubmitState()


@pytest.mark.asyncio
async def test_submit_state_is_in_publish_only_on_its_publishing_task() -> None:
    """The backend refuses a submit inside its publish; another task's errors are someone else's."""
    state = SubmitState()

    async def in_publish() -> bool:
        return state.in_publish()

    # A retry is claimed in its click handler, then published by its worker.
    state.begin("hello", kind="retry")
    assert state.in_publish() is False

    async def worker() -> bool:
        state.publish_from_current_task()
        return state.in_publish()

    assert await asyncio.create_task(worker()) is True
    assert state.in_publish() is False

    state.publish_from_current_task()
    assert state.in_publish() is True
    assert await asyncio.create_task(in_publish()) is False

    state.clear()
    assert state.in_publish() is False


def test_each_claim_gets_a_new_number_and_only_the_latest_holds() -> None:
    state = SubmitState()

    first = state.begin("first")
    assert state.holds(first) is True

    second = state.begin("second", kind="retry")

    assert second != first
    assert state.holds(second) is True
    assert state.holds(first) is False


def test_release_ends_only_the_claim_still_in_flight() -> None:
    state = SubmitState()
    first = state.begin("first")
    second = state.begin("second")

    # The earlier claim's late cleanup leaves the newer one alone.
    assert state.release(first) is False
    assert state.active is True
    assert state.text == "second"
    assert state.holds(second) is True

    assert state.release(second) is True
    assert state.active is False
    assert state.holds(second) is False
    # Releasing twice does nothing more.
    assert state.release(second) is False


def test_a_number_never_comes_back_after_clear() -> None:
    state = SubmitState()
    first = state.begin("first")
    state.clear()

    second = state.begin("second")

    assert second != first
    assert state.holds(first) is False
    assert state.release(first) is False
    assert state.holds(second) is True


def test_claim_keeps_its_give_back_and_check_until_it_ends() -> None:
    state = SubmitState()

    def give_back() -> InputRestore | None:
        return InputRestore(generation=1, text="prepared")

    def still_valid() -> bool:
        return True

    claim = state.begin("prepared", give_back=give_back, still_valid=still_valid)
    assert state.give_back is give_back
    assert state.still_valid is still_valid

    # A plain claim after it carries none of the earlier send's.
    plain = state.begin("plain")
    assert state.give_back is None
    assert state.still_valid is None

    state.begin("prepared again", give_back=give_back, still_valid=still_valid)
    state.block()
    state.clear()
    assert state == SubmitState(generation=state.generation)
    assert state.give_back is None
    assert state.still_valid is None
    assert state.holds(claim) is False
    assert state.holds(plain) is False

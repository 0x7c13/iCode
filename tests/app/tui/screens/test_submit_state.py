# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for main-screen submit lifecycle state."""

from __future__ import annotations

import asyncio

import pytest

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

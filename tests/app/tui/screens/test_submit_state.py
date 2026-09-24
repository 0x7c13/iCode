# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for main-screen submit lifecycle state."""

from __future__ import annotations

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

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for persisted Chrys history marker constants."""

from __future__ import annotations

from chrys.foundation.models.history_markers import HistoryMarkerKind


def test_history_marker_kind_values_are_stable_session_wire_strings() -> None:
    assert HistoryMarkerKind.KEY == "_chrys_kind"
    assert HistoryMarkerKind.PROFILE_SWITCH_TO_KEY == "_switch_to"
    assert HistoryMarkerKind.TURN == "turn"
    assert HistoryMarkerKind.SUMMARY == "summary"
    assert HistoryMarkerKind.INTERRUPTED == "interrupted"
    assert HistoryMarkerKind.AWAITING_SUB_AGENTS == "awaiting_sub_agents"


def test_history_marker_sets_cover_status_and_visibility_rules() -> None:
    assert (
        frozenset({HistoryMarkerKind.INTERRUPTED, HistoryMarkerKind.AWAITING_SUB_AGENTS})
        == HistoryMarkerKind.STATUS_MARKERS
    )
    assert (
        frozenset({HistoryMarkerKind.TURN, HistoryMarkerKind.INTERRUPTED, HistoryMarkerKind.AWAITING_SUB_AGENTS})
        == HistoryMarkerKind.SESSION_COUNT_EXCLUDED
    )
    assert (
        frozenset({HistoryMarkerKind.TURN, HistoryMarkerKind.SUMMARY, HistoryMarkerKind.INTERRUPTED})
        == HistoryMarkerKind.LAST_WORDS_EXCLUDED
    )

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tool-result fan-in proof through context-revision membership, Phase-4 bridging, and carrier mappings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chrys.foundation.trajectory.envelope import Actor, SegmentedField, measurement
from chrys.foundation.trajectory.event_types import EventType
from chrys.foundation.trajectory.metadata import ANALYTICS_ITEM_ID_KEY
from chrys.service.analytics import (
    Precision,
    TrajectoryAnalyzer,
    analyze_trajectory,
)
from chrys.service.analytics._context_evidence import _replay_delta
from chrys.service.analytics._facts import _RevisionEntry
from tests.service.analytics._events import NS, EventLog, caused_by, operation_index


def test_serial_model_cycle_fans_in_from_prior_tool_subtree(tmp_path) -> None:
    call_item_id = "9" * 32
    result_item_id = "a" * 32
    revision_id = "6" * 32
    segment_group_id = "5" * 32
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 20 * NS, links=caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        10 * NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "d" * 32},
    )
    log.span("model.exchange", "d" * 32, 0, 2 * NS, parent_operation_id="c" * 32)
    log.span(
        "preparation",
        "e" * 32,
        2 * NS,
        2 * NS,
        parent_operation_id="c" * 32,
        start_payload={"scope": "tool_preamble", "phase": "dispatch", "target_operation_id": "f" * 32},
    )
    log.span(
        "tool.operation",
        "f" * 32,
        2 * NS,
        10 * NS,
        parent_operation_id="c" * 32,
        start_payload={
            "tool_name": "read",
            "tool_kind": "filesystem.read",
            "batch_index": 0,
            "parent_model_operation_id": "d" * 32,
            "call_item_id": call_item_id,
        },
        finish_payload={"result_item_id": result_item_id},
        links=caused_by("e" * 32),
    )
    revision = log.add(
        "context.revision.recorded",
        10 * NS,
        operation_id=revision_id,
        parent_operation_id="8" * 32,
        payload={"revision_id": revision_id, "is_checkpoint": True, "item_count": 2, "unidentified_item_count": 0},
        segmented_fields=(
            SegmentedField(
                field_pointer="/payload/refs",
                segment_group_id=segment_group_id,
                segment_count=1,
            ),
        ),
    )
    log.add(
        "event.segment",
        10 * NS,
        operation_id=None,
        payload={
            "parent_event_id": revision.event_id,
            "field_pointer": "/payload/refs",
            "segment_group_id": segment_group_id,
            "segment_index": 0,
            "segment_count": 1,
            "encoding": "array_slice",
            "entries": [
                {"item_id": call_item_id, "occurrence": 0, "position": 0, "action": "add"},
                {"item_id": result_item_id, "occurrence": 0, "position": 1, "action": "add"},
            ],
        },
    )
    log.span(
        "model.cycle",
        "7" * 32,
        10 * NS,
        20 * NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "8" * 32},
    )
    log.span(
        "model.exchange",
        "8" * 32,
        10 * NS,
        20 * NS,
        parent_operation_id="7" * 32,
        start_payload={"context_revision_id": revision_id},
    )
    log.add("turn.finished", 20 * NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.compute_cp_ns.precision is Precision.EXACT
    assert turn.compute_cp_ns.value == 20 * NS


def test_invalid_duplicate_revision_segments_cannot_prove_tool_fan_in(tmp_path) -> None:
    path = _write_fan_in_membership_probe(
        tmp_path,
        {"item_count": 2, "unidentified_item_count": 0},
        duplicate_segment_index=True,
        file_name="invalid-revision-segments.jsonl",
    )

    turn = analyze_trajectory(path).turns[0]

    assert turn.compute_cp_ns.precision is Precision.UNRESOLVED
    assert turn.response_cp_ns.precision is Precision.UNRESOLVED
    assert any("invalid segmented membership" in diagnostic for diagnostic in turn.diagnostics)


@pytest.mark.parametrize(
    ("revision_payload", "expected_diagnostic", "fingerprint_key"),
    [
        (
            {"item_count": 1, "unidentified_item_count": 0},
            "item_count does not match replayed membership",
            None,
        ),
        (
            {"item_count": 2, "unidentified_item_count": 0, "membership_hash": "0" * 64},
            "membership_hash does not match replay",
            b"k" * 32,
        ),
    ],
)
def test_invalid_frozen_revision_membership_cannot_forge_tool_fan_in(
    tmp_path,
    revision_payload: dict[str, object],
    expected_diagnostic: str,
    fingerprint_key: bytes | None,
) -> None:
    path = _write_fan_in_membership_probe(tmp_path, revision_payload)

    turn = analyze_trajectory(path, fingerprint_key=fingerprint_key).turns[0]

    assert turn.compute_cp_ns.precision is Precision.UNRESOLVED
    assert turn.response_cp_ns.precision is Precision.UNRESOLVED
    assert any(expected_diagnostic in diagnostic for diagnostic in turn.diagnostics)


@pytest.mark.parametrize(
    ("parent_operation_id", "duplicate_revision", "expected_diagnostic"),
    [
        ("7" * 32, False, "does not belong to its claiming exchange"),
        ("9" * 32, True, "is defined more than once"),
    ],
)
def test_corrupt_revision_identity_or_exchange_ownership_cannot_prove_fan_in(
    tmp_path,
    parent_operation_id: str,
    duplicate_revision: bool,
    expected_diagnostic: str,
) -> None:
    path = _write_fan_in_membership_probe(
        tmp_path,
        {"item_count": 2, "unidentified_item_count": 0},
        parent_operation_id=parent_operation_id,
        duplicate_revision=duplicate_revision,
    )

    turn = analyze_trajectory(path).turns[0]

    assert turn.compute_cp_ns.precision is Precision.UNRESOLVED
    assert turn.response_cp_ns.precision is Precision.UNRESOLVED
    assert any(expected_diagnostic in diagnostic for diagnostic in turn.diagnostics)


def test_event_carrier_item_id_proves_chat_style_tool_fan_in_without_session_store(tmp_path) -> None:
    carrier_item_id = "7" * 32
    path = _write_fan_in_membership_probe(
        tmp_path,
        {"item_count": 2, "unidentified_item_count": 0},
        result_carrier_item_id=carrier_item_id,
        result_membership_item_id=carrier_item_id,
    )

    turn = analyze_trajectory(path).turns[0]

    assert turn.compute_cp_ns.precision is Precision.EXACT
    assert turn.response_cp_ns.precision is Precision.EXACT


@pytest.mark.parametrize("retry_mode", ["validation", "context_overflow"])
def test_two_exchange_retries_keep_exchange_references_family_exact(tmp_path, retry_mode: str) -> None:
    call_item_id = "1" * 32
    result_item_id = "2" * 32
    revision_id = "3" * 32
    log = EventLog()
    log.coverage()
    log.add(EventType.TURN_STARTED, 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 20 * NS, links=caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        10 * NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "f" * 32},
    )
    log.span("model.exchange", "d" * 32, 0, 2 * NS, parent_operation_id="c" * 32)
    for previous, following, start_ns in (
        ("d" * 32, "e" * 32, 2 * NS),
        ("e" * 32, "f" * 32, 5 * NS),
    ):
        payload = {
            "retry_mode": retry_mode,
            "previous_operation_id": previous,
            "next_operation_id": following,
        }
        log.add(
            EventType.RETRY_SCHEDULED,
            start_ns,
            operation_id=following,
            parent_operation_id="c" * 32,
            payload=payload,
        )
        log.add(
            EventType.RETRY_STARTED,
            start_ns + NS,
            operation_id=following,
            parent_operation_id="c" * 32,
            payload=payload,
        )
        log.span(
            "model.exchange",
            following,
            start_ns + NS,
            start_ns + 2 * NS,
            parent_operation_id="c" * 32,
        )
    log.span(
        "compaction",
        "6" * 32,
        7 * NS,
        8 * NS,
        parent_operation_id="f" * 32,
        start_payload={"compaction_run_id": "6" * 32, "trigger": "usage_threshold"},
    )
    log.span(
        "hook.operation",
        "5" * 32,
        8 * NS,
        9 * NS,
        start_payload={
            "hook_event": "user_interrupt",
            "execution_mode": "async",
            "target_operation_id": "f" * 32,
        },
    )
    log.span(
        "preparation",
        "0" * 32,
        8 * NS,
        8 * NS,
        parent_operation_id="f" * 32,
        start_payload={"scope": "tool_preamble", "phase": "dispatch", "target_operation_id": "7" * 32},
        links=caused_by("f" * 32),
    )
    log.span(
        "tool.operation",
        "7" * 32,
        8 * NS,
        10 * NS,
        parent_operation_id="f" * 32,
        start_payload={
            "tool_name": "read",
            "tool_kind": "filesystem.read",
            "parent_model_operation_id": "f" * 32,
            "call_item_id": call_item_id,
        },
        finish_payload={"result_item_id": result_item_id},
        links=caused_by("0" * 32),
    )
    revision = log.add(
        EventType.CONTEXT_REVISION_RECORDED,
        10 * NS,
        operation_id=revision_id,
        parent_operation_id="9" * 32,
        payload={
            "revision_id": revision_id,
            "is_checkpoint": True,
            "item_count": 2,
            "unidentified_item_count": 0,
        },
        segmented_fields=(SegmentedField(field_pointer="/payload/refs", segment_group_id="4" * 32, segment_count=1),),
    )
    log.add(
        EventType.SEGMENT,
        10 * NS,
        operation_id=None,
        payload={
            "parent_event_id": revision.event_id,
            "field_pointer": "/payload/refs",
            "segment_group_id": "4" * 32,
            "segment_index": 0,
            "segment_count": 1,
            "encoding": "array_slice",
            "entries": [
                {"item_id": call_item_id, "occurrence": 0, "position": 0, "action": "add"},
                {"item_id": result_item_id, "occurrence": 0, "position": 1, "action": "add"},
            ],
        },
    )
    log.span(
        "model.cycle",
        "8" * 32,
        10 * NS,
        20 * NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "9" * 32},
    )
    log.span(
        "model.exchange",
        "9" * 32,
        10 * NS,
        20 * NS,
        parent_operation_id="8" * 32,
        start_payload={"context_revision_id": revision_id},
    )
    log.add(EventType.TURN_FINISHED, 20 * NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / f"double-{retry_mode}-retry.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.elapsed_ns.precision is Precision.EXACT
    assert turn.compute_cp_ns.precision is Precision.EXACT
    assert turn.response_cp_ns.precision is Precision.EXACT
    assert not any(
        marker in diagnostic
        for diagnostic in turn.diagnostics
        for marker in ("retry predecessor", "retry successor", "preparation parent", "caused_by target")
    )
    assert turn.flow is not None
    assert (
        operation_index(turn, "model.exchange", "f" * 32),
        operation_index(turn, "hook.operation", "5" * 32),
    ) in set(turn.flow.causal_edges())


@pytest.mark.parametrize(
    ("consumed_item_ids", "with_successor_pointer", "expected_precision", "expected_diagnostic"),
    [
        (["2" * 32, "3" * 32], True, Precision.EXACT, None),
        (None, True, Precision.UNRESOLVED, "invalid segmented consumed item ids"),
        (["2" * 32, "3" * 32], False, Precision.UNRESOLVED, "lacks next_exchange_operation_id"),
        (["not-an-analytics-id"], True, Precision.UNRESOLVED, "invalid segmented consumed item ids"),
    ],
)
def test_phase4_consumption_bridges_pre_request_tool_results_fail_closed(
    tmp_path,
    consumed_item_ids: list[str] | None,
    with_successor_pointer: bool,
    expected_precision: Precision,
    expected_diagnostic: str | None,
) -> None:
    path = _write_phase4_fan_in_probe(
        tmp_path,
        consumed_item_ids=consumed_item_ids,
        with_successor_pointer=with_successor_pointer,
    )

    turn = analyze_trajectory(path).turns[0]

    assert turn.compute_cp_ns.precision is expected_precision
    assert turn.response_cp_ns.precision is expected_precision
    if expected_diagnostic is None:
        assert not any("tool result fan-in" in diagnostic for diagnostic in turn.diagnostics)
        assert turn.flow is not None
        causal_edges = set(turn.flow.causal_edges())
        assert (
            operation_index(turn, "tool.operation", "f" * 32),
            operation_index(turn, "compaction", "6" * 32),
        ) in causal_edges
        assert (
            operation_index(turn, "compaction", "6" * 32),
            operation_index(turn, "model.exchange", "9" * 32),
        ) in causal_edges
    else:
        assert any(expected_diagnostic in diagnostic for diagnostic in turn.diagnostics)


@pytest.mark.parametrize(
    ("scenario", "expected_diagnostic"),
    [
        ("duplicate_phase4", "compaction run declares more than one Phase-4 consumption event"),
        ("missing_successor", "Phase-4 compaction successor exchange cannot be resolved uniquely"),
        ("multiple_runs", "tool result fan-in matches multiple Phase-4 compaction runs"),
    ],
)
def test_phase4_consumption_rejects_ambiguous_topologies(
    tmp_path: Path,
    scenario: str,
    expected_diagnostic: str,
) -> None:
    path = _write_phase4_fan_in_probe(
        tmp_path,
        consumed_item_ids=["2" * 32, "3" * 32],
        with_successor_pointer=True,
        scenario=scenario,
    )

    turn = analyze_trajectory(path).turns[0]

    assert turn.compute_cp_ns.precision is Precision.UNRESOLVED
    assert turn.response_cp_ns.precision is Precision.UNRESOLVED
    assert any(expected_diagnostic in diagnostic for diagnostic in turn.diagnostics)


@pytest.mark.parametrize("with_session_store", [True, False])
def test_legacy_tool_fan_in_uses_only_a_reachable_session_carrier_mapping(
    tmp_path,
    with_session_store: bool,
) -> None:
    carrier_item_id = "7" * 32
    result_item_id = "2" * 32
    trajectory_dir = tmp_path / "sessions" / "fixture" / "trajectory"
    trajectory_dir.mkdir(parents=True)
    path = _write_fan_in_membership_probe(
        trajectory_dir,
        {"item_count": 2, "unidentified_item_count": 0},
        result_membership_item_id=carrier_item_id,
        file_name="events.jsonl",
    )
    if with_session_store:
        session = {
            "state": {
                "messages": [
                    {
                        "role": "tool",
                        "additional_properties": {ANALYTICS_ITEM_ID_KEY: carrier_item_id},
                        "contents": [
                            {
                                "type": "function_result",
                                "additional_properties": {ANALYTICS_ITEM_ID_KEY: result_item_id},
                            }
                        ],
                    }
                ]
            }
        }
        (trajectory_dir.parent / "session.json").write_text(json.dumps(session), encoding="utf-8")

    turn = analyze_trajectory(path).turns[0]

    if with_session_store:
        assert turn.compute_cp_ns.precision is Precision.EXACT
        assert turn.response_cp_ns.precision is Precision.EXACT
    else:
        assert turn.compute_cp_ns.precision is Precision.UNRESOLVED
        assert turn.response_cp_ns.precision is Precision.UNRESOLVED
        assert "carrier mapping unavailable" in turn.diagnostics


def test_live_tail_reprojects_clean_legacy_turn_when_session_carrier_mapping_changes(tmp_path) -> None:
    carrier_item_id = "7" * 32
    result_item_id = "2" * 32
    trajectory_dir = tmp_path / "sessions" / "fixture" / "trajectory"
    trajectory_dir.mkdir(parents=True)
    path = _write_fan_in_membership_probe(
        trajectory_dir,
        {"item_count": 2, "unidentified_item_count": 0},
        result_membership_item_id=carrier_item_id,
        file_name="events.jsonl",
    )
    analyzer = TrajectoryAnalyzer()
    initial = analyzer.load(path)
    assert initial.turns[0].compute_cp_ns.precision is Precision.UNRESOLVED

    session = {
        "state": {
            "messages": [
                {
                    "role": "tool",
                    "additional_properties": {ANALYTICS_ITEM_ID_KEY: carrier_item_id},
                    "contents": [
                        {
                            "type": "function_result",
                            "additional_properties": {ANALYTICS_ITEM_ID_KEY: result_item_id},
                        }
                    ],
                }
            ]
        }
    }
    session_path = trajectory_dir.parent / "session.json"
    session_path.write_text(json.dumps(session), encoding="utf-8")
    _append_profile_switched_event(path, monotonic_ns=21 * NS)

    mapped = analyzer.refresh()

    assert mapped.turns[0].compute_cp_ns.precision is Precision.EXACT
    assert mapped.turns[0].response_cp_ns.precision is Precision.EXACT

    session_path.unlink()
    _append_profile_switched_event(path, monotonic_ns=22 * NS)

    unavailable = analyzer.refresh()

    assert unavailable.turns[0].compute_cp_ns.precision is Precision.UNRESOLVED
    assert unavailable.turns[0].response_cp_ns.precision is Precision.UNRESOLVED
    assert "carrier mapping unavailable" in unavailable.turns[0].diagnostics


def _append_profile_switched_event(path, *, monotonic_ns: int) -> None:
    appended = EventLog()
    appended.add("profile.switched", monotonic_ns, turn_id=None, payload={"profile": "Code"})
    append_path = path.with_name("append.jsonl")
    appended.write(append_path, start_sequence=len(path.read_bytes().splitlines()) + 1)
    path.write_bytes(path.read_bytes() + append_path.read_bytes())


def test_real_shape_acceptance_keeps_side_calls_out_and_accepts_carrier_and_approval_shapes(tmp_path) -> None:
    side_actor = Actor(kind="side_call", role="approval_judge", actor_id="0" * 32)
    side_revision_id = "1" * 32
    side_exchange_id = "2" * 32
    side_segment_id = "3" * 32
    call_item_id = "4" * 32
    result_item_id = "5" * 32
    carrier_item_id = "6" * 32
    consuming_revision_id = "7" * 32
    consuming_segment_id = "8" * 32
    approval_id = "9" * 32
    tool_id = "f" * 32
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 20 * NS, links=caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        10 * NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "d" * 32},
    )
    log.span("model.exchange", "d" * 32, 0, 2 * NS, parent_operation_id="c" * 32)
    log.add(
        "approval.requested",
        1_800_000_000,
        operation_id=approval_id,
        parent_operation_id="d" * 32,
        payload={"approval_request_id": approval_id, "target_tool_operation_id": tool_id},
    )
    side_revision = log.add(
        "context.revision.recorded",
        2_100_000_000,
        operation_id=side_revision_id,
        parent_operation_id=side_exchange_id,
        payload={
            "revision_id": side_revision_id,
            "is_checkpoint": True,
            "item_count": 0,
            "unidentified_item_count": 2,
        },
        segmented_fields=(
            SegmentedField(field_pointer="/payload/refs", segment_group_id=side_segment_id, segment_count=1),
        ),
        actor=side_actor,
    )
    log.add(
        "event.segment",
        2_100_000_000,
        operation_id=None,
        payload={
            "parent_event_id": side_revision.event_id,
            "field_pointer": "/payload/refs",
            "segment_group_id": side_segment_id,
            "segment_index": 0,
            "segment_count": 1,
            "encoding": "array_slice",
            "entries": [],
        },
        actor=side_actor,
    )
    log.span(
        "model.exchange",
        side_exchange_id,
        2_200_000_000,
        2_700_000_000,
        parent_operation_id="d" * 32,
        start_payload={"context_revision_id": side_revision_id},
        actor=side_actor,
    )
    log.add(
        "approval.resolved",
        3 * NS,
        operation_id=approval_id,
        parent_operation_id="d" * 32,
        payload={
            "approval_request_id": approval_id,
            "target_tool_operation_id": tool_id,
            "outcome": "approved",
            "wait_ms": 2500,
        },
        measurements={"/payload/wait_ms": measurement("monotonic_clock", method_version=1)},
    )
    log.span(
        "preparation",
        "e" * 32,
        3 * NS,
        3 * NS,
        parent_operation_id="d" * 32,
        start_payload={"scope": "tool_preamble", "phase": "dispatch", "target_operation_id": tool_id},
    )
    log.span(
        "tool.operation",
        tool_id,
        3 * NS,
        10 * NS,
        parent_operation_id="d" * 32,
        start_payload={
            "tool_name": "read",
            "tool_kind": "filesystem.read",
            "parent_model_operation_id": "d" * 32,
            "call_item_id": call_item_id,
        },
        finish_payload={
            "result_item_id": result_item_id,
            "result_carrier_item_id": carrier_item_id,
        },
        links=caused_by("e" * 32),
    )
    revision = log.add(
        "context.revision.recorded",
        10 * NS,
        operation_id=consuming_revision_id,
        parent_operation_id="0" * 32,
        payload={
            "revision_id": consuming_revision_id,
            "is_checkpoint": True,
            "item_count": 2,
            "unidentified_item_count": 0,
        },
        segmented_fields=(
            SegmentedField(
                field_pointer="/payload/refs",
                segment_group_id=consuming_segment_id,
                segment_count=1,
            ),
        ),
    )
    log.add(
        "event.segment",
        10 * NS,
        operation_id=None,
        payload={
            "parent_event_id": revision.event_id,
            "field_pointer": "/payload/refs",
            "segment_group_id": consuming_segment_id,
            "segment_index": 0,
            "segment_count": 1,
            "encoding": "array_slice",
            "entries": [
                {"item_id": call_item_id, "occurrence": 0, "position": 0, "action": "add"},
                {"item_id": carrier_item_id, "occurrence": 0, "position": 1, "action": "add"},
            ],
        },
    )
    log.span(
        "model.cycle",
        "9" * 32,
        10 * NS,
        20 * NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "0" * 32},
    )
    log.span(
        "model.exchange",
        "0" * 32,
        10 * NS,
        20 * NS,
        parent_operation_id="9" * 32,
        start_payload={"context_revision_id": consuming_revision_id},
    )
    log.add("turn.finished", 20 * NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "real-shapes.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)
    turn = analysis.turns[0]

    assert turn.compute_cp_ns.precision is Precision.EXACT
    assert turn.response_cp_ns.precision is Precision.EXACT
    assert side_revision_id not in (turn.compute_cp_ns.reason or "")
    assert side_revision_id not in (turn.response_cp_ns.reason or "")
    assert len([operation for operation in turn.operations if operation.family == "model.exchange"]) == 2
    assert analysis.diagnostics.span_duration_mismatch_count == 0
    assert analysis.diagnostics.containment_violation_count == 0
    assert side_revision_id in analysis.diagnostics.side_call_empty_shell_revisions


def test_delta_membership_replay_requires_exact_occurrence_and_position() -> None:
    parent = (("1" * 32, 0), ("2" * 32, 0))
    valid = (
        _RevisionEntry("2" * 32, 0, 1, "remove"),
        _RevisionEntry("3" * 32, 0, 1, "add"),
    )
    wrong_occurrence = (
        _RevisionEntry("2" * 32, 1, 1, "remove"),
        _RevisionEntry("3" * 32, 0, 1, "add"),
    )

    assert _replay_delta(parent, valid) == (("1" * 32, 0), ("3" * 32, 0))
    assert _replay_delta(parent, wrong_occurrence) is None


def test_invalid_context_membership_is_diagnostic_without_poisoning_pure_timing(tmp_path) -> None:
    revision_id = "1" * 32
    exchange_id = "2" * 32
    group_id = "3" * 32
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, NS, links=caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": exchange_id},
    )
    revision = log.add(
        "context.revision.recorded",
        0,
        operation_id=revision_id,
        parent_operation_id=exchange_id,
        payload={"revision_id": revision_id, "is_checkpoint": True, "item_count": 1, "unidentified_item_count": 0},
        segmented_fields=(SegmentedField(field_pointer="/payload/refs", segment_group_id=group_id, segment_count=1),),
    )
    log.add(
        "event.segment",
        0,
        operation_id=None,
        payload={
            "parent_event_id": revision.event_id,
            "field_pointer": "/payload/refs",
            "segment_group_id": group_id,
            "segment_index": 0,
            "segment_count": 1,
            "encoding": "array_slice",
            "entries": [
                {"item_id": "4" * 32, "occurrence": 0, "position": 0, "action": "add"},
                {"item_id": "5" * 32, "occurrence": 0, "position": 1, "action": "add"},
            ],
        },
    )
    log.span(
        "model.exchange",
        exchange_id,
        0,
        NS,
        parent_operation_id="c" * 32,
        start_payload={"context_revision_id": revision_id},
    )
    log.add("turn.finished", NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "invalid-context-only.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.compute_cp_ns.precision is Precision.EXACT
    assert turn.response_cp_ns.precision is Precision.EXACT
    assert any("item_count does not match replayed membership" in item for item in turn.diagnostics)


def _write_fan_in_membership_probe(
    tmp_path,
    revision_payload: dict[str, object],
    *,
    parent_operation_id: str = "9" * 32,
    duplicate_revision: bool = False,
    duplicate_segment_index: bool = False,
    result_carrier_item_id: str | None = None,
    result_membership_item_id: str | None = None,
    file_name: str = "membership-fan-in.jsonl",
    tool_kind: str = "filesystem.read",
    tool_context: dict[str, str] | None = None,
):
    call_item_id = "1" * 32
    result_item_id = "2" * 32
    revision_id = "3" * 32
    group_id = "4" * 32
    consuming_exchange_id = "9" * 32
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 20 * NS, links=caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        10 * NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "d" * 32},
    )
    log.span("model.exchange", "d" * 32, 0, 2 * NS, parent_operation_id="c" * 32)
    log.span(
        "preparation",
        "e" * 32,
        2 * NS,
        2 * NS,
        parent_operation_id="d" * 32,
        start_payload={"scope": "tool_preamble", "phase": "dispatch", "target_operation_id": "f" * 32},
    )
    log.span(
        "tool.operation",
        "f" * 32,
        2 * NS,
        10 * NS,
        parent_operation_id="d" * 32,
        start_payload={
            "tool_name": "read",
            "tool_kind": tool_kind,
            "parent_model_operation_id": "d" * 32,
            "call_item_id": call_item_id,
            **({"tool_context": tool_context} if tool_context is not None else {}),
        },
        finish_payload={
            "result_item_id": result_item_id,
            **({"result_carrier_item_id": result_carrier_item_id} if result_carrier_item_id is not None else {}),
        },
        links=caused_by("e" * 32),
    )
    membership = [
        {"item_id": call_item_id, "occurrence": 0, "position": 0, "action": "add"},
        {
            "item_id": result_membership_item_id or result_item_id,
            "occurrence": 0,
            "position": 1,
            "action": "add",
        },
    ]
    segment_count = 2 if duplicate_segment_index else 1
    revision = log.add(
        "context.revision.recorded",
        10 * NS,
        operation_id=revision_id,
        parent_operation_id=parent_operation_id,
        payload={"revision_id": revision_id, "is_checkpoint": True, **revision_payload},
        segmented_fields=(
            SegmentedField(field_pointer="/payload/refs", segment_group_id=group_id, segment_count=segment_count),
        ),
    )
    # A duplicated segment index splits the membership into two slices that both
    # claim index 0 and position 0, so the replay cannot prove the fan-in.
    segments = [[{**entry, "position": 0}] for entry in membership] if duplicate_segment_index else [membership]
    for entries in segments:
        log.add(
            "event.segment",
            10 * NS,
            operation_id=None,
            payload={
                "parent_event_id": revision.event_id,
                "field_pointer": "/payload/refs",
                "segment_group_id": group_id,
                "segment_index": 0,
                "segment_count": segment_count,
                "encoding": "array_slice",
                "entries": entries,
            },
        )
    if duplicate_revision:
        log.add(
            "context.revision.recorded",
            10 * NS,
            operation_id=revision_id,
            parent_operation_id=parent_operation_id,
            payload={"revision_id": revision_id, "is_checkpoint": True, **revision_payload},
        )
    log.span(
        "model.cycle",
        "8" * 32,
        10 * NS,
        20 * NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": consuming_exchange_id},
    )
    log.span(
        "model.exchange",
        consuming_exchange_id,
        10 * NS,
        20 * NS,
        parent_operation_id="8" * 32,
        start_payload={"context_revision_id": revision_id},
    )
    log.add("turn.finished", 20 * NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / file_name
    log.write(path)
    return path


def _write_phase4_fan_in_probe(
    tmp_path,
    *,
    consumed_item_ids: list[str] | None,
    with_successor_pointer: bool,
    scenario: str = "valid",
):
    call_item_id = "1" * 32
    result_item_id = "2" * 32
    carrier_item_id = "3" * 32
    compaction_id = "6" * 32
    consuming_cycle_id = "7" * 32
    consuming_exchange_id = "9" * 32
    log = EventLog()
    log.coverage()
    log.add(EventType.TURN_STARTED, 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 20 * NS, links=caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        8 * NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "d" * 32},
    )
    log.span("model.exchange", "d" * 32, 0, 2 * NS, parent_operation_id="c" * 32)
    log.span(
        "preparation",
        "e" * 32,
        2 * NS,
        2 * NS,
        parent_operation_id="d" * 32,
        start_payload={"scope": "tool_preamble", "phase": "dispatch", "target_operation_id": "f" * 32},
    )
    log.span(
        "tool.operation",
        "f" * 32,
        2 * NS,
        8 * NS,
        parent_operation_id="d" * 32,
        start_payload={
            "tool_name": "read",
            "tool_kind": "filesystem.read",
            "parent_model_operation_id": "d" * 32,
            "call_item_id": call_item_id,
        },
        finish_payload={
            "result_item_id": result_item_id,
            "result_carrier_item_id": carrier_item_id,
        },
        links=caused_by("e" * 32),
    )
    log.add(
        EventType.MODEL_CYCLE_STARTED,
        8 * NS,
        operation_id=consuming_cycle_id,
        parent_operation_id="b" * 32,
    )

    def add_phase4_run(
        *,
        run_id: str,
        phase_id: str,
        segment_group_id: str,
        start_ns: int,
        phase_finish_ns: int,
        finish_ns: int,
        successor_id: str,
        duplicate_phase: bool = False,
    ) -> None:
        log.add(
            EventType.COMPACTION_STARTED,
            start_ns,
            operation_id=run_id,
            parent_operation_id=consuming_cycle_id,
            payload={"compaction_run_id": run_id, "trigger": "usage_threshold"},
        )

        def add_phase(*, operation_id: str, group_id: str) -> None:
            segmented_fields = ()
            if consumed_item_ids is not None:
                segmented_fields = (
                    SegmentedField(
                        field_pointer="/payload/consumed_item_ids",
                        segment_group_id=group_id,
                        segment_count=1,
                    ),
                )
            phase_payload = {
                "compaction_run_id": run_id,
                "phase": "phase4",
                "groups_compacted": 1,
                "duration_ms": (phase_finish_ns - start_ns) // 1_000_000,
                "last_words_generated": True,
            }
            if with_successor_pointer:
                phase_payload["next_exchange_operation_id"] = successor_id
            phase = log.add(
                EventType.COMPACTION_PHASE_FINISHED,
                phase_finish_ns,
                operation_id=operation_id,
                parent_operation_id=run_id,
                payload=phase_payload,
                measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
                segmented_fields=segmented_fields,
            )
            if consumed_item_ids is not None:
                log.add(
                    EventType.SEGMENT,
                    phase_finish_ns,
                    operation_id=None,
                    payload={
                        "parent_event_id": phase.event_id,
                        "field_pointer": "/payload/consumed_item_ids",
                        "segment_group_id": group_id,
                        "segment_index": 0,
                        "segment_count": 1,
                        "encoding": "array_slice",
                        "entries": consumed_item_ids,
                    },
                )

        add_phase(operation_id=phase_id, group_id=segment_group_id)
        if duplicate_phase:
            add_phase(operation_id="4" * 32, group_id="1" * 32)
        log.add(
            EventType.COMPACTION_FINISHED,
            finish_ns,
            operation_id=run_id,
            parent_operation_id=consuming_cycle_id,
            payload={"outcome": "success", "duration_ms": (finish_ns - start_ns) // 1_000_000},
            measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
        )

    successor_id = "8" * 32 if scenario == "missing_successor" else consuming_exchange_id
    if scenario == "multiple_runs":
        add_phase4_run(
            run_id=compaction_id,
            phase_id="5" * 32,
            segment_group_id="0" * 32,
            start_ns=8 * NS,
            phase_finish_ns=9 * NS,
            finish_ns=10 * NS,
            successor_id=successor_id,
        )
        add_phase4_run(
            run_id="4" * 32,
            phase_id="8" * 32,
            segment_group_id="1" * 32,
            start_ns=10 * NS,
            phase_finish_ns=11 * NS,
            finish_ns=12 * NS,
            successor_id=successor_id,
        )
    else:
        add_phase4_run(
            run_id=compaction_id,
            phase_id="5" * 32,
            segment_group_id="0" * 32,
            start_ns=8 * NS,
            phase_finish_ns=11 * NS,
            finish_ns=12 * NS,
            successor_id=successor_id,
            duplicate_phase=scenario == "duplicate_phase4",
        )
    log.span(
        "model.exchange",
        consuming_exchange_id,
        12 * NS,
        20 * NS,
        parent_operation_id=consuming_cycle_id,
    )
    log.add(
        EventType.MODEL_CYCLE_FINISHED,
        20 * NS,
        operation_id=consuming_cycle_id,
        parent_operation_id="b" * 32,
        payload={
            "outcome": "success",
            "duration_ms": 12_000,
            "final_exchange_operation_id": consuming_exchange_id,
        },
        measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
    )
    log.add(EventType.TURN_FINISHED, 20 * NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / f"phase4-fan-in-{consumed_item_ids is not None}-{with_successor_pointer}-{scenario}.jsonl"
    log.write(path)
    return path


def test_server_cp_group_is_exact_only_for_exact_response_turns(tmp_path) -> None:
    exact_path = _write_fan_in_membership_probe(
        tmp_path,
        {"item_count": 2, "unidentified_item_count": 0},
        file_name="exact-server-cp.jsonl",
        tool_kind="mcp",
        tool_context={"server_name": "figma", "remote_name": "read"},
    )
    exact = analyze_trajectory(exact_path)
    assert exact.insights is not None
    assert exact.turns[0].response_cp_ns.precision is Precision.EXACT
    assert exact.insights.mcp.rows[0].critical_path_exclusive_ns.precision is Precision.EXACT
    assert exact.insights.mcp.rows[0].critical_path_exclusive_ns.value == 8 * NS

    unresolved_path = _write_fan_in_membership_probe(
        tmp_path,
        {"item_count": 1, "unidentified_item_count": 0},
        file_name="unresolved-server-cp.jsonl",
        tool_kind="mcp",
        tool_context={"server_name": "figma", "remote_name": "read"},
    )
    unresolved = analyze_trajectory(unresolved_path)
    assert unresolved.insights is not None
    assert unresolved.turns[0].response_cp_ns.precision is Precision.UNRESOLVED
    cp_metric = unresolved.insights.mcp.rows[0].critical_path_exclusive_ns
    assert cp_metric.value is None
    assert cp_metric.precision is Precision.UNRESOLVED

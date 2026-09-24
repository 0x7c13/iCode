# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Which events belong to a turn, and how corruption, foreign lifecycles, and out-of-turn actions degrade it."""

from __future__ import annotations

import pytest

from chrys.foundation.trajectory.envelope import Actor
from chrys.foundation.trajectory.event_types import EventType
from chrys.service.analytics import (
    AnalysisAvailability,
    Metric,
    Precision,
    UsageBucket,
    analyze_trajectory,
)
from tests.service.analytics._events import NS, EventLog, caused_by


def test_all_corrupt_input_degrades_every_exported_session_metric_group(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text("not-json\n", encoding="utf-8")

    analysis = analyze_trajectory(path)

    assert analysis.turns == ()
    assert analysis.overview is not None
    assert analysis.overview.elapsed_ns == Metric(
        0,
        Precision.UNRESOLVED,
        "session trajectory integrity is unresolved: corrupt lines",
    )
    assert analysis.token_usage is not None
    assert analysis.token_usage.buckets[UsageBucket.INPUT].precision is Precision.UNRESOLVED
    assert analysis.token_usage.buckets[UsageBucket.OUTPUT].precision is Precision.UNRESOLVED
    assert all(
        analysis.token_usage.buckets[bucket].precision is Precision.MISSING
        for bucket in (UsageBucket.REASONING, UsageBucket.CACHE_READ, UsageBucket.CACHE_CREATION)
    )
    assert analysis.validation is not None
    assert analysis.validation.tool_count.precision is Precision.UNRESOLVED
    assert analysis.validation.time_to_first_edit_ns.precision is Precision.MISSING
    assert analysis.change_verification is not None
    assert analysis.change_verification.files_touched.precision is Precision.UNRESOLVED
    assert analysis.change_verification.files_touched.reason == (
        "summed per-turn summary counts; usable session.json file detail is unavailable to fold repeat touches; "
        "session trajectory integrity is unresolved: corrupt lines"
    )


def test_corruption_after_a_turn_degrades_only_session_scope_metrics(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add("turn.finished", NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)
    path.write_bytes(path.read_bytes() + b"{corrupt after the turn}\n")

    analysis = analyze_trajectory(path)

    assert analysis.turns[0].elapsed_ns.precision is Precision.EXACT
    assert analysis.overview is not None
    assert analysis.overview.elapsed_ns.precision is Precision.UNRESOLVED
    assert analysis.overview.elapsed_ns.reason == "session trajectory integrity is unresolved: corrupt lines"


def test_empty_log_degrades_exact_zero_session_metrics(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"")

    analysis = analyze_trajectory(path)

    assert analysis.availability is AnalysisAvailability.AVAILABLE
    assert analysis.diagnostics.line_count == 0
    assert analysis.diagnostics.integrity_unresolved is True
    assert analysis.overview is not None
    assert analysis.overview.elapsed_ns == Metric(
        0,
        Precision.UNRESOLVED,
        "session trajectory integrity is unresolved: empty log",
    )


def test_torn_only_log_is_not_mislabeled_as_empty(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    tail = b'{"incomplete"'
    path.write_bytes(tail)

    analysis = analyze_trajectory(path)

    assert analysis.diagnostics.line_count == 0
    assert analysis.diagnostics.byte_count == 0
    assert analysis.diagnostics.torn_tail_bytes == len(tail)
    assert analysis.overview is not None
    assert analysis.overview.elapsed_ns.reason == "session trajectory integrity is unresolved: torn tail"


def test_integrity_damage_caps_usage_panels_and_nested_insights(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span(
        "tool.operation",
        "a" * 32,
        NS,
        2 * NS,
        start_payload={
            "tool_name": "load_skill",
            "tool_kind": "skill",
            "tool_context": {"skill_name": "slides", "skill_revision": "rev-a"},
        },
        finish_payload={"outcome": "success"},
    )
    log.span(
        "tool.operation",
        "b" * 32,
        3 * NS,
        4 * NS,
        start_payload={
            "tool_name": "run_skill_script",
            "tool_kind": "skill",
            "tool_context": {
                "skill_name": "slides",
                "skill_revision": "rev-a",
                "script_name": "scripts/render.py",
            },
        },
        finish_payload={"outcome": "success"},
    )
    log.span(
        "tool.operation",
        "c" * 32,
        5 * NS,
        6 * NS,
        start_payload={
            "tool_name": "figma_render",
            "tool_kind": "mcp",
            "tool_context": {"server_name": "figma", "remote_name": "render"},
        },
        finish_payload={"outcome": "success"},
    )
    log.add("turn.finished", 7 * NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)
    path.write_bytes(path.read_bytes() + b"{corrupt after the turn}\n")

    analysis = analyze_trajectory(path)

    reason = "session trajectory integrity is unresolved: corrupt lines"
    assert analysis.skill_usage is not None
    assert (analysis.skill_usage.total, analysis.skill_usage.precision, analysis.skill_usage.reason) == (
        2,
        Precision.UNRESOLVED,
        reason,
    )
    assert analysis.mcp_usage is not None
    assert (analysis.mcp_usage.total, analysis.mcp_usage.precision, analysis.mcp_usage.reason) == (
        1,
        Precision.UNRESOLVED,
        reason,
    )
    assert analysis.insights is not None
    assert analysis.insights.tools.precision is Precision.UNRESOLVED
    assert analysis.insights.mcp.precision is Precision.UNRESOLVED
    assert analysis.insights.skills.precision is Precision.UNRESOLVED
    assert analysis.insights.context_carrying_precision is Precision.UNRESOLVED
    assert {
        analysis.insights.tools.reason,
        analysis.insights.mcp.reason,
        analysis.insights.skills.reason,
        analysis.insights.context_carrying_reason,
    } == {reason}
    assert analysis.insights.tools.rows[0].duration_share.precision is Precision.UNRESOLVED
    assert analysis.insights.mcp.rows[0].duration_share.precision is Precision.UNRESOLVED
    assert analysis.insights.skills.rows[0].first_action_median_ns.precision is Precision.UNRESOLVED


@pytest.mark.parametrize("damage", ["gap", "unsupported", "torn"])
def test_non_turn_integrity_damage_degrades_session_scope_metrics(tmp_path, damage: str) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add("turn.finished", NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    if damage == "gap":
        log.add(EventType.GAP, NS, turn_id=None, payload={"first_sequence": 100, "last_sequence": 100})
    elif damage == "unsupported":
        log.add("future.event", NS, turn_id=None)
    path = tmp_path / "events.jsonl"
    log.write(path)
    if damage == "torn":
        path.write_bytes(path.read_bytes() + b'{"incomplete"')

    analysis = analyze_trajectory(path)

    assert analysis.turns[0].elapsed_ns.precision is Precision.EXACT
    assert analysis.overview is not None
    assert analysis.overview.elapsed_ns.precision is Precision.UNRESOLVED


def test_cross_runtime_turn_terminal_degrades_wall_and_derived_metrics(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add(
        "turn.finished",
        5 * NS,
        payload={"end_reason": "cancelled", "duration_ms": 0},
        runtime_id="9" * 32,
    )
    path = tmp_path / "cross-runtime-turn.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.elapsed_ns.precision is Precision.UNRESOLVED
    assert all(metric.precision is Precision.UNRESOLVED for metric in turn.wall_time_ns.values())
    assert turn.exclusive_work_ns.precision is Precision.UNRESOLVED
    assert turn.parallelism.precision is Precision.UNRESOLVED
    assert turn.overlap_gain_ns.precision is Precision.UNRESOLVED
    assert "turn interval has invalid monotonic endpoints" in turn.diagnostics


def test_main_role_lifecycle_endpoints_from_different_actors_are_not_paired_exactly(tmp_path) -> None:
    other_main_actor = Actor(kind="agent", role="main", actor_id="f" * 32)
    operation_id = "a" * 32
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add("model.exchange.started", 0, operation_id=operation_id)
    log.add(
        "model.exchange.finished",
        NS,
        operation_id=operation_id,
        actor=other_main_actor,
        payload={
            "outcome": "success",
            "duration_ms": 1000,
            "usage": {"normalized": {"input_total": 25, "output_total": 5}},
        },
        measurements={
            "/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1},
            "/payload/usage/normalized/input_total": {"source": "provider", "adapter_version": 1},
            "/payload/usage/normalized/output_total": {"source": "provider", "adapter_version": 1},
        },
    )
    log.add("turn.finished", NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "actor-mismatch.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.usage_tokens == Metric(
        30,
        Precision.UNRESOLVED,
        "usage requires a uniquely closed exchange and exact turn lifecycle",
    )
    operation = next(item for item in turn.operations if item.operation_id == operation_id)
    assert operation.precision is Precision.UNRESOLVED


@pytest.mark.parametrize(
    ("runtime_id", "branch_id"),
    [("9" * 32, "3" * 32), ("1" * 32, "9" * 32)],
)
def test_foreign_runtime_or_branch_subtree_never_enters_turn_arithmetic(
    tmp_path,
    runtime_id: str,
    branch_id: str,
) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span(
        "preparation",
        "a" * 32,
        0,
        0,
        start_payload={"scope": "turn_preamble", "phase": "dispatch"},
        runtime_id=runtime_id,
        branch_id=branch_id,
    )
    log.span(
        "model.run",
        "b" * 32,
        0,
        10 * NS,
        links=caused_by("a" * 32),
        runtime_id=runtime_id,
        branch_id=branch_id,
    )
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        10 * NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "d" * 32},
        runtime_id=runtime_id,
        branch_id=branch_id,
    )
    log.span(
        "model.exchange",
        "d" * 32,
        0,
        10 * NS,
        parent_operation_id="c" * 32,
        runtime_id=runtime_id,
        branch_id=branch_id,
    )
    log.add("turn.finished", 10 * NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "foreign-subtree.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.elapsed_ns.precision is Precision.UNRESOLVED
    assert turn.compute_cp_ns.precision is Precision.UNRESOLVED
    assert turn.response_cp_ns.precision is Precision.UNRESOLVED
    assert turn.exclusive_work_ns.precision is Precision.UNRESOLVED
    assert turn.parallelism.precision is Precision.UNRESOLVED
    assert turn.overlap_gain_ns.precision is Precision.UNRESOLVED
    assert all(metric.precision is Precision.UNRESOLVED for metric in turn.wall_time_ns.values())
    assert all(metric.precision is Precision.UNRESOLVED for metric in turn.utilization.values())
    assert not any(item.operation_id is not None for item in turn.slices)
    assert any("owning turn runtime and branch" in diagnostic for diagnostic in turn.diagnostics)


def test_lifecycle_terminal_must_follow_its_start_in_sequence_order(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.add("model.run.started", 0, operation_id="b" * 32, links=caused_by("a" * 32))
    log.add("model.cycle.started", 0, operation_id="c" * 32, parent_operation_id="b" * 32)
    log.add(
        "model.exchange.finished",
        10 * NS,
        operation_id="d" * 32,
        parent_operation_id="c" * 32,
        payload={"outcome": "success", "duration_ms": 10_000},
        measurements={"/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1}},
    )
    log.add("model.exchange.started", 0, operation_id="d" * 32, parent_operation_id="c" * 32)
    log.add(
        "model.cycle.finished",
        10 * NS,
        operation_id="c" * 32,
        parent_operation_id="b" * 32,
        payload={"outcome": "success", "duration_ms": 10_000, "final_exchange_operation_id": "d" * 32},
        measurements={"/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1}},
    )
    log.add(
        "model.run.finished",
        10 * NS,
        operation_id="b" * 32,
        payload={"outcome": "success", "duration_ms": 10_000},
        measurements={"/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1}},
    )
    log.add("turn.finished", 10 * NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "reversed-exchange.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.compute_cp_ns.precision is Precision.UNRESOLVED
    assert turn.response_cp_ns.precision is Precision.UNRESOLVED
    assert any("model.exchange interval has invalid monotonic endpoints" in item for item in turn.diagnostics)


def test_corruption_between_response_fence_and_segment_degrades_turn_integrity(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 10 * NS, links=caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        10 * NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "d" * 32},
    )
    log.span("model.exchange", "d" * 32, 0, 10 * NS, parent_operation_id="c" * 32)
    log.add("turn.finished", 10 * NS, payload={"end_reason": "completed", "duration_ms": 0})
    log.settled(11 * NS)
    path = tmp_path / "corrupt-fence-evidence.jsonl"
    log.write(path)
    lines = path.read_bytes().splitlines(keepends=True)
    lines.insert(-1, b"{corrupt between fence and segment}\n")
    path.write_bytes(b"".join(lines))

    turn = analyze_trajectory(path).turns[0]

    assert turn.elapsed_ns.precision is Precision.UNRESOLVED
    assert turn.response_cp_ns.precision is Precision.UNRESOLVED
    assert any("corrupt trajectory line intersects the turn" in item for item in turn.diagnostics)


def test_turn_after_closed_coverage_and_runtime_is_unresolved(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("trajectory.coverage.ended", 0, turn_id=None, payload={"last_sequence": 1})
    log.add("trajectory.runtime.finished", 0, turn_id=None, payload={"reason": "session_close"})
    log.add("turn.started", NS, payload={"turn_number": 1})
    log.add("turn.finished", 2 * NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "closed-runtime-turn.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.elapsed_ns.precision is Precision.UNRESOLVED
    assert turn.compute_cp_ns.precision is Precision.UNRESOLVED
    assert any("outside its trajectory coverage window" in item for item in turn.diagnostics)
    assert any("after trajectory.runtime.finished" in item for item in turn.diagnostics)


def test_usage_is_unresolved_when_a_corrupt_line_intersects_the_turn(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add("model.exchange.started", 0, operation_id="a" * 32)
    log.add(
        "model.exchange.finished",
        NS,
        operation_id="a" * 32,
        payload={
            "outcome": "success",
            "duration_ms": 1000,
            "usage": {"normalized": {"input_total": 100, "output_total": 20}},
        },
        measurements={
            "/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1},
            "/payload/usage/normalized/input_total": {"source": "provider", "adapter_version": 1},
            "/payload/usage/normalized/output_total": {"source": "provider", "adapter_version": 1},
        },
    )
    log.add("turn.finished", NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "corrupt-usage-region.jsonl"
    log.write(path)
    lines = path.read_bytes().splitlines(keepends=True)
    lines.insert(3, b"{complete but corrupt}\n")
    path.write_bytes(b"".join(lines))

    analysis = analyze_trajectory(path)
    usage = analysis.turns[0].usage_tokens

    assert usage.value == 120
    assert usage.precision is Precision.UNRESOLVED
    assert usage.reason == "corrupt trajectory line intersects the turn"
    assert analysis.diagnostics.corrupt_lines[0].after_sequence == 3
    assert analysis.diagnostics.corrupt_lines[0].line_number == 4


def test_closed_turn_before_later_prefix_violation_remains_exact(tmp_path) -> None:
    first = EventLog()
    first.coverage()
    first.turn(0, NS)
    path = tmp_path / "regional-prefix.jsonl"
    first.write(path)
    second_turn_id = "5" * 32
    second = EventLog()
    second.add("turn.started", 2 * NS, turn_id=second_turn_id, payload={"turn_number": 2})
    second.add(
        "turn.finished",
        3 * NS,
        turn_id=second_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 0},
    )
    append_path = tmp_path / "regional-prefix-append.jsonl"
    second.write(append_path, start_sequence=5)
    path.write_bytes(path.read_bytes() + append_path.read_bytes())

    analysis = analyze_trajectory(path)

    assert analysis.turns[0].elapsed_ns.precision is Precision.EXACT
    assert analysis.turns[1].elapsed_ns.precision is Precision.UNRESOLVED
    assert analysis.diagnostics.accounted_prefix_violations == (
        "sequence 4 is missing and not covered by an earlier gap",
    )
    assert analysis.diagnostics.accounted_prefix_violation_details[0].first_sequence == 4
    assert analysis.diagnostics.accounted_prefix_violation_details[0].last_sequence == 4


def test_tool_action_start_beyond_the_turn_terminal_degrades_action_projection(tmp_path) -> None:
    """The scope check bounds actions from below; the terminal bounds them from above."""
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("tool.operation", "a" * 32, 0, NS, start_payload={"tool_name": "zsh", "tool_kind": "shell"})
    log.add("turn.finished", NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    log.span("tool.operation", "b" * 32, 2 * NS, 3 * NS, start_payload={"tool_name": "zsh", "tool_kind": "shell"})
    path = tmp_path / "events.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.action_projection_precision is not Precision.EXACT
    assert turn.action_projection_reason is not None
    assert "tool action endpoint lies beyond the turn terminal" in turn.action_projection_reason


def test_tool_action_between_turn_finish_and_response_fence_degrades_action_projection(tmp_path) -> None:
    """The bound is ``turn.finished`` itself, not the later response fence."""
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("tool.operation", "a" * 32, 0, NS, start_payload={"tool_name": "zsh", "tool_kind": "shell"})
    log.add("turn.finished", 2 * NS, payload={"end_reason": "completed", "duration_ms": 0})
    log.span("tool.operation", "b" * 32, 3 * NS, 4 * NS, start_payload={"tool_name": "zsh", "tool_kind": "shell"})
    log.settled(5 * NS)
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)
    turn = analysis.turns[0]

    assert turn.action_projection_precision is not Precision.EXACT
    assert turn.action_projection_reason is not None
    assert "tool action endpoint lies beyond the turn terminal" in turn.action_projection_reason
    stray = next(action for action in analysis.actions if action.operation_id == "b" * 32)
    assert stray.outcome is None
    assert stray.outcome_precision is Precision.UNRESOLVED


def test_tool_terminal_beyond_the_turn_finish_degrades_action_projection(tmp_path) -> None:
    """An outcome read from a finish beyond the turn is as unfounded as a stray start."""
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add(
        "tool.operation.started",
        NS,
        operation_id="a" * 32,
        payload={"tool_name": "zsh", "tool_kind": "shell"},
    )
    log.add("turn.finished", 2 * NS, payload={"end_reason": "completed", "duration_ms": 0})
    log.add(
        "tool.operation.finished",
        3 * NS,
        operation_id="a" * 32,
        payload={"outcome": "success", "duration_ms": 2000},
    )
    log.settled(4 * NS)
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)
    turn = analysis.turns[0]

    assert turn.action_projection_precision is not Precision.EXACT
    assert turn.action_projection_reason is not None
    assert "tool action endpoint lies beyond the turn terminal" in turn.action_projection_reason
    (action,) = analysis.actions
    assert action.outcome is None
    assert action.outcome_precision is Precision.UNRESOLVED


def test_tool_action_within_a_completed_turn_stays_exact_with_a_response_fence(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("tool.operation", "a" * 32, 0, NS, start_payload={"tool_name": "zsh", "tool_kind": "shell"})
    log.add("turn.finished", 2 * NS, payload={"end_reason": "completed", "duration_ms": 0})
    log.settled(3 * NS)
    path = tmp_path / "events.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.action_projection_precision is Precision.EXACT
    assert turn.action_projection_reason is None

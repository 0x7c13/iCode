# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Collected 200 MiB resident-memory benchmark for trajectory aggregation.

Run with::

    uv run pytest tests/service/analytics/test_aggregation_memory_acceptance.py -m integration -n0

The fixture has roughly the audited event density per turn, mixes weighted
operation spans with marker-only events, and is generated outside the measured
child process. The child samples RSS while the production analyzer retains its
intermediate state and resolved view model.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from time import perf_counter

import psutil
import pytest

from chrys.foundation.trajectory.envelope import (
    SYSTEM_ACTOR,
    Actor,
    ActorKind,
    ActorRole,
    EventDraft,
    Link,
    LinkRelation,
    SegmentedField,
    build_event,
    encode_event_line,
    measurement,
)
from chrys.foundation.trajectory.event_types import EventType
from chrys.foundation.trajectory.revisions import RevisionChain, membership_of
from chrys.service.analytics._critical_path import _longest_interval_path
from chrys.service.analytics.aggregation import MAX_RESIDENT_MEMORY_BYTES, analyze_trajectory
from tests.service.analytics._events import BRANCH_ID, COVERAGE_ID, RUNTIME_ID, SESSION_ID

_MIB = 1024 * 1024
_FIXTURE_BYTES = 200 * _MIB
_MARKER_PADDING = "x" * 480


def _identifier(value: int) -> str:
    return f"{value:032x}"


def _write_fixture(path: Path) -> None:
    sequence = 0
    identifier = 10

    with path.open("wb") as handle:

        def emit(draft: EventDraft) -> None:
            nonlocal sequence
            sequence += 1
            handle.write(
                encode_event_line(
                    build_event(
                        draft,
                        sequence=sequence,
                        runtime_id=RUNTIME_ID,
                        coverage_id=COVERAGE_ID,
                        session_id=SESSION_ID,
                        branch_id=BRANCH_ID,
                    )
                )
            )

        emit(
            EventDraft(
                event_type=EventType.COVERAGE_STARTED,
                actor=SYSTEM_ACTOR,
                payload={"coverage_reason": "session_started"},
                monotonic_ns=0,
            )
        )
        turn_number = 0
        while handle.tell() < _FIXTURE_BYTES:
            turn_number += 1
            turn_id = _identifier(identifier)
            identifier += 1
            clock = turn_number * 1_000_000_000
            emit(
                EventDraft(
                    event_type=EventType.TURN_STARTED,
                    actor=SYSTEM_ACTOR,
                    turn_id=turn_id,
                    payload={"turn_number": turn_number},
                    monotonic_ns=clock,
                )
            )
            preamble_id = _identifier(identifier)
            identifier += 1
            emit(
                EventDraft(
                    event_type=EventType.PREPARATION_STARTED,
                    actor=SYSTEM_ACTOR,
                    turn_id=turn_id,
                    operation_id=preamble_id,
                    payload={"scope": "turn_preamble", "phase": "turn_dispatch"},
                    monotonic_ns=clock,
                )
            )
            emit(
                EventDraft(
                    event_type=EventType.PREPARATION_FINISHED,
                    actor=SYSTEM_ACTOR,
                    turn_id=turn_id,
                    operation_id=preamble_id,
                    payload={
                        "scope": "turn_preamble",
                        "phase": "turn_dispatch",
                        "outcome": "handoff",
                        "duration_ms": 1,
                    },
                    measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
                    monotonic_ns=clock + 1_000_000,
                )
            )
            run_id = _identifier(identifier)
            identifier += 1
            emit(
                EventDraft(
                    event_type=EventType.MODEL_RUN_STARTED,
                    actor=SYSTEM_ACTOR,
                    turn_id=turn_id,
                    operation_id=run_id,
                    links=(Link(relation=LinkRelation.CAUSED_BY, target_operation_id=preamble_id),),
                    monotonic_ns=clock + 1_000_000,
                )
            )
            for operation_index in range(20):
                exchange_id = _identifier(identifier)
                identifier += 1
                start = clock + (operation_index + 2) * 1_000_000
                emit(
                    EventDraft(
                        event_type=EventType.MODEL_EXCHANGE_STARTED,
                        actor=SYSTEM_ACTOR,
                        turn_id=turn_id,
                        operation_id=exchange_id,
                        parent_operation_id=run_id,
                        monotonic_ns=start,
                    )
                )
                emit(
                    EventDraft(
                        event_type=EventType.MODEL_EXCHANGE_FINISHED,
                        actor=SYSTEM_ACTOR,
                        turn_id=turn_id,
                        operation_id=exchange_id,
                        parent_operation_id=run_id,
                        payload={
                            "outcome": "success",
                            "duration_ms": 1,
                            "usage": {"normalized": {"input_total": 100, "output_total": 20}},
                        },
                        measurements={
                            "/payload/duration_ms": measurement("monotonic_clock", method_version=1),
                            "/payload/usage/normalized/input_total": measurement("provider", adapter_version=1),
                            "/payload/usage/normalized/output_total": measurement("provider", adapter_version=1),
                        },
                        monotonic_ns=start + 1_000_000,
                    )
                )
            for operation_index in range(20):
                preamble_id = _identifier(identifier)
                tool_id = _identifier(identifier + 1)
                call_item_id = _identifier(identifier + 2)
                identifier += 3
                start = clock + (operation_index + 25) * 1_000_000
                emit(
                    EventDraft(
                        event_type=EventType.PREPARATION_STARTED,
                        actor=SYSTEM_ACTOR,
                        turn_id=turn_id,
                        operation_id=preamble_id,
                        parent_operation_id=run_id,
                        payload={
                            "scope": "tool_preamble",
                            "phase": "tool_dispatch",
                            "target_operation_id": tool_id,
                        },
                        monotonic_ns=start,
                    )
                )
                emit(
                    EventDraft(
                        event_type=EventType.PREPARATION_FINISHED,
                        actor=SYSTEM_ACTOR,
                        turn_id=turn_id,
                        operation_id=preamble_id,
                        parent_operation_id=run_id,
                        payload={
                            "scope": "tool_preamble",
                            "phase": "tool_dispatch",
                            "outcome": "handoff",
                            "duration_ms": 1,
                        },
                        measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
                        monotonic_ns=start + 1_000_000,
                    )
                )
                emit(
                    EventDraft(
                        event_type=EventType.TOOL_OPERATION_STARTED,
                        actor=SYSTEM_ACTOR,
                        turn_id=turn_id,
                        operation_id=tool_id,
                        parent_operation_id=run_id,
                        payload={
                            "tool_name": "read_file",
                            "tool_kind": "filesystem.read",
                            "call_item_id": call_item_id,
                            "argument_fingerprint": f"fixture-read-{operation_index}",
                        },
                        links=(Link(relation=LinkRelation.CAUSED_BY, target_operation_id=preamble_id),),
                        monotonic_ns=start + 1_000_000,
                    )
                )
                emit(
                    EventDraft(
                        event_type=EventType.TOOL_OPERATION_FINISHED,
                        actor=SYSTEM_ACTOR,
                        turn_id=turn_id,
                        operation_id=tool_id,
                        parent_operation_id=run_id,
                        payload={"outcome": "success", "duration_ms": 1},
                        measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
                        monotonic_ns=start + 2_000_000,
                    )
                )
            for marker_index in range(90):
                emit(
                    EventDraft(
                        event_type=EventType.CONTEXT_REVISION_RECORDED,
                        actor=SYSTEM_ACTOR,
                        turn_id=turn_id,
                        payload={"fixture_padding": _MARKER_PADDING, "revision": marker_index},
                        monotonic_ns=clock + (50 + marker_index) * 1_000_000,
                    )
                )
            emit(
                EventDraft(
                    event_type=EventType.MODEL_RUN_FINISHED,
                    actor=SYSTEM_ACTOR,
                    turn_id=turn_id,
                    operation_id=run_id,
                    payload={"outcome": "completed", "duration_ms": 150},
                    measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
                    monotonic_ns=clock + 150_000_000,
                )
            )
            emit(
                EventDraft(
                    event_type=EventType.TURN_FINISHED,
                    actor=SYSTEM_ACTOR,
                    turn_id=turn_id,
                    payload={"end_reason": "cancelled", "duration_ms": 0},
                    monotonic_ns=clock + 150_000_000,
                )
            )


def _write_segmented_fixture(path: Path) -> None:
    sequence = 0
    identifier = 10
    actor = Actor(kind=ActorKind.AGENT, role=ActorRole.MAIN, actor_id=_identifier(1))
    chain = RevisionChain()
    item_ids = [_identifier(index) for index in range(100_000, 100_020)]

    with path.open("wb") as handle:

        def emit(draft: EventDraft) -> None:
            nonlocal sequence
            sequence += 1
            handle.write(
                encode_event_line(
                    build_event(
                        draft,
                        sequence=sequence,
                        runtime_id=RUNTIME_ID,
                        coverage_id=COVERAGE_ID,
                        session_id=SESSION_ID,
                        branch_id=BRANCH_ID,
                    )
                )
            )

        emit(
            EventDraft(
                event_type=EventType.COVERAGE_STARTED,
                actor=SYSTEM_ACTOR,
                payload={"coverage_reason": "session_started"},
                monotonic_ns=0,
            )
        )
        turn_number = 0
        while handle.tell() < _FIXTURE_BYTES:
            turn_number += 1
            turn_id = _identifier(identifier)
            identifier += 1
            clock = turn_number * 1_000_000_000
            emit(
                EventDraft(
                    event_type=EventType.TURN_STARTED,
                    actor=actor,
                    turn_id=turn_id,
                    payload={"turn_number": turn_number},
                    monotonic_ns=clock,
                )
            )
            preamble_id = _identifier(identifier)
            identifier += 1
            emit(
                EventDraft(
                    event_type=EventType.PREPARATION_STARTED,
                    actor=actor,
                    turn_id=turn_id,
                    operation_id=preamble_id,
                    payload={"scope": "turn_preamble", "phase": "turn_dispatch"},
                    monotonic_ns=clock,
                )
            )
            emit(
                EventDraft(
                    event_type=EventType.PREPARATION_FINISHED,
                    actor=actor,
                    turn_id=turn_id,
                    operation_id=preamble_id,
                    payload={
                        "scope": "turn_preamble",
                        "phase": "turn_dispatch",
                        "outcome": "handoff",
                        "duration_ms": 1,
                    },
                    measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
                    monotonic_ns=clock + 1_000_000,
                )
            )
            run_id = _identifier(identifier)
            identifier += 1
            emit(
                EventDraft(
                    event_type=EventType.MODEL_RUN_STARTED,
                    actor=actor,
                    turn_id=turn_id,
                    operation_id=run_id,
                    links=(Link(relation=LinkRelation.CAUSED_BY, target_operation_id=preamble_id),),
                    monotonic_ns=clock + 1_000_000,
                )
            )
            for revision_index in range(15):
                cycle_id = _identifier(identifier)
                exchange_id = _identifier(identifier + 1)
                segment_group_id = _identifier(identifier + 2)
                identifier += 3
                item_ids.pop(0)
                item_ids.append(_identifier(100_020 + turn_number * 15 + revision_index))
                plan = chain.plan(membership_of(item_ids))
                revision_payload = {
                    "revision_id": plan.revision_id,
                    "parent_revision_id": plan.parent_revision_id,
                    "item_count": plan.item_count,
                    "untokenized_item_count": 0,
                    "unidentified_item_count": 0,
                    "is_checkpoint": plan.is_checkpoint,
                    "token_buckets": {
                        "system": 1200,
                        "live_history": 4800,
                        "compressed_summaries": 900,
                        "tool_results": 2100,
                        "current_user": 350,
                    },
                    "tokenizer_fingerprint": "chrys-estimator-v1",
                }
                start = clock + (2 + revision_index * 8) * 1_000_000
                emit(
                    EventDraft(
                        event_type=EventType.MODEL_CYCLE_STARTED,
                        actor=actor,
                        turn_id=turn_id,
                        operation_id=cycle_id,
                        parent_operation_id=run_id,
                        monotonic_ns=start,
                    )
                )
                revision = EventDraft(
                    event_type=EventType.CONTEXT_REVISION_RECORDED,
                    actor=actor,
                    turn_id=turn_id,
                    operation_id=plan.revision_id,
                    parent_operation_id=exchange_id,
                    payload=revision_payload,
                    segmented_fields=(
                        SegmentedField(
                            field_pointer="/payload/refs",
                            segment_group_id=segment_group_id,
                            segment_count=1,
                        ),
                    ),
                    monotonic_ns=start,
                )
                emit(revision)
                emit(
                    EventDraft(
                        event_type=EventType.SEGMENT,
                        actor=actor,
                        turn_id=turn_id,
                        payload={
                            "parent_event_id": revision.event_id,
                            "field_pointer": "/payload/refs",
                            "segment_group_id": segment_group_id,
                            "segment_index": 0,
                            "segment_count": 1,
                            "encoding": "array_slice",
                            "entries": list(plan.entries),
                        },
                        monotonic_ns=start,
                    )
                )
                chain.commit(plan)
                emit(
                    EventDraft(
                        event_type=EventType.MODEL_EXCHANGE_STARTED,
                        actor=actor,
                        turn_id=turn_id,
                        operation_id=exchange_id,
                        parent_operation_id=cycle_id,
                        payload={"context_revision_id": plan.revision_id},
                        monotonic_ns=start,
                    )
                )
                emit(
                    EventDraft(
                        event_type=EventType.MODEL_EXCHANGE_FINISHED,
                        actor=actor,
                        turn_id=turn_id,
                        operation_id=exchange_id,
                        parent_operation_id=cycle_id,
                        payload={
                            "outcome": "success",
                            "duration_ms": 4,
                            "usage": {"normalized": {"input_total": 10_000, "output_total": 500}},
                        },
                        measurements={
                            "/payload/duration_ms": measurement("monotonic_clock", method_version=1),
                            "/payload/usage/normalized/input_total": measurement("provider", adapter_version=1),
                            "/payload/usage/normalized/output_total": measurement("provider", adapter_version=1),
                        },
                        monotonic_ns=start + 4_000_000,
                    )
                )
                emit(
                    EventDraft(
                        event_type=EventType.MODEL_CYCLE_FINISHED,
                        actor=actor,
                        turn_id=turn_id,
                        operation_id=cycle_id,
                        parent_operation_id=run_id,
                        payload={
                            "outcome": "success",
                            "duration_ms": 4,
                            "final_exchange_operation_id": exchange_id,
                        },
                        measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
                        monotonic_ns=start + 4_000_000,
                    )
                )
            emit(
                EventDraft(
                    event_type=EventType.MODEL_RUN_FINISHED,
                    actor=actor,
                    turn_id=turn_id,
                    operation_id=run_id,
                    payload={"outcome": "completed", "duration_ms": 125},
                    measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
                    monotonic_ns=clock + 125_000_000,
                )
            )
            emit(
                EventDraft(
                    event_type=EventType.TURN_FINISHED,
                    actor=actor,
                    turn_id=turn_id,
                    payload={"end_reason": "cancelled", "duration_ms": 0},
                    monotonic_ns=clock + 125_000_000,
                )
            )


def _probe(path: Path) -> dict[str, int | float]:
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = baseline
    stop = threading.Event()

    def sample() -> None:
        nonlocal peak
        while not stop.wait(0.002):
            peak = max(peak, process.memory_info().rss)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    started = perf_counter()
    try:
        analysis = analyze_trajectory(path)
        peak = max(peak, process.memory_info().rss)
    finally:
        stop.set()
        monitor.join()
    assert analysis.overview is not None
    return {
        "fixture_bytes": path.stat().st_size,
        "line_count": analysis.diagnostics.line_count,
        "turn_count": len(analysis.turns),
        "finding_count": len(analysis.findings),
        "seconds": perf_counter() - started,
        "baseline_rss_bytes": baseline,
        "peak_rss_bytes": peak,
        "resident_growth_bytes": peak - baseline,
        "resident_ceiling_bytes": MAX_RESIDENT_MEMORY_BYTES,
    }


def _diamond_probe() -> dict[str, int | float | bool | None]:
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = baseline
    stop = threading.Event()

    def sample() -> None:
        nonlocal peak
        while not stop.wait(0.002):
            peak = max(peak, process.memory_info().rss)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    intervals, edges, terminal = _layered_diamond(16)
    started = perf_counter()
    try:
        result = _longest_interval_path(intervals, edges, root_id="root", terminal_id=terminal)
        peak = max(peak, process.memory_info().rss)
    finally:
        stop.set()
        monitor.join()
    return {
        "seconds": perf_counter() - started,
        "baseline_rss_bytes": baseline,
        "peak_rss_bytes": peak,
        "resident_growth_bytes": peak - baseline,
        "resident_ceiling_bytes": MAX_RESIDENT_MEMORY_BYTES,
        "acyclic": result.acyclic,
        "bounded": result.bounded,
        "value": result.value,
    }


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--diamond-probe":
        sys.stdout.write(json.dumps(_diamond_probe()))
        return 0
    if len(sys.argv) == 3 and sys.argv[1] == "--probe":
        sys.stdout.write(json.dumps(_probe(Path(sys.argv[2]))))
        return 0
    with tempfile.TemporaryDirectory(prefix="chrys-trajectory-memory-") as directory:
        path = Path(directory) / "events.jsonl"
        _write_fixture(path)
        result = _run_isolated_probe(path)
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    growth = int(result["resident_growth_bytes"])
    return 0 if growth <= MAX_RESIDENT_MEMORY_BYTES else 1


def _run_isolated_probe(path: Path) -> dict[str, int | float]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.service.analytics.test_aggregation_memory_acceptance",
            "--probe",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    return json.loads(completed.stdout)


def _run_isolated_diamond_probe() -> dict[str, int | float | bool | None]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.service.analytics.test_aggregation_memory_acceptance",
            "--diamond-probe",
        ],
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    return json.loads(completed.stdout)


# The residency probes run in a fresh subprocess; on a machine saturated by
# the full parallel suite that child is CPU-starved and the parent's wall
# time can blow through the global 60s budget, which kills the whole xdist
# worker. The wider per-test budget keeps the hang guard without making
# honest load look like a crash.
@pytest.mark.timeout(180)
@pytest.mark.integration
def test_trajectory_intermediate_residency_stays_below_calibrated_ceiling(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    _write_fixture(path)

    result = _run_isolated_probe(path)

    assert int(result["fixture_bytes"]) >= _FIXTURE_BYTES
    assert int(result["finding_count"]) > 0
    assert int(result["resident_growth_bytes"]) < MAX_RESIDENT_MEMORY_BYTES


@pytest.mark.timeout(180)
@pytest.mark.integration
def test_segmented_revision_residency_stays_below_calibrated_ceiling(tmp_path: Path) -> None:
    path = tmp_path / "segmented-events.jsonl"
    _write_segmented_fixture(path)

    result = _run_isolated_probe(path)

    assert int(result["fixture_bytes"]) >= _FIXTURE_BYTES
    assert int(result["resident_growth_bytes"]) < MAX_RESIDENT_MEMORY_BYTES


@pytest.mark.timeout(180)
@pytest.mark.integration
def test_layered_diamond_resolution_stays_below_residency_ceiling() -> None:
    result = _run_isolated_diamond_probe()

    assert result["acyclic"] is True
    assert (result["bounded"] is True and result["value"] is not None) or (
        result["bounded"] is False and result["value"] is None
    )
    assert int(result["resident_growth_bytes"]) < MAX_RESIDENT_MEMORY_BYTES


def _layered_diamond(
    layer_count: int,
) -> tuple[dict[str, list[tuple[int, int]]], dict[str, set[str]], str]:
    intervals = {"root": [(0, 1)]}
    edges: dict[str, set[str]] = {}
    previous = "root"
    for layer in range(layer_count):
        left = f"left-{layer}"
        right = f"right-{layer}"
        join = f"join-{layer}"
        base = 1 + layer * 2
        intervals[left] = [(base, base + 1)]
        intervals[right] = [(base + 1, base + 2)]
        intervals[join] = []
        edges[previous] = {left, right}
        edges[left] = {join}
        edges[right] = {join}
        previous = join
    return intervals, edges, previous


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Bounded interval-union critical-path resolution and its candidate cap."""

from __future__ import annotations

from collections.abc import Iterable
from random import Random

import pytest

from chrys.service.analytics import (
    Precision,
    analyze_trajectory,
)
from chrys.service.analytics import _critical_path as critical_path_module
from chrys.service.analytics._critical_path import _longest_interval_path
from chrys.service.analytics.math import Interval
from tests.service.analytics._events import NS, EventLog, caused_by


def test_critical_path_prunes_interval_unions_dominated_at_the_same_node() -> None:
    intervals = {
        "root": [(0, 1)],
        "short": [(1, 2)],
        "long": [(1, 3)],
        "terminal": [],
    }
    edges = {"root": {"short", "long"}, "short": {"terminal"}, "long": {"terminal"}}

    result = _longest_interval_path(intervals, edges, root_id="root", terminal_id="terminal")

    assert result.acyclic is True
    assert result.bounded is True
    assert result.value == 3


def test_sixteen_layer_diamond_critical_path_stays_bounded() -> None:
    intervals, edges, terminal = _layered_diamond(16)

    result = _longest_interval_path(intervals, edges, root_id="root", terminal_id=terminal)

    assert result.acyclic is True
    if result.bounded:
        assert result.value is not None
    else:
        assert result.value is None


def test_disjoint_critical_path_is_certified_without_candidate_enumeration(monkeypatch) -> None:
    """A max-sum path whose intervals never overlap bounds every interval union."""
    monkeypatch.setattr(critical_path_module, "_MAX_CRITICAL_PATH_CANDIDATES", 0)
    intervals = {"root": [(0, 1)], "left": [(1, 3)], "right": [(1, 2)], "terminal": [(3, 4)]}
    edges = {"root": {"left", "right"}, "left": {"terminal"}, "right": {"terminal"}}

    result = _longest_interval_path(intervals, edges, root_id="root", terminal_id="terminal")

    assert (result.value, result.acyclic, result.bounded) == (4, True, True)


def test_completion_edges_carry_displaced_descendants_but_forks_do_not() -> None:
    intervals = {
        "exchange": [(0, 1)],
        "tool": [(1, 2)],
        "approval": [(2, 5)],
        "hook": [(1, 2)],
        "consumer": [(5, 6)],
    }
    edges = {"exchange": {"tool"}, "tool": {"approval", "hook", "consumer"}, "hook": set(), "approval": set()}
    parents = {"tool": "exchange", "approval": "tool"}

    through_consumer = _longest_interval_path(
        intervals,
        edges,
        parents=parents,
        fork_edges=frozenset({("tool", "hook")}),
        root_id="exchange",
        terminal_id="consumer",
    )
    through_fork = _longest_interval_path(
        intervals,
        edges,
        parents=parents,
        fork_edges=frozenset({("tool", "hook")}),
        root_id="exchange",
        terminal_id="hook",
    )

    assert (through_consumer.value, through_consumer.bounded) == (6, True)
    assert (through_fork.value, through_fork.bounded) == (2, True)


def test_long_overlapping_path_does_not_remerge_every_history_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    size = 128
    intervals = {str(index): [(0, 1), (3 * index + 2, 3 * index + 3)] for index in range(size)}
    edges = {str(index): {str(index + 1)} for index in range(size - 1)}
    merged_spans = 0
    real_union = critical_path_module.interval_union

    def count_union(members: Iterable[Interval]) -> tuple[Interval, ...]:
        nonlocal merged_spans
        materialized = tuple(members)
        merged_spans += len(materialized)
        return real_union(materialized)

    monkeypatch.setattr(critical_path_module, "interval_union", count_union)

    result = _longest_interval_path(intervals, edges, root_id="0", terminal_id=str(size - 1))

    assert (result.value, result.acyclic, result.bounded) == (size + 1, True, True)
    # Linear setup is allowed; repeatedly merging every growing prefix is not.
    assert merged_spans <= 12 * size


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("terminal", [None, "6"])
def test_overlapping_paths_match_exhaustive_nanosecond_union(seed: int, terminal: str | None) -> None:
    rng = Random(seed)
    intervals: dict[str, list[Interval]] = {}
    edges: dict[str, set[str]] = {}
    parents = {"2": "1", "3": "2"}
    forks = frozenset({("1", "4")})
    for index in range(7):
        start = rng.randrange(-10, 10)
        intervals[str(index)] = [(start, start + rng.randrange(0, 8)), (30 + index, 32 + index)]
        edges[str(index)] = {str(target) for target in range(index + 1, 7) if rng.random() < 0.6}
    # Exercise descendant carrying and a fork edge, including their overlaps.
    edges["0"].add("1")
    edges["1"].update(("2", "4"))
    edges["2"].add("3")
    edges["3"].add("6")
    samples = {node: {ns for start, end in members for ns in range(start, end)} for node, members in intervals.items()}

    def descendants(node: str) -> set[int]:
        covered: set[int] = set()
        for child, parent in parents.items():
            if parent == node:
                covered.update(samples[child], descendants(child))
        return covered

    def enumerate_paths(node: str, covered: set[int]) -> list[int]:
        covered = covered | samples[node]
        values = [len(covered)] if terminal is None or node == terminal else []
        for target in edges[node]:
            crossing = set() if parents.get(target) == node or (node, target) in forks else descendants(node)
            values.extend(enumerate_paths(target, covered | crossing))
        return values

    expected = max(enumerate_paths("0", set()))
    result = _longest_interval_path(
        intervals, edges, parents=parents, fork_edges=forks, root_id="0", terminal_id=terminal
    )

    assert (result.value, result.acyclic, result.bounded) == (expected, True, True)


def test_critical_path_candidate_cap_degrades_metrics_with_diagnostic(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(critical_path_module, "_MAX_CRITICAL_PATH_CANDIDATES", 0)
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
    # The waited hook overlaps its fork origin, so no disjoint witness exists
    # and the bounded enumeration must run.
    log.span(
        "hook.operation",
        "e" * 32,
        8 * NS,
        13 * NS,
        start_payload={
            "hook_event": "after_turn",
            "execution_mode": "async",
            "scope": "turn",
            "target_operation_id": "d" * 32,
        },
        links=caused_by("d" * 32),
    )
    log.add("turn.finished", 10 * NS, payload={"end_reason": "completed", "duration_ms": 0})
    log.settled(13 * NS, waited_hook_ids=["e" * 32])
    path = tmp_path / "critical-path-cap.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.compute_cp_ns.precision is Precision.UNRESOLVED
    assert turn.response_cp_ns.precision is Precision.UNRESOLVED
    assert "compute critical-path candidate cap exceeded" in turn.diagnostics
    assert "response critical-path candidate cap exceeded" in turn.diagnostics


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

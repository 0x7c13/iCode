# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Bounded interval-union critical-path resolution."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from threading import Event
from typing import Final

from chrys.service.analytics.math import interval_length, interval_union
from chrys.service.analytics.reader import raise_if_cancelled as _check_cancelled

_MAX_CRITICAL_PATH_CANDIDATES: Final = 4096
"""Hard bound on retained interval histories in one critical-path resolution."""


@dataclass(frozen=True, slots=True)
class _PathResolution:
    value: int | None
    acyclic: bool
    bounded: bool


def _longest_interval_path(
    intervals: dict[str, list[tuple[int, int]]],
    edges: dict[str, set[str]],
    *,
    parents: dict[str, str] | None = None,
    fork_edges: frozenset[tuple[str, str]] = frozenset(),
    root_id: str | None = None,
    terminal_id: str | None = None,
    cancel_event: Event | None = None,
) -> _PathResolution:
    """Resolve the path whose interval union is longest.

    A causal edge that leaves a container (neither a displacing parent→child
    edge nor a concurrent-hook fork) depends on the container *completing*, so
    the walk carries the container's displaced descendants across that edge:
    a consumer that waits on a tool also waited for the approval and the
    sub-agent the tool displaced.

    Interval unions make the exact problem combinatorial, so a max-sum walk
    runs first: when the best-sum path's intervals are pairwise disjoint its
    sum equals its union and bounds every other path's union, which certifies
    it. Only overlapping paths fall back to the bounded non-dominated search.
    """
    vertices = set(intervals)
    for source, targets in edges.items():
        _check_cancelled(cancel_event)
        vertices.add(source)
        vertices.update(targets)
    indegree = dict.fromkeys(vertices, 0)
    for targets in edges.values():
        for target in targets:
            indegree[target] += 1
    roots = [vertex for vertex, degree in indegree.items() if degree == 0]
    ready = list(roots)
    order: list[str] = []
    while ready:
        _check_cancelled(cancel_event)
        vertex = ready.pop()
        order.append(vertex)
        for target in edges.get(vertex, ()):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if len(order) != len(vertices):
        return _PathResolution(None, False, True)

    displacing_parents = parents or {}
    children: dict[str, list[str]] = defaultdict(list)
    for child, parent in displacing_parents.items():
        children[parent].append(child)
    completions: dict[str, tuple[tuple[int, int], ...]] = {}

    def carried(source: str, target: str) -> tuple[tuple[int, int], ...]:
        if displacing_parents.get(target) == source or (source, target) in fork_edges:
            return ()
        completion = completions.get(source)
        if completion is None:
            members: list[tuple[int, int]] = []
            seen: set[str] = set()
            pending = list(children.get(source, ()))
            while pending:
                member = pending.pop()
                if member in seen:
                    continue
                seen.add(member)
                members.extend(intervals.get(member, ()))
                pending.extend(children.get(member, ()))
            completion = completions[source] = interval_union(members)
        return completion

    starts = roots if root_id is None else [root_id] if root_id in vertices else []
    lengths = {vertex: interval_length(interval_union(intervals.get(vertex, ()))) for vertex in vertices}
    sums: dict[str, int] = {vertex: lengths[vertex] for vertex in starts}
    predecessors: dict[str, str] = {}
    for vertex in order:
        _check_cancelled(cancel_event)
        value = sums.get(vertex)
        if value is None:
            continue
        for target in edges.get(vertex, ()):
            candidate = value + interval_length(carried(vertex, target)) + lengths[target]
            if candidate > sums.get(target, -1):
                sums[target] = candidate
                predecessors[target] = vertex
    end = terminal_id if terminal_id is not None else max(sums, key=sums.__getitem__, default=None)
    if end is None:
        return _PathResolution(0, True, True)
    if end not in sums:
        return _PathResolution(None, True, True)
    witness: list[tuple[int, int]] = list(intervals.get(end, ()))
    cursor = end
    while cursor in predecessors:
        previous = predecessors[cursor]
        witness.extend(carried(previous, cursor))
        witness.extend(intervals.get(previous, ()))
        cursor = previous
    if interval_length(interval_union(witness)) == sums[end]:
        return _PathResolution(sums[end], True, True)

    paths: dict[str, set[tuple[tuple[int, int], ...]]] = defaultdict(set)
    retained = 0
    for vertex in starts:
        paths[vertex].add(interval_union(intervals.get(vertex, ())))
        retained += 1
    if retained > _MAX_CRITICAL_PATH_CANDIDATES:
        return _PathResolution(None, True, False)
    best = 0
    terminal_value: int | None = None
    for vertex in order:
        _check_cancelled(cancel_event)
        candidates = paths.get(vertex, set())
        if candidates:
            candidate_max = max(interval_length(path) for path in candidates)
            best = max(best, candidate_max)
            if vertex == terminal_id:
                terminal_value = candidate_max
        for target in edges.get(vertex, ()):
            target_paths = paths[target]
            crossing = carried(vertex, target)
            for path in candidates:
                candidate = interval_union((*path, *crossing, *intervals.get(target, ())))
                retained += _retain_nondominated(target_paths, candidate)
                if retained > _MAX_CRITICAL_PATH_CANDIDATES:
                    return _PathResolution(None, True, False)
        retained -= len(candidates)
        paths.pop(vertex, None)
    if terminal_id is not None:
        return _PathResolution(terminal_value, True, True)
    return _PathResolution(best, True, True)


def _retain_nondominated(
    retained: set[tuple[tuple[int, int], ...]],
    candidate: tuple[tuple[int, int], ...],
) -> int:
    if any(_intervals_subset(candidate, existing) for existing in retained):
        return 0
    dominated = {existing for existing in retained if _intervals_subset(existing, candidate)}
    retained.difference_update(dominated)
    retained.add(candidate)
    return 1 - len(dominated)


def _intervals_subset(
    subset: tuple[tuple[int, int], ...],
    superset: tuple[tuple[int, int], ...],
) -> bool:
    superset_index = 0
    for start, end in subset:
        while superset_index < len(superset) and superset[superset_index][1] <= start:
            superset_index += 1
        if superset_index >= len(superset) or superset[superset_index][0] > start or superset[superset_index][1] < end:
            return False
    return True

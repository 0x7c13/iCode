# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Interval and dependency math shared by trajectory aggregators."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

type Interval = tuple[int, int]


def interval_union(intervals: Iterable[Interval]) -> tuple[Interval, ...]:
    """Return sorted disjoint half-open intervals, discarding empty inputs."""
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return ()
    merged: list[Interval] = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)


def interval_length(intervals: Iterable[Interval]) -> int:
    """Return the union length of *intervals*."""
    return sum(end - start for start, end in interval_union(intervals))


def clip_interval(interval: Interval, bounds: Interval) -> Interval | None:
    """Clip one half-open interval to *bounds*, returning ``None`` when empty."""
    start = max(interval[0], bounds[0])
    end = min(interval[1], bounds[1])
    return (start, end) if end > start else None


def subtract_intervals(base: Interval, removed: Iterable[Interval]) -> tuple[Interval, ...]:
    """Subtract the clipped union of *removed* from *base*."""
    clipped = [part for interval in removed if (part := clip_interval(interval, base)) is not None]
    cursor = base[0]
    remaining: list[Interval] = []
    for start, end in interval_union(clipped):
        if start > cursor:
            remaining.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < base[1]:
        remaining.append((cursor, base[1]))
    return tuple(remaining)


def longest_weighted_path(
    weights: Mapping[str, int],
    edges: Mapping[str, set[str]],
) -> tuple[int | None, bool]:
    """Return the longest node-weighted DAG path and whether the graph is acyclic."""
    vertices = set(weights)
    for source, targets in edges.items():
        vertices.add(source)
        vertices.update(targets)
    indegree = dict.fromkeys(vertices, 0)
    for targets in edges.values():
        for target in targets:
            indegree[target] += 1
    ready = [vertex for vertex, degree in indegree.items() if degree == 0]
    distances = {vertex: weights.get(vertex, 0) for vertex in vertices}
    visited = 0
    while ready:
        vertex = ready.pop()
        visited += 1
        for target in edges.get(vertex, set()):
            distances[target] = max(distances[target], distances[vertex] + weights.get(target, 0))
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if visited != len(vertices):
        return None, False
    return max(distances.values(), default=0), True

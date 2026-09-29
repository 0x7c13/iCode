# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Pure arithmetic helpers for the GC-freeze calibration gate."""

from __future__ import annotations

from collections.abc import Iterable
from math import ceil


def validated_absorb_points(points: Iterable[int], *, max_absorbs: int) -> tuple[int, ...]:
    """Normalize calibration points that can complete as actual absorbs."""
    normalized = tuple(sorted(set(points)))
    if max_absorbs < 0:
        raise ValueError("max_absorbs must be non-negative")
    if not normalized or normalized[0] != 0 or any(point < 0 or point > max_absorbs for point in normalized):
        raise ValueError(f"--absorb-points must contain 0 and values in [0, {max_absorbs}]")
    return normalized


def minimum_deferred_diff_surfaces(turns: int) -> int:
    """Return the 90% settle threshold while tolerating at least one worker tail."""
    if turns < 1:
        raise ValueError("turns must be positive")
    expected = turns * 2
    return min(ceil(expected * 0.9), expected - 1)


def dead_cyclic_fraction(
    *,
    collected_objects: int,
    zero_collected: int,
    frozen_objects_added_by_absorbs: int,
) -> float | None:
    """Return dead cyclic objects as a fraction of the already-inclusive freeze delta."""
    excess_dead = max(0, collected_objects - zero_collected)
    if frozen_objects_added_by_absorbs <= 0:
        return 1.0 if excess_dead else None
    return excess_dead / frozen_objects_added_by_absorbs


def aggregate_dead_cyclic_fraction(
    cohorts: Iterable[tuple[int, int, int]],
) -> float | None:
    """Combine matched intervals before applying the direct dead/additions ratio."""
    total_excess_dead = 0
    total_frozen_additions = 0
    for collected_objects, zero_collected, frozen_additions in cohorts:
        total_excess_dead += max(0, collected_objects - zero_collected)
        total_frozen_additions += max(0, frozen_additions)
    return dead_cyclic_fraction(
        collected_objects=total_excess_dead,
        zero_collected=0,
        frozen_objects_added_by_absorbs=total_frozen_additions,
    )


def latency_in_unfrozen_collects(action_ms: float, *, unfrozen_collect_ms: float) -> float:
    """Express an action's latency in bare full collections of a matched unfrozen heap.

    The basis is what one ``gc.collect()`` costs with no freeze machinery at all, so no
    choice the freeze design makes (cache hooks, the reflows they force) can move it. A
    basis that runs the design's own hooks instead rewards overhead shared by both sides:
    dropping a cost ``c`` from ``(a + c) / (b + c)`` raises the ratio although the action
    got faster.
    """
    if unfrozen_collect_ms <= 0:
        raise ValueError("unfrozen_collect_ms must be positive")
    return action_ms / unfrozen_collect_ms

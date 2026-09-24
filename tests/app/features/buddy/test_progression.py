# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The experience curve: what work is worth, what a level costs, and where a buddy stops."""

from __future__ import annotations

import pytest

from chrys.app.features.buddy.progression import (
    FINAL_LEVEL,
    LEVELS_PER_STAGE,
    PETS_COUNTED_PER_TURN,
    XP_PER_PET,
    XP_PER_TURN,
    Progress,
    earned_xp,
    progress,
    xp_for_level,
)


def _stage_cost(stage: int) -> int:
    return sum(xp_for_level(level, stage) for level in range(1, LEVELS_PER_STAGE + 1))


def test_the_pets_a_turn_pays_for_are_worth_no_more_than_the_turn() -> None:
    assert earned_xp(turns=1, pets=0) == XP_PER_TURN
    assert earned_xp(turns=0, pets=1) == XP_PER_PET
    assert XP_PER_PET * PETS_COUNTED_PER_TURN <= XP_PER_TURN


def test_petting_without_working_stops_paying() -> None:
    ceiling = XP_PER_PET * PETS_COUNTED_PER_TURN
    assert earned_xp(turns=0, pets=PETS_COUNTED_PER_TURN) == ceiling
    assert earned_xp(turns=0, pets=10_000) == ceiling
    # Every turn worked makes room for another handful of pets.
    assert earned_xp(turns=3, pets=10_000) == 3 * XP_PER_TURN + 4 * ceiling


def test_levels_get_dearer_within_a_stage_and_from_stage_to_stage() -> None:
    costs = [xp_for_level(level, 0) for level in range(1, LEVELS_PER_STAGE + 1)]
    assert costs == sorted(costs)
    assert costs[0] < costs[-1]
    assert all(xp_for_level(level, 1) > xp_for_level(level, 0) for level in range(1, LEVELS_PER_STAGE + 1))


def test_the_first_stage_takes_well_over_a_thousand_turns() -> None:
    """The curve this one replaced evolved a buddy after 990 turns; this one is meant to be slower."""
    turns = _stage_cost(0) / XP_PER_TURN
    assert 1_300 <= turns <= 1_600
    assert [_stage_cost(stage) for stage in range(4)] == sorted(_stage_cost(stage) for stage in range(4))


def test_a_new_buddy_starts_at_level_one_with_nothing_earned() -> None:
    assert progress(0, final_stage=3) == Progress(stage=0, level=1, xp_into_level=0, xp_for_level=xp_for_level(1, 0))


def test_a_level_is_reached_on_exactly_its_last_point() -> None:
    cost = xp_for_level(1, 0)
    assert progress(cost - 1, final_stage=3) == Progress(0, 1, cost - 1, cost)
    assert progress(cost, final_stage=3) == Progress(0, 2, 0, xp_for_level(2, 0))


def test_finishing_a_stage_evolves_a_buddy_that_has_a_tier_left_to_reach() -> None:
    stage = _stage_cost(0)
    assert progress(stage - 1, final_stage=1).stage == 0
    assert progress(stage - 1, final_stage=1).level == LEVELS_PER_STAGE
    assert progress(stage, final_stage=1) == Progress(1, 1, 0, xp_for_level(1, 1))


def test_the_last_stage_ends_at_the_final_level_and_stays_there() -> None:
    stage = _stage_cost(0)
    top = progress(stage, final_stage=0)
    assert top == Progress(stage=0, level=FINAL_LEVEL, xp_into_level=0, xp_for_level=0)
    assert top.maxed
    assert progress(stage * 1_000, final_stage=0) == top
    assert not progress(stage - 1, final_stage=0).maxed


@pytest.mark.parametrize("final_stage", [0, 1, 2, 3])
def test_every_point_earned_is_accounted_for(final_stage: int) -> None:
    total = sum(_stage_cost(stage) for stage in range(final_stage + 1))
    for xp in (0, 1, total // 3, total // 2, total - 1):
        reached = progress(xp, final_stage)
        spent = sum(_stage_cost(stage) for stage in range(reached.stage))
        spent += sum(xp_for_level(level, reached.stage) for level in range(1, reached.level))
        assert spent + reached.xp_into_level == xp
        assert 0 <= reached.xp_into_level < reached.xp_for_level

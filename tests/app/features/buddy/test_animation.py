# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Pixel animation keeps resting, blinking, and petting poses in their own ranges."""

from itertools import groupby, pairwise

import pytest

from chrys.app.features.buddy.animation import FRAME_COUNT, IDLE_FRAME_COUNT, get_idle_frame, get_pet_frame
from chrys.app.features.buddy.model import Species
from chrys.app.features.buddy.pixel_sprites import species_sprite

# Long enough for the slowest species to stretch many times and for blinks to have fallen everywhere.
_TICKS = 2_000


def _idle(species: Species) -> list[tuple[int, bool]]:
    return [get_idle_frame(species, tick) for tick in range(_TICKS)]


@pytest.mark.parametrize("species", list(Species))
def test_every_species_starts_at_rest_and_uses_all_of_its_idle_poses_and_no_other(species: Species) -> None:
    idle = _idle(species)

    assert idle[0] == (0, False)
    assert len(species_sprite(species).frames) == IDLE_FRAME_COUNT
    assert {pose for pose, _ in idle} == set(range(IDLE_FRAME_COUNT))


@pytest.mark.parametrize("species", list(Species))
def test_every_stretch_holds_pose_one_then_pose_two_equally_long_and_returns_to_rest(species: Species) -> None:
    runs = [(pose, len(list(ticks))) for pose, ticks in groupby(pose for pose, _ in _idle(species))]
    runs.pop()  # the window cuts the last run short
    whole = runs[: len(runs) - len(runs) % 3]

    assert [pose for pose, _ in whole] == [0, 1, 2] * (len(whole) // 3)
    rests, first_holds, second_holds = ({length for _, length in whole[offset::3]} for offset in range(3))
    assert len(rests) == 1
    assert first_holds == second_holds
    assert len(first_holds) == 1
    assert min(rests) > max(first_holds)  # longer at rest than in either pose


@pytest.mark.parametrize("species", list(Species))
def test_blinks_keep_their_own_time_and_so_fall_on_every_pose(species: Species) -> None:
    idle = _idle(species)
    blink_ticks = [tick for tick, (_, blinking) in enumerate(idle) if blinking]

    assert len({later - earlier for earlier, later in pairwise(blink_ticks)}) == 1
    assert {idle[tick][0] for tick in blink_ticks} == set(range(IDLE_FRAME_COUNT))
    # One tick long, and rare: eyes are open nearly all the time.
    assert all(later - earlier > 2 * IDLE_FRAME_COUNT for earlier, later in pairwise(blink_ticks))


def test_restless_species_change_pose_more_often_than_placid_ones() -> None:
    def pose_changes(species: Species) -> int:
        return sum(earlier != later for (earlier, _), (later, _) in pairwise(_idle(species)))

    assert (
        pose_changes(Species.BEE) > pose_changes(Species.FOX) > pose_changes(Species.CAT) > pose_changes(Species.SNAIL)
    )
    assert pose_changes(Species.SHARK) > pose_changes(Species.TURTLE)


def test_petting_cycles_all_three_poses_without_idle_pauses() -> None:
    assert [get_pet_frame(tick) for tick in range(9)] == [3, 4, 5, 3, 4, 5, 3, 4, 5]
    assert {get_pet_frame(tick) for tick in range(30)} == set(range(IDLE_FRAME_COUNT, FRAME_COUNT))

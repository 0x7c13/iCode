# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""What a user can do to the buddy, against the save file under the test's own config directory."""

from __future__ import annotations

import random
import threading
from random import Random
from typing import TYPE_CHECKING

from chrys.app.features.buddy import actions
from chrys.app.features.buddy.hatchery import hatchling
from chrys.app.features.buddy.model import NAME_MAX_LENGTH
from chrys.app.features.buddy.progression import XP_PER_PET, XP_PER_TURN
from chrys.app.features.buddy.store import BuddyStore
from tests.support.threads import run_to_the_end

if TYPE_CHECKING:
    import pytest

# One budget for every wait of a test: far more than six hatches need, and well inside the per-test timeout.
_WAIT_SECONDS = 20


def test_there_is_no_buddy_until_one_hatches() -> None:
    assert actions.current_buddy() is None
    assert actions.rename("Pico") is None
    assert actions.set_muted(True) is None
    assert actions.record_pet() is None
    assert actions.record_turn() is None
    # Most users never hatch one: for them a finished turn leaves no lock file and no directory behind.
    assert not BuddyStore().path.parent.exists()


def test_hatching_saves_the_buddy_that_was_drawn() -> None:
    buddy = actions.hatch(Random(11))

    assert buddy.record == hatchling(Random(11), now=buddy.record.hatched_at)
    assert actions.current_buddy() == buddy
    assert (buddy.level, buddy.record.turns, buddy.record.pets) == (1, 0, 0)


def test_a_second_hatch_returns_the_buddy_that_already_exists() -> None:
    first = actions.hatch(Random(1))
    assert actions.hatch(Random(2)) == first
    assert actions.current_buddy() == first


def test_hatching_without_a_generator_draws_from_the_operating_system(monkeypatch: pytest.MonkeyPatch) -> None:
    assert actions.SystemRandom is random.SystemRandom
    asked: list[Random] = []

    def system_random() -> Random:
        asked.append(Random(11))
        return asked[-1]

    monkeypatch.setattr(actions, "SystemRandom", system_random)

    buddy = actions.hatch()

    assert len(asked) == 1
    assert buddy.record == hatchling(Random(11), now=buddy.record.hatched_at)
    assert actions.current_buddy() == buddy


def test_instances_hatching_at_once_end_up_with_one_buddy() -> None:
    callers = 6
    start = threading.Barrier(callers)
    hatched = []

    def worker(seed: int) -> None:
        start.wait(_WAIT_SECONDS)
        hatched.append(actions.hatch(Random(seed)))

    run_to_the_end([threading.Thread(target=worker, args=(seed,)) for seed in range(callers)], within=_WAIT_SECONDS)

    assert len(hatched) == callers
    assert all(buddy == actions.current_buddy() for buddy in hatched)


def test_work_and_pets_are_counted_and_earn_experience() -> None:
    actions.hatch(Random(3))

    assert actions.record_turn().record.turns == 1
    assert actions.record_pet().record.pets == 1
    buddy = actions.current_buddy()
    assert (buddy.record.turns, buddy.record.pets) == (1, 1)
    assert buddy.progress.xp_into_level == XP_PER_TURN + XP_PER_PET


def test_renaming_cleans_the_name_and_ignores_an_empty_one() -> None:
    hatched = actions.hatch(Random(4))

    assert actions.rename("  Sir \n Quacks ").name == "Sir Quacks"
    assert actions.rename(" \t ").name == "Sir Quacks"
    assert len(actions.rename("z" * 100).name) == NAME_MAX_LENGTH
    assert actions.current_buddy().record.species is hatched.species


def test_muting_is_saved_and_changes_nothing_else() -> None:
    hatched = actions.hatch(Random(5))

    assert actions.set_muted(True).muted
    assert actions.current_buddy().muted
    assert actions.set_muted(False) == hatched

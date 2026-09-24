# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Hatching is one random draw: reproducible from its generator, and from nothing else."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from random import Random

from chrys.app.features.buddy.hatchery import (
    NAMES,
    RARITY_ODDS,
    SHINY_ODDS,
    SPECIES_RARITY,
    TRAIT_POINTS,
    hatchling,
)
from chrys.app.features.buddy.model import TRAIT_MAX, TRAIT_MIN, BuddyRecord, Rarity, Species, Trait

_DRAWS = 4_000


def _hatchlings(seed: int = 20260922, count: int = _DRAWS) -> list[BuddyRecord]:
    rng = Random(seed)
    return [hatchling(rng) for _ in range(count)]


def test_the_tables_cover_every_species_and_tier() -> None:
    assert list(SPECIES_RARITY) == list(Species)
    assert set(SPECIES_RARITY.values()) == set(Rarity) == set(RARITY_ODDS) == set(TRAIT_POINTS)
    assert sum(RARITY_ODDS.values()) == 100
    assert list(RARITY_ODDS.values()) == sorted(RARITY_ODDS.values(), reverse=True)
    assert list(TRAIT_POINTS.values()) == sorted(TRAIT_POINTS.values())
    assert max(TRAIT_POINTS.values()) / 2 <= TRAIT_MAX


def test_the_same_generator_hatches_the_same_buddy() -> None:
    now = datetime(2026, 9, 22, tzinfo=UTC)
    assert hatchling(Random(7), now=now) == hatchling(Random(7), now=now)
    assert any(hatchling(Random(seed), now=now) != hatchling(Random(7), now=now) for seed in range(8, 40))


def test_a_hatchling_is_new_and_stamped_with_the_hatch_time() -> None:
    now = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)
    record = hatchling(Random(1), now=now)
    assert (record.hatched_at, record.turns, record.pets, record.muted) == (now, 0, 0, False)
    assert record.name in NAMES
    assert hatchling(Random(1)).hatched_at.tzinfo is UTC


def test_every_hatchling_is_a_species_of_its_own_tier() -> None:
    for record in _hatchlings():
        assert SPECIES_RARITY[record.species] is record.rarity


def test_every_species_can_hatch() -> None:
    assert {record.species for record in _hatchlings()} == set(Species)


def test_tiers_and_shinies_turn_up_about_as_often_as_the_odds_say() -> None:
    hatchlings = _hatchlings()
    tiers = Counter(record.rarity for record in hatchlings)
    for rarity, odds in RARITY_ODDS.items():
        assert abs(tiers[rarity] / _DRAWS - odds / 100) < 0.03
    shinies = sum(record.shiny for record in hatchlings)
    assert abs(shinies / _DRAWS - SHINY_ODDS) < 0.015


def test_traits_spend_the_tier_budget_unevenly_and_stay_in_range() -> None:
    for record in _hatchlings(count=500):
        assert set(record.traits) == set(Trait)
        assert all(TRAIT_MIN <= value <= TRAIT_MAX for value in record.traits.values())
        # Each share is rounded on its own, so the total may be off by a point or two.
        assert abs(sum(record.traits.values()) - TRAIT_POINTS[record.rarity]) <= len(Trait) / 2
    assert any(max(record.traits.values()) >= 2 * min(record.traits.values()) for record in _hatchlings(count=50))

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Hatching: one random draw, saved whole. Nothing about a buddy is derived from who the user is."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from chrys.app.features.buddy.model import BuddyRecord, Rarity, Species, Trait

if TYPE_CHECKING:
    from random import Random

RARITY_ODDS: dict[Rarity, int] = {Rarity.N: 55, Rarity.R: 30, Rarity.SR: 12, Rarity.SSR: 3}

SHINY_ODDS = 1 / 40

# The tier each species hatches in, by how far from the back garden you would have to go to meet one:
# N lives outside the window, R takes a trip, SR takes luck, SSR is not from around here at all.
# Evolution moves a buddy up; it never changes species.
SPECIES_RARITY: dict[Species, Rarity] = {
    Species.ALIEN: Rarity.SSR,
    Species.AXOLOTL: Rarity.SR,
    Species.BAT: Rarity.R,
    Species.BEE: Rarity.N,
    Species.CACTUS: Rarity.R,
    Species.CAPYBARA: Rarity.SR,
    Species.CAT: Rarity.N,
    Species.CRAB: Rarity.R,
    Species.DRAGON: Rarity.SSR,
    Species.DUCK: Rarity.N,
    Species.FOX: Rarity.R,
    Species.FROG: Rarity.N,
    Species.GHOST: Rarity.SSR,
    Species.GOOSE: Rarity.N,
    Species.JELLY: Rarity.SR,
    Species.LLAMA: Rarity.R,
    Species.MUSHROOM: Rarity.N,
    Species.OCTOPUS: Rarity.SR,
    Species.OWL: Rarity.R,
    Species.PANDA: Rarity.SR,
    Species.PENGUIN: Rarity.R,
    Species.RABBIT: Rarity.N,
    Species.ROBOT: Rarity.SSR,
    Species.SHARK: Rarity.R,
    Species.SNAIL: Rarity.N,
    Species.SNAKE: Rarity.N,
    Species.TABBY: Rarity.N,
    Species.TURTLE: Rarity.N,
}

# Trait points a hatchling of each tier has to share out. The largest possible share is half
# the budget (see _share_out), so no budget here can push a trait past its maximum.
TRAIT_POINTS: dict[Rarity, int] = {Rarity.N: 120, Rarity.R: 150, Rarity.SR: 180, Rarity.SSR: 200}

NAMES = ("Byte", "Dot", "Miso", "Mochi", "Nori", "Pico", "Pixel", "Sprout", "Tofu", "Widget")


def hatchling(rng: Random, *, now: datetime | None = None) -> BuddyRecord:
    """Draw a new buddy from *rng*."""
    rarity = rng.choices(list(RARITY_ODDS), weights=list(RARITY_ODDS.values()))[0]
    return BuddyRecord(
        species=rng.choice([species for species, home in SPECIES_RARITY.items() if home is rarity]),
        rarity=rarity,
        shiny=rng.random() < SHINY_ODDS,
        traits=_share_out(rng, TRAIT_POINTS[rarity]),
        name=rng.choice(NAMES),
        hatched_at=now if now is not None else datetime.now(UTC),
    )


def _share_out(rng: Random, points: int) -> dict[Trait, int]:
    """Split *points* between the traits unevenly, so every buddy leans some way.

    An appetite is at most three times another, which bounds a share between a tenth and a half of *points*.
    """
    appetites = {trait: rng.uniform(0.5, 1.5) for trait in Trait}
    unit = points / sum(appetites.values())
    return {trait: round(appetite * unit) for trait, appetite in appetites.items()}

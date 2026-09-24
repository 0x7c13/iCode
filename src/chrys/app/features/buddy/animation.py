# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Timing for the three pixel poses, independent of artwork and terminal rendering."""

from __future__ import annotations

from dataclasses import dataclass

from chrys.app.features.buddy.model import Species

IDLE_FRAME_COUNT = 3
FRAME_COUNT = IDLE_FRAME_COUNT * 2
# Eyelids keep their own time. The period shares no factor with any temperament's, so a blink lands on a
# different moment of the rhythm each time instead of becoming part of it.
_BLINK_EVERY_TICKS = 13


@dataclass(frozen=True)
class _Temperament:
    """How a species idles: it rests, then stretches through its two other poses, holding each one."""

    rest_ticks: int
    hold_ticks: int

    def pose(self, tick: int) -> int:
        """The idle pose at *tick*: 0 while resting, then 1 and 2 for *hold_ticks* each."""
        into_stretch = tick % (self.rest_ticks + 2 * self.hold_ticks) - self.rest_ticks
        return 0 if into_stretch < 0 else 1 + into_stretch // self.hold_ticks


_DARTING = _Temperament(rest_ticks=2, hold_ticks=1)
_LIVELY = _Temperament(rest_ticks=5, hold_ticks=1)
_EASY = _Temperament(rest_ticks=8, hold_ticks=2)
_SLOW = _Temperament(rest_ticks=14, hold_ticks=3)

# How restless each species is. The rhythm comes from the temperament, never from a per-species cue list.
_TEMPERAMENTS: dict[Species, _Temperament] = {
    Species.ALIEN: _LIVELY,
    Species.AXOLOTL: _EASY,
    Species.BAT: _LIVELY,
    Species.BEE: _DARTING,
    Species.CACTUS: _SLOW,
    Species.CAPYBARA: _SLOW,
    Species.CAT: _EASY,
    Species.CRAB: _LIVELY,
    Species.DRAGON: _LIVELY,
    Species.DUCK: _EASY,
    Species.FOX: _LIVELY,
    Species.FROG: _LIVELY,
    Species.GHOST: _EASY,
    Species.GOOSE: _EASY,
    Species.JELLY: _EASY,
    Species.LLAMA: _EASY,
    Species.MUSHROOM: _SLOW,
    Species.OCTOPUS: _LIVELY,
    Species.OWL: _EASY,
    Species.PANDA: _SLOW,
    Species.PENGUIN: _EASY,
    Species.RABBIT: _LIVELY,
    Species.ROBOT: _LIVELY,
    Species.SHARK: _DARTING,
    Species.SNAIL: _SLOW,
    Species.SNAKE: _DARTING,
    Species.TABBY: _EASY,
    Species.TURTLE: _SLOW,
}


def get_idle_frame(species: Species, tick: int) -> tuple[int, bool]:
    """Choose an idle pixel pose and eyelid state for the current tick."""
    return _TEMPERAMENTS[species].pose(tick), tick % _BLINK_EVERY_TICKS == _BLINK_EVERY_TICKS - 1


def get_pet_frame(tick: int) -> int:
    """Cycle the three intact petting poses at the caller's faster cadence."""
    return IDLE_FRAME_COUNT + tick % IDLE_FRAME_COUNT

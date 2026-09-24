# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""How experience turns into levels and evolutions. Plain integers in, plain integers out."""

from __future__ import annotations

from dataclasses import dataclass

XP_PER_TURN = 10
XP_PER_PET = 2
PETS_COUNTED_PER_TURN = 5
"""Petting is affection, not work: only this many pets per finished turn earn experience."""

LEVELS_PER_STAGE = 99
"""Level-ups spent in one stage. The last one evolves the buddy, or reaches ``FINAL_LEVEL`` in the final stage."""

FINAL_LEVEL = LEVELS_PER_STAGE + 1


@dataclass(frozen=True)
class Progress:
    """Where a buddy stands: its stage, its level there, and the way to the next level."""

    stage: int
    level: int
    xp_into_level: int
    xp_for_level: int

    @property
    def maxed(self) -> bool:
        """Whether there is nothing left to earn."""
        return self.xp_for_level == 0


def earned_xp(turns: int, pets: int) -> int:
    """Experience earned by *turns* finished turns and *pets* pets."""
    counted_pets = min(pets, PETS_COUNTED_PER_TURN * (turns + 1))
    return XP_PER_TURN * turns + XP_PER_PET * counted_pets


def xp_for_level(level: int, stage: int) -> int:
    """Experience needed to leave *level* of *stage*: every ten levels and every stage cost more."""
    return XP_PER_TURN * (10 + level // 10 + 2 * stage)


def progress(xp: int, final_stage: int) -> Progress:
    """Spend *xp* level by level, evolving until *final_stage*."""
    stage, level = 0, 1
    while not (stage == final_stage and level == FINAL_LEVEL):
        cost = xp_for_level(level, stage)
        if xp < cost:
            return Progress(stage, level, xp, cost)
        xp -= cost
        if level == LEVELS_PER_STAGE and stage < final_stage:
            stage, level = stage + 1, 1
        else:
            level += 1
    return Progress(stage, level, 0, 0)

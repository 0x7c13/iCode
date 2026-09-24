# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Everything that can happen to the buddy. Each change is one locked read-modify-write of the save file.

A change raises ``OSError`` when the save file cannot be written, which includes the
``TimeoutError`` of another instance holding it for too long. Reading never raises.
"""

from __future__ import annotations

from dataclasses import replace
from random import SystemRandom
from typing import TYPE_CHECKING

from chrys.app.features.buddy.hatchery import hatchling
from chrys.app.features.buddy.model import Buddy, BuddyRecord, clean_name
from chrys.app.features.buddy.store import BuddyStore

if TYPE_CHECKING:
    from collections.abc import Callable
    from random import Random

_STORE = BuddyStore()


def current_buddy() -> Buddy | None:
    """The user's buddy, or None before one has hatched."""
    return _grown(_STORE.load())


def hatch(rng: Random | None = None) -> Buddy:
    """Hatch a buddy. When one already exists, in this instance or another, that one is returned."""
    newborn = hatchling(rng if rng is not None else SystemRandom())
    record = _STORE.update(lambda current: current if current is not None else newborn)
    assert record is not None  # the change above always leaves a record behind
    return Buddy.of(record)


def rename(name: str) -> Buddy | None:
    """Give the buddy a new name. A name that cleans up to nothing changes nothing."""
    cleaned = clean_name(name)
    return _change(lambda record: replace(record, name=cleaned) if cleaned else record)


def set_muted(muted: bool) -> Buddy | None:
    return _change(lambda record: replace(record, muted=muted))


def record_pet() -> Buddy | None:
    return _change(lambda record: replace(record, pets=record.pets + 1))


def record_turn() -> Buddy | None:
    return _change(lambda record: replace(record, turns=record.turns + 1))


def _change(edit: Callable[[BuddyRecord], BuddyRecord]) -> Buddy | None:
    """Apply *edit* to the hatched buddy. Without one, nothing happens and None comes back."""
    # Most users never hatch one, and every finished turn ends up here: look before queueing for the
    # lock, so that for them a turn touches nothing on disk and cannot wait on another instance.
    if _STORE.load() is None:
        return None
    return _grown(_STORE.update(lambda current: edit(current) if current is not None else None))


def _grown(record: BuddyRecord | None) -> Buddy | None:
    return Buddy.of(record) if record is not None else None

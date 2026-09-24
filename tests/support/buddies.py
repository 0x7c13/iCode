# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A known buddy for tests that need one without hatching at random."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from chrys.app.features.buddy.model import Buddy, BuddyRecord, Rarity, Species, Trait
from chrys.app.features.buddy.progression import LEVELS_PER_STAGE, XP_PER_TURN, xp_for_level
from chrys.app.features.buddy.replies import reply_gate
from chrys.app.features.buddy.store import BuddyStore
from chrys.app.tui import buddy_reply

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest


def a_record(**changes: Any) -> BuddyRecord:
    """Pico the rare owl, freshly hatched, with *changes* applied."""
    record = BuddyRecord(
        species=Species.OWL,
        rarity=Rarity.R,
        shiny=False,
        traits={Trait.FOCUS: 60, Trait.CURIOSITY: 40, Trait.GRIT: 30, Trait.CHARM: 20},
        name="Pico",
        hatched_at=datetime(2026, 9, 22, 8, 30, tzinfo=UTC),
    )
    return replace(record, **changes)


def a_buddy(**changes: Any) -> Buddy:
    """The buddy that :func:`a_record` grows into."""
    return Buddy.of(a_record(**changes))


def turns_to_finish(stages: int) -> int:
    """Turns of work, with no petting, that complete *stages* whole stages."""
    xp = sum(xp_for_level(level, stage) for stage in range(stages) for level in range(1, LEVELS_PER_STAGE + 1))
    return -(-xp // XP_PER_TURN)


def assert_reply_gate_open() -> None:
    """Fail when a test left the process-wide reply gate held, and open it again either way.

    The gate outlives the test. Left held, every later gate test on the same worker would
    fail too, and the first red test would rarely be the one at fault.
    """
    held = not reply_gate.acquire(blocking=False)
    reply_gate.release()
    assert not held, "the test left the process-wide reply gate held"


def wedge_the_save_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """From now on every change to the save file times out, as it does while another instance sits on the lock."""

    def update(_self: BuddyStore, _change: Callable[[BuddyRecord | None], BuddyRecord | None]) -> BuddyRecord | None:
        raise TimeoutError("another instance holds the save file")

    monkeypatch.setattr(BuddyStore, "update", update)


class HeldSaveFile:
    """Another instance sits on the save file: every change waits, where it is called, until :meth:`release`.

    :attr:`entered` is set by the first change to arrive. A change that reaches it from a thread
    lets the event loop see it; one that reaches it on the loop stops the loop, and the test
    with it, until the hold gives up.
    """

    _GIVE_UP_SECONDS = 10.0

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.entered = threading.Event()
        self._released = threading.Event()
        real_update = BuddyStore.update

        def update(store: BuddyStore, change: Callable[[BuddyRecord | None], BuddyRecord | None]) -> BuddyRecord | None:
            self.entered.set()
            if not self._released.wait(timeout=self._GIVE_UP_SECONDS):
                raise TimeoutError("the save file was never released")
            return real_update(store, change)

        monkeypatch.setattr(BuddyStore, "update", update)

    def release(self) -> None:
        """Let every change, waiting or still to come, through to the save file."""
        self._released.set()


class HeldPetReply:
    """Stands in for the model behind every TUI pet: asked at once, it answers only when :attr:`go` is set."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, text: str = "💛 done") -> None:
        self.calls = 0
        self.asked = asyncio.Event()
        self.go = asyncio.Event()
        self._text = text
        monkeypatch.setattr(buddy_reply, "pet_reply", self._pet_reply)

    async def _pet_reply(self, _buddy: Buddy) -> str:
        self.calls += 1
        self.asked.set()
        await self.go.wait()
        return self._text

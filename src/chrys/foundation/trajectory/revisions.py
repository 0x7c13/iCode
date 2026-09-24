# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Context-revision chains: exact request membership as checkpoints and deltas.

Every wire request's context is a *revision* of the actor's previous one.
A revision records which items (by ``analytics_item_id``) the request
carried, in order and with repeats, so the analysis layer can replay the
exact context of any exchange without re-reading ``session.json``. The
first revision of a chain and every ``CHECKPOINT_INTERVAL``-th one is a
full checkpoint; the rest carry only the entries added and removed since
their parent. A keyed ``membership_hash`` over the full ordered membership
rides on every revision so a reconstructed chain can be verified.

Entries are ``{item_id, occurrence, position, action}``: ``occurrence`` is
the 0-based repeat index of that item within the request (so an item sent
twice is two distinct entries) and ``position`` its 0-based index in the
request; both are required for exact membership.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from chrys.foundation.trajectory.fingerprint import DOMAIN_MEMBERSHIP, canonical_json_bytes, keyed_fingerprint
from chrys.foundation.trajectory.ids import new_analytics_id

CHECKPOINT_INTERVAL: Final = 16
"""Every this many revisions of a chain is a full checkpoint (the first always is)."""

ACTION_ADD: Final = "add"
ACTION_REMOVE: Final = "remove"

MembershipRef = tuple[str, int]
"""``(item_id, occurrence)`` — one slot of a request's ordered membership."""


def membership_of(item_ids: Sequence[str]) -> tuple[MembershipRef, ...]:
    """Number the repeats of each item id in request order."""
    seen: dict[str, int] = {}
    refs: list[MembershipRef] = []
    for item_id in item_ids:
        occurrence = seen.get(item_id, 0)
        seen[item_id] = occurrence + 1
        refs.append((item_id, occurrence))
    return tuple(refs)


def membership_hash(key: bytes | None, membership: Sequence[MembershipRef]) -> str | None:
    """Keyed digest of the full ordered membership; ``None`` before a key is loaded."""
    if key is None:
        return None
    return keyed_fingerprint(key, DOMAIN_MEMBERSHIP, canonical_json_bytes([list(ref) for ref in membership]))


def _entry(ref: MembershipRef, *, position: int, action: str) -> dict[str, Any]:
    return {"item_id": ref[0], "occurrence": ref[1], "position": position, "action": action}


@dataclass(frozen=True, slots=True)
class RevisionPlan:
    """One computed revision, not yet committed to its chain."""

    revision_id: str
    parent_revision_id: str | None
    is_checkpoint: bool
    membership: tuple[MembershipRef, ...]
    entries: tuple[dict[str, Any], ...]
    """Checkpoint: every slot as ``add``; delta: removed slots (old positions) then added slots (new positions)."""

    @property
    def item_count(self) -> int:
        return len(self.membership)


@dataclass(slots=True)
class RevisionChain:
    """The revision lineage of one actor within one runtime."""

    last_revision_id: str | None = None
    membership: tuple[MembershipRef, ...] = ()
    revisions_since_checkpoint: int = 0

    def plan(self, membership: Sequence[MembershipRef]) -> RevisionPlan:
        """Compute the next revision for *membership* (a checkpoint when due or cheaper)."""
        new_membership = tuple(membership)
        revision_id = new_analytics_id()
        parent = self.last_revision_id
        checkpoint_entries = tuple(_entry(ref, position=i, action=ACTION_ADD) for i, ref in enumerate(new_membership))
        if parent is None or self.revisions_since_checkpoint + 1 >= CHECKPOINT_INTERVAL:
            return RevisionPlan(revision_id, parent, True, new_membership, checkpoint_entries)
        old_slots = set(self.membership)
        new_slots = set(new_membership)
        if tuple(ref for ref in self.membership if ref in new_slots) != tuple(
            ref for ref in new_membership if ref in old_slots
        ):
            # A delta is replayed as "drop these old positions, insert these
            # new ones", which reproduces the request only when the slots that
            # survived kept their relative order. Nothing in the entry shape
            # can say "these two swapped", so a reorder takes a checkpoint.
            return RevisionPlan(revision_id, parent, True, new_membership, checkpoint_entries)
        removed = tuple(
            _entry(ref, position=i, action=ACTION_REMOVE)
            for i, ref in enumerate(self.membership)
            if ref not in new_slots
        )
        added = tuple(
            _entry(ref, position=i, action=ACTION_ADD) for i, ref in enumerate(new_membership) if ref not in old_slots
        )
        delta = removed + added
        if len(delta) >= len(checkpoint_entries):
            # A delta no smaller than the snapshot buys nothing; checkpoint instead.
            return RevisionPlan(revision_id, parent, True, new_membership, checkpoint_entries)
        return RevisionPlan(revision_id, parent, False, new_membership, delta)

    def commit(self, plan: RevisionPlan) -> None:
        """Advance the chain to *plan* (call only once its events were queued)."""
        self.last_revision_id = plan.revision_id
        self.membership = plan.membership
        self.revisions_since_checkpoint = 0 if plan.is_checkpoint else self.revisions_since_checkpoint + 1


@dataclass(slots=True)
class RevisionRegistry:
    """Revision chains per actor, shared by every context derived from one root."""

    _chains: dict[str, RevisionChain] = field(default_factory=dict)

    def chain(self, actor_id: str) -> RevisionChain:
        chain = self._chains.get(actor_id)
        if chain is None:
            chain = RevisionChain()
            self._chains[actor_id] = chain
        return chain

    def forget(self, actor_id: str) -> None:
        """Drop an actor's chain (its next revision starts a fresh checkpoint)."""
        self._chains.pop(actor_id, None)


__all__ = [
    "ACTION_ADD",
    "ACTION_REMOVE",
    "CHECKPOINT_INTERVAL",
    "MembershipRef",
    "RevisionChain",
    "RevisionPlan",
    "RevisionRegistry",
    "membership_hash",
    "membership_of",
]

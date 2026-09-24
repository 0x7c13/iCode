# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Context-revision chains: checkpoints, deltas and keyed membership hashes."""

from __future__ import annotations

from chrys.foundation.trajectory.ids import is_valid_analytics_id
from chrys.foundation.trajectory.revisions import (
    ACTION_ADD,
    ACTION_REMOVE,
    CHECKPOINT_INTERVAL,
    RevisionChain,
    RevisionRegistry,
    membership_hash,
    membership_of,
)

KEY = b"k" * 32


def _ids(count: int, *, start: int = 0) -> list[str]:
    return [f"{index:032x}" for index in range(start, start + count)]


# -------------------------------------------------------------- membership


def test_membership_numbers_repeats_in_request_order() -> None:
    first, second = _ids(2)

    assert membership_of([first, second, first, first]) == (
        (first, 0),
        (second, 0),
        (first, 1),
        (first, 2),
    )


def test_membership_of_nothing_is_empty() -> None:
    assert membership_of([]) == ()


def test_membership_hash_is_keyed_order_sensitive_and_stable() -> None:
    first, second = _ids(2)
    forward = membership_of([first, second])
    backward = membership_of([second, first])

    digest = membership_hash(KEY, forward)
    assert digest is not None
    assert digest == membership_hash(KEY, forward)
    assert digest != membership_hash(KEY, backward)
    assert digest != membership_hash(b"j" * 32, forward)


def test_membership_hash_without_a_key_is_none() -> None:
    # Before the writer loads its key nothing may be fingerprinted at all.
    assert membership_hash(None, membership_of(_ids(2))) is None


# ------------------------------------------------------------------ chain


def test_first_revision_is_a_parentless_checkpoint() -> None:
    chain = RevisionChain()
    membership = membership_of(_ids(3))

    plan = chain.plan(membership)

    assert plan.is_checkpoint is True
    assert plan.parent_revision_id is None
    assert is_valid_analytics_id(plan.revision_id)
    assert plan.item_count == 3
    assert [entry["action"] for entry in plan.entries] == [ACTION_ADD] * 3
    assert [entry["position"] for entry in plan.entries] == [0, 1, 2]
    assert all(entry["occurrence"] == 0 for entry in plan.entries)


def test_planning_does_not_advance_the_chain_until_committed() -> None:
    chain = RevisionChain()
    membership = membership_of(_ids(3))

    first = chain.plan(membership)
    # A revision that was never queued must not become anybody's parent.
    assert chain.plan(membership).parent_revision_id is None

    chain.commit(first)
    assert chain.plan(membership).parent_revision_id == first.revision_id


def test_second_revision_is_a_delta_naming_only_what_changed() -> None:
    items = _ids(6)
    chain = RevisionChain()
    chain.commit(chain.plan(membership_of(items[:4])))

    # Drop the item at position 1, keep the rest, append two new ones.
    plan = chain.plan(membership_of([items[0], items[2], items[3], items[4], items[5]]))

    assert plan.is_checkpoint is False
    assert plan.item_count == 5
    removed = [entry for entry in plan.entries if entry["action"] == ACTION_REMOVE]
    added = [entry for entry in plan.entries if entry["action"] == ACTION_ADD]
    assert [entry["item_id"] for entry in removed] == [items[1]]
    # A removal names the slot's position in the PARENT membership.
    assert removed[0]["position"] == 1
    assert [entry["item_id"] for entry in added] == [items[4], items[5]]
    # An addition names its position in the NEW membership.
    assert [entry["position"] for entry in added] == [3, 4]
    # Removals are listed before additions so a replay can apply them in order.
    assert plan.entries.index(removed[0]) < plan.entries.index(added[0])


def test_unchanged_membership_is_an_empty_delta() -> None:
    chain = RevisionChain()
    membership = membership_of(_ids(3))
    chain.commit(chain.plan(membership))

    plan = chain.plan(membership)

    assert plan.is_checkpoint is False
    assert plan.entries == ()
    assert plan.item_count == 3


def test_repeat_occurrences_are_distinct_slots_in_a_delta() -> None:
    item = _ids(1)[0]
    chain = RevisionChain()
    chain.commit(chain.plan(membership_of([item])))

    plan = chain.plan(membership_of([item, item]))

    # The second copy is its own slot, not a no-op against the first.
    assert plan.entries == ({"item_id": item, "occurrence": 1, "position": 1, "action": ACTION_ADD},)


def test_a_delta_no_smaller_than_the_snapshot_becomes_a_checkpoint() -> None:
    chain = RevisionChain()
    chain.commit(chain.plan(membership_of(_ids(2))))

    # Wholesale replacement: 2 removals + 2 additions cost more than 2 adds.
    plan = chain.plan(membership_of(_ids(2, start=100)))

    assert plan.is_checkpoint is True
    assert [entry["action"] for entry in plan.entries] == [ACTION_ADD, ACTION_ADD]


def test_a_reordered_membership_takes_a_checkpoint() -> None:
    first, second, third = _ids(3)
    chain = RevisionChain()
    chain.commit(chain.plan(membership_of([first, second, third])))

    plan = chain.plan(membership_of([second, first, third]))

    # Same slots in a new order: a delta is replayed as removals then
    # insertions, and neither entry can say "these two swapped".
    assert plan.is_checkpoint is True
    assert [entry["item_id"] for entry in plan.entries] == [second, first, third]


def test_a_reorder_that_also_adds_an_item_takes_a_checkpoint() -> None:
    first, second, third = _ids(3)
    chain = RevisionChain()
    chain.commit(chain.plan(membership_of([first, second])))

    plan = chain.plan(membership_of([second, first, third]))

    # The cheap delta here is a single addition, and replaying it would put
    # the survivors back in their old order — [first, second, third].
    assert plan.is_checkpoint is True
    assert [entry["item_id"] for entry in plan.entries] == [second, first, third]


def test_checkpoint_returns_every_interval_and_resets_the_counter() -> None:
    chain = RevisionChain()
    membership = membership_of(_ids(40))
    checkpoints: list[int] = []
    for index in range(CHECKPOINT_INTERVAL * 2 + 1):
        plan = chain.plan(membership)
        if plan.is_checkpoint:
            checkpoints.append(index)
        chain.commit(plan)

    # The first revision, then one every CHECKPOINT_INTERVAL-th revision.
    assert checkpoints == [0, CHECKPOINT_INTERVAL, CHECKPOINT_INTERVAL * 2]


def test_committing_a_checkpoint_clears_the_since_checkpoint_count() -> None:
    chain = RevisionChain()
    plan = chain.plan(membership_of(_ids(2)))
    chain.commit(plan)

    assert chain.revisions_since_checkpoint == 0
    assert chain.membership == plan.membership
    assert chain.last_revision_id == plan.revision_id

    chain.commit(chain.plan(membership_of(_ids(3))))
    assert chain.revisions_since_checkpoint == 1


# --------------------------------------------------------------- registry


def test_registry_keeps_one_chain_per_actor() -> None:
    registry = RevisionRegistry()
    membership = membership_of(_ids(2))
    registry.chain("main").commit(registry.chain("main").plan(membership))

    # A second actor starts its own lineage: its first revision is a checkpoint.
    other = registry.chain("sub").plan(membership)
    assert other.parent_revision_id is None
    assert other.is_checkpoint is True
    assert registry.chain("main").plan(membership).parent_revision_id is not None


def test_forgetting_an_actor_restarts_its_chain() -> None:
    registry = RevisionRegistry()
    membership = membership_of(_ids(2))
    chain = registry.chain("main")
    chain.commit(chain.plan(membership))

    registry.forget("main")

    assert registry.chain("main").plan(membership).parent_revision_id is None
    # Forgetting an actor that was never seen is not an error.
    registry.forget("never-seen")

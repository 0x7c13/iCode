# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Unit pins for the weak-identity primitives.

``WeakIdentityRegistry`` replaces persisted bare-``id()`` sets: membership
requires the registered object itself to still be alive (``ref() is obj``),
so a recycled address can never impersonate a dead member and a copy is
never a member. ``ContentList`` exists because built-in ``list`` is not
weakref-able while several load-bearing identities are exactly LIST
identities (``id(msg.contents)`` in anchors and the summary cache).
"""

from __future__ import annotations

import copy
import gc
import json
import weakref

import pytest

from chrys.kernel.identity import ContentList, WeakIdentityRegistry
from chrys.kernel.types import Content, UsageDetails


def _text(text: str) -> Content:
    return Content.from_text(text)


def _usage() -> Content:
    return Content.from_usage(UsageDetails(input_token_count=1, output_token_count=2))


class TestWeakIdentityRegistry:
    def test_membership_is_object_identity_not_equality(self) -> None:
        registry = WeakIdentityRegistry()
        original = _text("landed")
        twin = copy.deepcopy(original)

        registry.register(original)

        assert original in registry
        assert twin not in registry  # a copy is a different object, never a member

    def test_dead_member_stops_matching_and_entry_prunes(self) -> None:
        registry = WeakIdentityRegistry()
        content = _text("short-lived")
        registry.register(content)
        assert len(registry) == 1

        del content
        gc.collect()

        assert len(registry) == 0

    def test_recycled_id_never_matches_a_dead_member(self) -> None:
        """The core id-reuse hazard the registry retires: an object occupying
        a dead member's recycled address must fail membership. The collision
        is simulated by planting the dead ref under the impostor's id — the
        ``ref() is obj`` clause can never hold for a different object."""
        registry = WeakIdentityRegistry()
        original = _text("dying")
        registry.register(original)
        dead_ref = registry._refs[id(original)]
        del original
        gc.collect()
        assert dead_ref() is None

        impostor = _text("dying")
        registry._refs[id(impostor)] = dead_ref
        assert impostor not in registry

    def test_stale_eviction_callback_spares_recycled_id_entry(self) -> None:
        """The one subtle line in the design: the weakref eviction callback
        must guard ``d.get(key) is ref`` so a STALE callback — firing after a
        new object was re-registered under the recycled id — cannot race out
        the new entry. Two live objects cannot share an id, so the recycled
        key is simulated by planting the fresh entry under the stale key
        before firing the captured callback by hand."""
        registry = WeakIdentityRegistry()
        original = _text("original")
        key = id(original)
        registry.register(original)
        stale_ref = registry._refs[key]
        stale_callback = stale_ref.__callback__
        assert stale_callback is not None

        replacement = _text("recycled-id occupant")
        fresh_ref = weakref.ref(replacement)
        registry._refs[key] = fresh_ref

        stale_callback(stale_ref)

        assert registry._refs[key] is fresh_ref  # guard held: fresh entry survives

        del original
        gc.collect()
        assert registry._refs[key] is fresh_ref  # natural stale callback also spares it

    def test_reregistering_same_object_is_idempotent(self) -> None:
        registry = WeakIdentityRegistry()
        content = _text("twice")
        registry.register(content)
        first_ref = registry._refs[id(content)]

        registry.register(content)

        assert registry._refs[id(content)] is first_ref
        assert len(registry) == 1

        del content
        gc.collect()
        assert len(registry) == 0

    def test_ignore_usage_registry_refuses_usage_contents(self) -> None:
        """The echo registry's construction-time policy: the usage exemption
        that six echo/assembly sites held by discipline becomes this one
        flag — usage contents never become members, so the echo filter can
        never strip them."""
        registry = WeakIdentityRegistry(ignore_usage=True)
        usage = _usage()
        text = _text("still registers")

        registry.register(usage)
        registry.register(text)

        assert usage not in registry
        assert text in registry
        assert len(registry) == 1

    def test_default_registry_keeps_usage_contents(self) -> None:
        """NOT universal behavior: compaction's identity sets deliberately
        include usage contents (fold exclusion must keep seeing them)."""
        registry = WeakIdentityRegistry()
        usage = _usage()

        registry.register(usage)

        assert usage in registry

    def test_unweakrefable_member_raises_loudly(self) -> None:
        registry = WeakIdentityRegistry()
        with pytest.raises(TypeError):
            registry.register([_text("plain lists cannot be weak members")])

    def test_content_list_registers(self) -> None:
        registry = WeakIdentityRegistry()
        contents = ContentList([_text("a")])
        registry.register(contents)

        assert contents in registry
        assert ContentList([_text("a")]) not in registry

        del contents
        gc.collect()
        assert len(registry) == 0


class TestContentList:
    def test_weakrefable_where_plain_list_is_not(self) -> None:
        contents = ContentList([_text("hold")])
        ref = weakref.ref(contents)
        assert ref() is contents
        with pytest.raises(TypeError):
            weakref.ref([])

    def test_ref_dies_with_the_list(self) -> None:
        contents = ContentList([_text("hold")])
        ref = weakref.ref(contents)
        del contents
        gc.collect()
        assert ref() is None

    def test_behaves_like_list(self) -> None:
        items = [_text("a"), _text("b")]
        contents = ContentList(items)

        assert isinstance(contents, list)
        assert contents == items
        assert json.dumps(ContentList(["x", "y"])) == '["x", "y"]'

    def test_deepcopy_preserves_subclass_and_copies_items(self) -> None:
        original = ContentList([_text("a")])
        duplicate = copy.deepcopy(original)

        assert type(duplicate) is ContentList
        assert duplicate is not original
        assert duplicate[0] is not original[0]
        assert duplicate[0].text == "a"

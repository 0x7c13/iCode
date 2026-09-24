# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Weak-identity primitives for transcript object bookkeeping.

A persisted bare ``id()`` integer neither keeps its object alive nor
notices its death: once the object is collected, the recycled address can
impersonate it. ``WeakIdentityRegistry`` fixes exactly that — membership
requires the registered object itself to still be alive (``ref() is obj``),
so a copy is never a member, a dead member never matches, and address
reuse is harmless by construction.

``ContentList`` exists because several load-bearing identities are LIST
identities (``id(msg.contents)`` captured by compaction anchors and the
summary cache) and built-in ``list`` does not support weak references.
``Message.contents`` coerces every assigned sequence to ``ContentList``,
so captors can hold ``weakref.ref(the_list)`` with today's snapshot
semantics: a replaced contents list stops matching, a shared-list twin
keeps matching, and the reference dies only when the list object truly
dies.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._content import Content

__all__ = ["ContentList", "WeakIdentityRegistry"]


class ContentList(list["Content"]):
    """A weakref-able ``list`` of message contents.

    Behaviorally a plain list — ``isinstance(x, list)``, equality, JSON
    serialization, and ``deepcopy`` (which preserves the subclass) are all
    unaffected. The single ``__weakref__`` slot adds weak-reference support
    without growing a per-instance ``__dict__``.
    """

    __slots__ = ("__weakref__",)


class WeakIdentityRegistry:
    """Object-identity membership that notices member death.

    Maps ``id(obj)`` to a weak reference; ``obj in registry`` holds only
    while the stored reference still points at that exact object. Identity
    semantics are bit-for-bit those of a bare ``id()`` set over live
    objects: a copy is a different object and is never a member; nothing
    here can serialize or be inherited through ``__deepcopy__``.

    Keyword Args:
        ignore_usage: When True, ``register`` refuses usage-type contents so
            they can never become members. This is the ECHO registry's
            construction-time policy — echo filtering must never strip
            usage carriers — not universal behavior: compaction's identity
            sets deliberately include usage contents.
    """

    __slots__ = ("_ignore_usage", "_refs")

    def __init__(self, *, ignore_usage: bool = False) -> None:
        self._refs: dict[int, weakref.ref[Any]] = {}
        self._ignore_usage = ignore_usage

    def register(self, obj: Any) -> None:
        """Add ``obj`` as a member for as long as it stays alive.

        Raises TypeError for objects that do not support weak references
        (notably plain ``list`` — contents lists must be ``ContentList``).
        """
        if self._ignore_usage and getattr(obj, "type", None) == "usage":
            return
        key = id(obj)
        refs = self._refs
        existing = refs.get(key)
        if existing is not None and existing() is obj:
            return

        def _evict(dead: weakref.ref[Any], *, key: int = key, refs: dict[int, weakref.ref[Any]] = refs) -> None:
            # A stale callback may fire after a NEW object was re-registered
            # under the recycled id; evict only our own entry.
            if refs.get(key) is dead:
                del refs[key]

        refs[key] = weakref.ref(obj, _evict)

    def __contains__(self, obj: object) -> bool:
        ref = self._refs.get(id(obj))
        return ref is not None and ref() is obj

    def __len__(self) -> int:
        return len(self._refs)

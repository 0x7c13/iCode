# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Patch: a node removed from the DOM stays in no cached lookup or arrangement.

Problem
-------
Textual 8.2.7 caches each ``query_one`` result on the node it queried (``_query_one_cache``, up
to 1024 entries), each widget's arrangement of its children (``_arrangement_cache``, up to four)
and each child list's displayed children (``NodeList._displayed_nodes``), keyed by the child-list
update count. Removing a node bumps the count on its parent and on every ancestor, so no entry
cached before the removal is read again, yet each stays until a newer one replaces it. A widget
that looks up a child and later removes it keeps the child's whole subtree alive for as long as
the widget lives: neither ``remove()`` nor a garbage collection releases it, and a
``gc.freeze()`` moves it into the permanent generation.

Solution
--------
Once ``NodeList._remove`` has taken a node out of its parent's list, clear the parent's and every
ancestor's ``query_one`` cache, the parent's arrangement cache and the list's displayed children.
The removal has just made every entry in them unreachable, so clearing them costs no cache hit.
A removal that finds no node changes no count and clears nothing. Every way out of the DOM goes
through ``_remove``: pruning a removed widget, ``App._unregister`` and ``move_child``, and
Textual clears a removed node's own caches. Wrapping the method on the class duplicates no
upstream body. It reads private attributes of the pinned Textual only: another release runs
unpatched, and GC freeze then detaches the screen caches instead of freezing them in place.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

_RUNTIME_PATCH_MARKER = "_chrys_removed_node_caches"
_RUNTIME_PATCH_TEXTUAL_VERSION = "8.2.7"
logger = logging.getLogger(__name__)


def apply_runtime_patch() -> None:
    """Patch ``NodeList._remove`` in the current process."""
    try:
        import textual
    except ImportError:
        return
    if textual.__version__ != _RUNTIME_PATCH_TEXTUAL_VERSION:
        logger.warning(
            "Skipping Textual removed-node cache runtime patch: loaded Textual is not the pinned %s that the "
            "patch targets.",
            _RUNTIME_PATCH_TEXTUAL_VERSION,
        )
        return
    from textual._node_list import NodeList
    from textual.widget import Widget

    original = NodeList._remove
    if getattr(original, _RUNTIME_PATCH_MARKER, False):
        return

    @functools.wraps(original)
    def _remove(self: Any, widget: Any) -> None:
        updates = self._updates
        original(self, widget)
        if self._updates == updates:
            return
        self._displayed_nodes = (-1, [])
        self._displayed_visible_nodes = (-1, [])
        owner = None if self._parent is None else self._parent()
        if isinstance(owner, Widget) and owner._arrangement_cache:
            owner._arrangement_cache.clear()
        node = owner
        while node is not None:
            if node._query_one_cache:
                node._query_one_cache.clear()
            node = node._parent

    setattr(_remove, _RUNTIME_PATCH_MARKER, True)
    NodeList._remove = _remove


def removal_clears_node_caches() -> bool:
    """Whether this process runs the patch, so a removed node leaves no cached entry behind."""
    try:
        from textual._node_list import NodeList
    except ImportError:
        return False
    return bool(getattr(NodeList._remove, _RUNTIME_PATCH_MARKER, False))

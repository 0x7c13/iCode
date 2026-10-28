# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Patch: a widget sized to its children measures them anew once one shows, hides, mounts or goes.

Problem
-------
Textual 8.2.7 caches each widget's box model under its ``_layout_updates`` count. A child laid out
anew bumps that count on its ancestors sized to their content only in ``_check_refresh``, which
runs once the child's own message pump idles. A child shown, hidden, mounted or removed bumps its
ancestors' child-list counts at once (``NodeList.updated``), so their arrangements are built again,
but not their box models: a screen layout that another widget asked for and that runs before the
child's pump idles measures each such ancestor at the size it had. A transcript entry that folds as
a tool call starts below it is laid out at its open height for a frame, and a view that follows the
bottom of the chat jumps down.

Solution
--------
Once ``NodeList.updated`` has counted the change, bump the count of the list's owner and of its
ancestors up to the first that is not sized to its content, as ``refresh(layout=True)`` and
``_check_refresh`` do for a widget laid out anew. A widget not mounted yet was never measured, so a
transcript precomposed detached stores no count. Wrapping the method on the class duplicates no
upstream body. The reflow-reuse patch installs its own ``NodeList.updated`` from the upstream
source, so this one is installed after it and wraps that one. It reads private attributes of the
pinned Textual only: another release runs unpatched.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

_RUNTIME_PATCH_MARKER = "_chrys_box_model_children"
_RUNTIME_PATCH_TEXTUAL_VERSION = "8.2.7"
logger = logging.getLogger(__name__)


def apply_runtime_patch() -> None:
    """Patch ``NodeList.updated`` in the current process."""
    try:
        import textual
    except ImportError:
        return
    if textual.__version__ != _RUNTIME_PATCH_TEXTUAL_VERSION:
        logger.warning(
            "Skipping Textual box-model children runtime patch: loaded Textual is not the pinned %s that the "
            "patch targets.",
            _RUNTIME_PATCH_TEXTUAL_VERSION,
        )
        return
    from textual._node_list import NodeList
    from textual.widget import Widget

    original = NodeList.updated
    if getattr(original, _RUNTIME_PATCH_MARKER, False):
        return

    @functools.wraps(original)
    def updated(self: Any) -> None:
        original(self)
        owner = None if self._parent is None else self._parent()
        if not isinstance(owner, Widget) or not owner._is_mounted:
            return
        owner._layout_updates += 1
        node = owner._parent
        while isinstance(node, Widget):
            node._layout_updates += 1
            if not node.styles.auto_dimensions:
                break
            node = node._parent

    setattr(updated, _RUNTIME_PATCH_MARKER, True)
    NodeList.updated = updated

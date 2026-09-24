# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Opt-in eager composition for large Textual mount batches.

Textual normally starts one message pump per newly registered widget and lets
each pump compose and register its own children. That lifecycle is ideal for
incremental UI changes, but a persisted transcript already exists as one stable
tree: hundreds of parent pumps otherwise serialize hundreds of small child
registration operations.

``precompose_tree`` materializes an unattached widget forest synchronously, so
Textual's existing recursive ``App._register`` path can register and style the
whole forest in one operation. The runtime wrapper only skips the later Compose
event for widgets explicitly placed in that forest; all other Textual widgets
retain the upstream lifecycle.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING
from weakref import WeakSet

if TYPE_CHECKING:
    from textual.widget import Widget

_RUNTIME_PATCH_MARKER = "_chrys_precomposed_tree_fast_path"
_precomposed_widgets: WeakSet[Widget] = WeakSet()
_runtime_compose: Callable[[Widget], Iterable[Widget]] | None = None


def apply_runtime_patch() -> None:
    """Teach ``Widget._compose`` to consume explicitly precomposed nodes."""
    try:
        from textual.compose import compose
        from textual.widget import Widget
    except ImportError:
        return

    global _runtime_compose
    _runtime_compose = compose

    existing = Widget._compose
    if getattr(existing, _RUNTIME_PATCH_MARKER, False):
        return

    original_compose = existing

    async def _compose(self: Widget) -> None:
        if self in _precomposed_widgets:
            _precomposed_widgets.discard(self)
            return
        await original_compose(self)

    setattr(_compose, _RUNTIME_PATCH_MARKER, True)
    _compose._original = original_compose  # type: ignore[attr-defined]
    Widget._compose = _compose


def _precompose_widget(widget: Widget) -> None:
    """Materialize one new widget and every descendant into its node lists."""
    compose = _runtime_compose
    if compose is None:
        return
    try:
        children = [*widget._pending_children, *compose(widget)]
        widget._pending_children.clear()
    except TypeError as error:
        raise TypeError(f"{widget!r} compose() method returned an invalid result; {error}") from error
    except Exception as error:
        widget.app._handle_exception(error)
        # Upstream considers the Compose event handled after routing the
        # exception. Skip the later event so a broken compose() is not called
        # a second time when Textual starts this widget's message pump.
        _precomposed_widgets.add(widget)
    else:
        widget._extend_compose(children)
        for child in children:
            widget._nodes._append(child)
        _precomposed_widgets.add(widget)
        for child in children:
            _precompose_widget(child)


def precompose_tree(widgets: Iterable[Widget]) -> None:
    """Precompose a forest of new, unattached widgets for one bulk mount."""
    apply_runtime_patch()
    for widget in widgets:
        _precompose_widget(widget)

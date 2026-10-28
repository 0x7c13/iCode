# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The chat transcript panel as the widgets in it see it, without importing it."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from textual.widget import Widget


class ChatTranscriptPanelMarker:
    """Nominal marker base identifying the chat transcript panel.

    Chat helper modules must not import :mod:`chrys.app.tui.widgets.chat.panel`
    (see ``tests/architecture/test_tui_structure.py``). Helpers that need to
    recognize the transcript panel among a widget's ancestors — e.g. the
    selection controller's hot per-mouse-move filter — check this base with a
    plain ``isinstance`` instead of importing the concrete ``ChatPanel``.
    """

    __slots__ = ()

    def entry_folded(self, *, by_user: bool) -> None:
        """An entry folded: show the transcript at its new height in the layout that places it.

        *by_user*, it was folded by hand: it shows as made, without the rows a turn holds below the transcript for
        its own folds.
        """
        raise NotImplementedError


def note_fold(widget: Widget, *, by_user: bool = False) -> None:
    """Tell the chat transcript *widget* is in that it folded, *by_user* by hand, before a layout shows it.

    The rows it gives up go in that layout, not a frame after it. As a turn runs, the transcript holds those of its
    own folds, up to a third of the view; one the user makes shows as made.
    """
    for ancestor in widget.ancestors:
        if isinstance(ancestor, ChatTranscriptPanelMarker):
            ancestor.entry_folded(by_user=by_user)
            return

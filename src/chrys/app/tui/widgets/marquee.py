# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Small, render-agnostic state machine for horizontally scrolling text."""

from __future__ import annotations

from rich.cells import split_graphemes
from textual.content import Content


class OverflowMarquee:
    """Cycle overflowing styled content while preserving its original form.

    Timer ownership deliberately stays with the consuming widget. This class
    only models the current frame and the delay before the next one, making it
    reusable without creating a timer (or mounted child widget) per row.
    """

    def __init__(
        self,
        *,
        start_delay: float = 0.8,
        step_interval: float = 0.08,
        end_delay: float = 0.8,
    ) -> None:
        self.start_delay = start_delay
        self.step_interval = step_interval
        self.end_delay = end_delay
        self._original = Content()
        self._prefix = Content()
        self._content = Content()
        self._offsets: tuple[int, ...] = ()
        self._position = 0

    @property
    def active(self) -> bool:
        """Whether the activated content overflows its viewport."""
        return len(self._offsets) > 1

    @property
    def frame(self) -> Content:
        """Return the fixed prefix plus current suffix, or the static content."""
        if not self.active or self._position == 0:
            return self._original
        return self._prefix + self._content[self._offsets[self._position] :]

    def activate(
        self,
        content: Content,
        viewport_width: int,
        *,
        fixed_prefix_end: int = 0,
    ) -> float | None:
        """Reset to *content* and return the delay before scrolling starts.

        ``fixed_prefix_end`` is a character offset into *content*. Everything
        before it remains pinned while the remaining content scrolls within
        the cells left over after rendering the prefix.

        Offsets follow grapheme boundaries rather than Python characters, so
        emoji sequences and combining marks are never split between frames.
        """
        self.reset()
        self._original = content
        prefix_end = min(max(fixed_prefix_end, 0), len(content.plain))
        self._prefix = content[:prefix_end]
        self._content = content[prefix_end:]
        scrolling_width = viewport_width - self._prefix.cell_length
        if scrolling_width <= 0 or self._content.cell_length <= scrolling_width:
            return None

        graphemes, total_width = split_graphemes(self._content.plain)
        offsets = [0]
        removed_width = 0
        for _start, end, cell_width in graphemes:
            removed_width += cell_width
            if cell_width:
                offsets.append(end)
            if total_width - removed_width <= scrolling_width:
                break

        self._offsets = tuple(offsets)
        return self.start_delay if self.active else None

    def advance(self) -> float | None:
        """Advance one frame and return the delay before the following frame."""
        if not self.active:
            return None
        if self._position == len(self._offsets) - 1:
            self._position = 0
            return self.start_delay
        self._position += 1
        return self.end_delay if self._position == len(self._offsets) - 1 else self.step_interval

    def reset(self) -> None:
        """Release active content and restore the static position."""
        self._original = Content()
        self._prefix = Content()
        self._content = Content()
        self._offsets = ()
        self._position = 0

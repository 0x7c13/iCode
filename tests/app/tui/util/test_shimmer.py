# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The running row's blinking dot and sweeping band, and the waiting turn's breathing dot."""

from __future__ import annotations

from rich.color import Color
from rich.text import Text

from chrys.app.tui.util.shimmer import breathing_dot, running_dot, shimmer

_BASE = Color.parse("#404040")
_HIGHLIGHT = Color.parse("#f0f0f0")


def _lit_colors(text: Text) -> dict[int, Color | None]:
    return {span.start: span.style.color for span in text.spans if not isinstance(span.style, str)}


def test_the_running_dot_is_lit_then_dim_in_turns() -> None:
    assert [running_dot(now) for now in (0.0, 0.5, 0.6, 1.1, 1.2)] == [
        ("•", True),
        ("•", True),
        ("◦", False),
        ("◦", False),
        ("•", True),
    ]


def test_the_band_lights_the_title_around_its_centre_and_fades_out_at_its_edges() -> None:
    text = Text("x" * 20)
    # A sweep crosses the title and ten characters either side of it in two seconds: halfway, it is at 10.
    shimmer(text, 20, _BASE, _HIGHLIGHT, 1.0)
    lit = _lit_colors(text)
    assert sorted(lit) == list(range(5, 16))
    assert lit[10] == _HIGHLIGHT
    assert lit[5] == lit[15] == _BASE


def test_the_band_never_lights_past_the_end_it_is_given() -> None:
    text = Text("x" * 30)
    shimmer(text, 20, _BASE, _HIGHLIGHT, 1.6)
    assert sorted(_lit_colors(text)) == [17, 18, 19]


def test_terminal_colours_light_only_the_band_s_middle_in_the_highlight() -> None:
    text = Text("x" * 20)
    shimmer(text, 20, Color.parse("white"), Color.parse("bright_white"), 1.0)
    lit = _lit_colors(text)
    assert sorted(lit) == list(range(8, 13))
    assert set(lit.values()) == {Color.parse("bright_white")}


def test_the_band_lights_nothing_without_both_colours_or_text() -> None:
    for base, highlight, end in ((None, _HIGHLIGHT, 20), (_BASE, None, 20), (_BASE, _HIGHLIGHT, 0)):
        text = Text("x" * 20)
        shimmer(text, end, base, highlight, 1.0)
        assert not text.spans


def test_the_breathing_dot_grows_and_brightens_then_shrinks_and_dims() -> None:
    smallest, grown, largest = (breathing_dot(_BASE, _HIGHLIGHT, now) for now in (0.0, 0.3, 0.6))
    assert (smallest.plain, grown.plain, largest.plain) == ("·", "•", "●")
    assert smallest.style.color == _BASE  # type: ignore[union-attr]
    assert largest.style.color == _HIGHLIGHT  # type: ignore[union-attr]
    assert breathing_dot(_BASE, _HIGHLIGHT, 0.9).plain == "•"
    assert breathing_dot(_BASE, _HIGHLIGHT, 1.2).plain == "·"


def test_without_colours_the_breathing_dot_stands_still() -> None:
    dot = breathing_dot(None, None, 0.6)
    assert dot.plain == "•"
    assert not dot.style

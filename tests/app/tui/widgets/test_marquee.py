# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the reusable overflow marquee state."""

from textual.content import Content

from chrys.app.tui.widgets.marquee import OverflowMarquee


def test_marquee_leaves_fitting_content_static() -> None:
    content = Content.assemble(("short", "bold"))
    marquee = OverflowMarquee()

    assert marquee.activate(content, viewport_width=5) is None
    assert marquee.active is False
    assert marquee.frame is content


def test_marquee_scrolls_on_grapheme_boundaries_and_preserves_styles() -> None:
    content = Content.assemble(("Á", "bold"), ("界BC", "dim"))
    marquee = OverflowMarquee(start_delay=1.0, step_interval=0.1, end_delay=2.0)

    assert marquee.activate(content, viewport_width=3) == 1.0
    assert marquee.frame.plain == "Á界BC"

    assert marquee.advance() == 0.1
    assert marquee.frame.plain == "界BC"
    assert marquee.frame.spans[0].style == "dim"

    assert marquee.advance() == 2.0
    assert marquee.frame.plain == "BC"

    assert marquee.advance() == 1.0
    assert marquee.frame.plain == "Á界BC"


def test_marquee_keeps_fixed_prefix_out_of_scroll() -> None:
    content = Content.assemble(("/review  ", "bold"), ("Long description", "dim"))
    marquee = OverflowMarquee(start_delay=1.0)

    assert marquee.activate(content, viewport_width=14, fixed_prefix_end=len("/review  ")) == 1.0
    assert marquee.advance() is not None
    assert marquee.frame.plain.startswith("/review  ")
    assert marquee.frame.plain != content.plain
    assert marquee.frame.spans[0].style == "bold"


def test_marquee_reset_releases_content_and_stops_advancing() -> None:
    marquee = OverflowMarquee()
    marquee.activate(Content("a long value"), viewport_width=4)

    marquee.reset()

    assert marquee.active is False
    assert marquee.frame.plain == ""
    assert marquee.advance() is None

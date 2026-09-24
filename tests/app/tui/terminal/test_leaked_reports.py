# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Telling the outer terminal's split focus and mouse reports from keys the user typed."""

from __future__ import annotations

import pytest

from chrys.app.tui.terminal.leaked_reports import MAX_HELD_LENGTH, ReportMatch, match_outer_report

_REPORTS = {
    "focus-in": "\x1b[I",
    "focus-out": "\x1b[O",
    "sgr-press": "\x1b[<0;3;4M",
    "sgr-release": "\x1b[<0;3;4m",
    "sgr-wide-coordinates": "\x1b[<35;120;48M",
    "sgr-negative-coordinate": "\x1b[<0;-1;4M",
    "urxvt": "\x1b[32;3;4M",
    # What is left of a numeric report whose first field the outer parser already consumed.
    "two-field-tail": "\x1b[3;4M",
    "x10": '\x1b[M !"',
}


@pytest.mark.parametrize("report", _REPORTS.values(), ids=_REPORTS.keys())
def test_whole_report_is_complete(report: str) -> None:
    assert match_outer_report(report) is ReportMatch.COMPLETE


@pytest.mark.parametrize("report", _REPORTS.values(), ids=_REPORTS.keys())
def test_every_beginning_of_a_report_may_yet_become_one(report: str) -> None:
    beginnings = [report[:length] for length in range(1, len(report))]

    assert [match_outer_report(beginning) for beginning in beginnings] == [ReportMatch.PARTIAL] * len(beginnings)


@pytest.mark.parametrize(
    "sequence",
    [
        "\x1bx",
        "\x1bO",
        "\x1b[A",
        "\x1b[1;5C",
        "\x1b[3~",
        "\x1b[200~",
        "\x1b[1M",
        "\x1b[1;2;3;4M",
        "\x1b[<0;3;4R",
        "\x1b[?1;2c",
    ],
    ids=[
        "alt-x",
        "ss3",
        "cursor-up",
        "ctrl-right",
        "delete",
        "paste-start",
        "one-field",
        "four-fields",
        "wrong-final",
        "device-attributes",
    ],
)
def test_ordinary_sequences_are_not_reports(sequence: str) -> None:
    assert match_outer_report(sequence) is ReportMatch.NONE


def test_x10_report_takes_any_three_characters() -> None:
    # The button and both coordinates are raw bytes, which may read as anything at all.
    assert match_outer_report("\x1b[MMMM") is ReportMatch.COMPLETE
    assert match_outer_report("\x1b[M;[I") is ReportMatch.COMPLETE


def test_numeric_prefix_is_held_up_to_the_cap_and_no_further() -> None:
    longest_held = "\x1b[" + "1" * (MAX_HELD_LENGTH - 2)

    assert len(longest_held) == MAX_HELD_LENGTH
    assert match_outer_report(longest_held) is ReportMatch.PARTIAL
    assert match_outer_report(longest_held + "1") is ReportMatch.NONE


def test_report_completes_at_the_cap() -> None:
    fields = "1" * ((MAX_HELD_LENGTH - 5) // 2)
    report = f"\x1b[<{fields};{fields}M"

    assert len(report) <= MAX_HELD_LENGTH
    assert match_outer_report(report) is ReportMatch.COMPLETE

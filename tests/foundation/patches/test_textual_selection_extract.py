# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the Textual selection-extraction bounds patch."""

from __future__ import annotations

import inspect

import pytest
from textual.geometry import Offset
from textual.selection import SELECT_ALL, Selection

from chrys.foundation.patches.textual_selection_extract import apply_runtime_patch

# Three lines of text that render a fourth, blank row at y=3.
_TEXT = "first\nsecond\nthird\n"


def test_upstream_extract_still_raises_past_the_last_line() -> None:
    """Pins that the patch is still needed; drop the patch once a Textual upgrade makes this fail."""
    apply_runtime_patch()
    upstream = inspect.unwrap(Selection.extract)

    with pytest.raises(IndexError):
        upstream(Selection(Offset(0, 3), None), _TEXT)


@pytest.mark.parametrize(
    "selection",
    [
        pytest.param(Selection(Offset(0, 3), None), id="first-widget-of-a-drag"),
        pytest.param(Selection.from_offsets(Offset(0, 3), Offset(4, 3)), id="within-the-blank-row"),
        pytest.param(Selection(Offset(2, 7), Offset(1, 9)), id="rows-far-below"),
    ],
)
def test_selection_starting_below_the_text_extracts_nothing(selection: Selection) -> None:
    apply_runtime_patch()

    assert selection.extract(_TEXT) == ""


@pytest.mark.parametrize(
    "selection",
    [
        pytest.param(SELECT_ALL, id="select-all"),
        pytest.param(Selection(None, Offset(3, 1)), id="last-widget-of-a-drag"),
        pytest.param(Selection(Offset(2, 0), None), id="first-widget-of-a-drag"),
        pytest.param(Selection(Offset(1, 1), Offset(4, 1)), id="single-line"),
        pytest.param(Selection(Offset(1, 2), Offset(2, 3)), id="onto-the-blank-row"),
    ],
)
def test_selection_starting_inside_the_text_matches_upstream(selection: Selection) -> None:
    apply_runtime_patch()
    upstream = inspect.unwrap(Selection.extract)

    assert selection.extract(_TEXT) == upstream(selection, _TEXT)


def test_runtime_patch_is_idempotent() -> None:
    apply_runtime_patch()
    patched = Selection.extract

    apply_runtime_patch()

    assert Selection.extract is patched
    assert inspect.unwrap(Selection.extract) is not patched

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the curated ``chrys.app.tui.widgets`` facade."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests.support.paths import REPO_ROOT, SRC_ROOT

ROOT = REPO_ROOT


def test_widget_facade_import_does_not_load_feature_packages() -> None:
    """Importing the facade should not eagerly import heavy widget packages."""
    code = """
import json
import sys

import chrys.app.tui.widgets as widgets

for name in widgets.__all__:
    getattr(widgets, name)

print(json.dumps({
    "exports": widgets.__all__,
    "loaded": sorted(
        name for name in sys.modules
        if name.startswith((
            "chrys.app.tui.widgets.chat",
            "chrys.app.tui.widgets.diff_view",
            "chrys.app.tui.widgets.markdown",
        ))
    ),
}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["loaded"] == []
    assert "VirtualizedMarkdown" not in payload["exports"]
    assert "DiffView" not in payload["exports"]
    assert "ToolCall" not in payload["exports"]


def test_widget_facade_resolves_curated_leaf_exports() -> None:
    """Facade names should resolve to the concrete leaf-widget implementations."""
    from chrys.app.tui import widgets
    from chrys.app.tui.widgets.ask_user_controls import AskUserOptions, AskUserResponseFooter
    from chrys.app.tui.widgets.buttons import ConfigActionButton, ConfigAddButton
    from chrys.app.tui.widgets.checkbox import CHECKED_MARKER, Checkbox
    from chrys.app.tui.widgets.dialog_scroll import StableAutoHeightScroll
    from chrys.app.tui.widgets.hatch import HATCH_GLYPH, HatchedEmptyState
    from chrys.app.tui.widgets.input import EnhancedInput
    from chrys.app.tui.widgets.loading import ChrysLoadingIndicator
    from chrys.app.tui.widgets.selection import normalize_selection_rich_style
    from chrys.app.tui.widgets.text_area import EnhancedTextArea

    assert widgets.AskUserOptions is AskUserOptions
    assert widgets.AskUserResponseFooter is AskUserResponseFooter
    assert widgets.CHECKED_MARKER is CHECKED_MARKER
    assert widgets.Checkbox is Checkbox
    assert widgets.ChrysLoadingIndicator is ChrysLoadingIndicator
    assert widgets.ConfigActionButton is ConfigActionButton
    assert widgets.ConfigAddButton is ConfigAddButton
    assert widgets.EnhancedInput is EnhancedInput
    assert widgets.EnhancedTextArea is EnhancedTextArea
    assert widgets.HATCH_GLYPH is HATCH_GLYPH
    assert widgets.HatchedEmptyState is HatchedEmptyState
    assert widgets.StableAutoHeightScroll is StableAutoHeightScroll
    assert widgets.normalize_selection_rich_style is normalize_selection_rich_style

    missing_name = "VirtualizedMarkdown"
    with pytest.raises(AttributeError):
        getattr(widgets, missing_name)

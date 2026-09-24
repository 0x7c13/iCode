# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the Textual ANSI button CSS patch."""

from __future__ import annotations

from textual.widgets import Button

from chrys.foundation.patches import textual_button_ansi


def test_runtime_patch_removes_ansi_button_css_from_loaded_button(monkeypatch) -> None:
    """Already-imported Button classes should be fixed in the current process."""
    css = f"{textual_button_ansi._RUNTIME_OLD}\n        width: auto;\n    }}"
    monkeypatch.setattr(Button, "DEFAULT_CSS", css)
    monkeypatch.delattr(Button, textual_button_ansi._RUNTIME_PATCH_MARKER, raising=False)

    textual_button_ansi.apply_runtime_patch()

    assert f"{textual_button_ansi._RUNTIME_NEW}\n        width: auto;\n    }}" == Button.DEFAULT_CSS
    assert "Button:ansi.-style-flat" not in Button.DEFAULT_CSS

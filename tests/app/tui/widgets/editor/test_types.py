# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Pure tests for editor values and the Standard keymap registry."""

from __future__ import annotations

import inspect
from typing import cast

import pytest

from chrys.app.tui.widgets.editor import (
    EditorCommandHost,
    EditorIntent,
    EditorKeymapRegistry,
    EditorKeyStroke,
    EditorMode,
    StandardKeymap,
    keymaps,
    vim,
    widget,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (EditorMode.VIM, EditorMode.VIM),
        ("standard", EditorMode.STANDARD),
        (" EMACS ", EditorMode.EMACS),
        ("ViM", EditorMode.VIM),
        ("unknown", EditorMode.STANDARD),
        (None, EditorMode.STANDARD),
    ],
)
def test_editor_mode_parser_normalizes_with_standard_fallback(value: object, expected: EditorMode) -> None:
    assert EditorMode.parse(value) is expected


def test_registry_returns_fresh_standard_keymaps() -> None:
    registry = EditorKeymapRegistry()

    first = registry.create(EditorMode.STANDARD)
    second = registry.create(EditorMode.STANDARD)

    assert isinstance(first, StandardKeymap)
    assert isinstance(second, StandardKeymap)
    assert first is not second


@pytest.mark.parametrize("mode", [EditorMode.EMACS, EditorMode.VIM])
def test_registry_returns_fresh_preference_keymaps(mode: EditorMode) -> None:
    registry = EditorKeymapRegistry()

    first = registry.create(mode)
    second = registry.create(mode)

    assert first.mode is mode
    assert second.mode is mode
    assert first is not second


def test_registry_normalizes_unknown_persisted_mode_to_standard() -> None:
    registry = EditorKeymapRegistry()

    assert registry.create(" unknown ").mode is EditorMode.STANDARD


def test_registry_requires_standard_factory() -> None:
    with pytest.raises(ValueError, match="Standard editor keymap factory is required"):
        EditorKeymapRegistry({EditorMode.EMACS: StandardKeymap})


def test_standard_keymap_only_consumes_guarded_escape() -> None:
    keymap = StandardKeymap()
    host = cast("EditorCommandHost", object())

    escape = keymap.handle_key(EditorKeyStroke("escape"), host)
    printable = keymap.handle_key(EditorKeyStroke("x", "x"), host)

    assert escape.consumed is True
    assert escape.intent is EditorIntent.REQUEST_ESCAPE_CANCEL
    assert printable.consumed is False
    assert printable.intent is EditorIntent.NONE


def test_keymap_dispatch_uses_no_reflective_or_private_history_api() -> None:
    source = "\n".join(inspect.getsource(module) for module in (keymaps, vim, widget))
    assert "getattr(" not in source
    assert "hasattr(" not in source
    assert "_pop_undo" not in source
    assert "_pop_redo" not in source
    assert ".undo_stack" not in source
    assert ".redo_stack" not in source

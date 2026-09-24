# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Detached drafts, transaction identity and distinct saved/opened baselines."""

from __future__ import annotations

from dataclasses import replace

import pytest
from textual.theme import Theme

from chrys.app.tui.theme import CHRYS_THEME
from chrys.app.tui.themes.document import ThemeDocument


def test_surface_edits_commit_once_and_never_mutate_source() -> None:
    source = replace(CHRYS_THEME, variables=dict(CHRYS_THEME.variables))
    document = ThemeDocument(source)
    token = document.begin("color:background")
    for value in ("#111111", "#222222", "#333333"):
        document.stage(token, value)
    assert document.undo_stack == []
    assert document.draft == source
    assert document.commit(token)
    assert len(document.undo_stack) == 1
    assert document.draft.background == "#333333"
    assert document.draft.surface == document.draft.panel == document.draft.boost == source.background
    assert source.background == CHRYS_THEME.background
    document.undo()
    assert document.draft == source
    document.redo()
    assert document.draft.background == "#333333"


@pytest.mark.parametrize("name", ["chrys", "chrys-ansi", "chrys-legacy", "chrys-custom", "my-theme"])
def test_theme_names_never_couple_independent_surfaces(name: str) -> None:
    document = ThemeDocument(replace(CHRYS_THEME, name=name))
    token = document.begin("color:background")
    document.stage(token, "#123456")
    document.commit(token)
    assert document.draft.background == "#123456"
    assert document.draft.surface == CHRYS_THEME.surface
    assert document.draft.panel == CHRYS_THEME.panel
    assert document.draft.boost == CHRYS_THEME.boost


def test_cancel_and_late_messages_cannot_touch_a_new_transaction() -> None:
    document = ThemeDocument(CHRYS_THEME)
    first = document.begin("var:footer-background")
    document.stage(first, "not-a-color")
    second = document.begin("color:primary")
    assert document.stage(first, "#112233") is None
    assert not document.commit(first)
    assert document.owns(second)
    document.cancel()
    assert document.draft == CHRYS_THEME
    assert document.undo_stack == []


def test_other_document_and_duplicate_commit_are_rejected() -> None:
    a, b = ThemeDocument(CHRYS_THEME), ThemeDocument(CHRYS_THEME)
    token = a.begin("color:primary")
    assert b.stage(token, "#123456") is None
    a.stage(token, "#123456")
    assert a.commit(token)
    assert not a.commit(token)
    assert len(a.undo_stack) == 1


def test_noop_confirmation_preserves_source_expression_and_history() -> None:
    document = ThemeDocument(Theme("custom", "red", variables={"text-muted": "ansi_white 40%"}))
    token = document.begin("var:text-muted")
    assert not document.commit(token)
    assert document.draft.variables["text-muted"] == "ansi_white 40%"
    assert document.undo_stack == []


def test_saved_baseline_does_not_replace_opening_baseline_or_history() -> None:
    document = ThemeDocument(CHRYS_THEME)
    token = document.begin("var:footer-background")
    document.stage(token, "#112233")
    document.commit(token)
    assert document.unsaved and document.changed
    document.mark_saved()
    assert not document.unsaved and document.changed
    assert len(document.undo_stack) == 1
    document.undo()
    assert document.unsaved and not document.changed


def test_unset_removes_override_and_redo_restores_expression() -> None:
    document = ThemeDocument(Theme("custom", "red", variables={"text-muted": "auto 60%"}))
    token = document.begin("var:text-muted")
    document.stage(token, None)
    document.commit(token)
    assert "text-muted" not in document.draft.variables
    document.undo()
    assert document.draft.variables["text-muted"] == "auto 60%"

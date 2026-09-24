# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for opt-in Textual tree precomposition."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widget import Widget
from textual.widgets import Static

from chrys.foundation.patches.textual_precompose import apply_runtime_patch, precompose_tree


class _Leaf(Static):
    compose_calls = 0

    def compose(self) -> ComposeResult:
        type(self).compose_calls += 1
        return []


class _Branch(Widget):
    compose_calls = 0

    def compose(self) -> ComposeResult:
        type(self).compose_calls += 1
        yield _Leaf("leaf")


class _InvalidCompose(Widget):
    def compose(self) -> ComposeResult:
        return 1  # type: ignore[return-value]


class _BrokenCompose(Widget):
    compose_calls = 0

    def compose(self) -> ComposeResult:
        type(self).compose_calls += 1
        raise RuntimeError("compose failed")


class _PrecomposeApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Static("root", id="root")


async def test_precompose_tree_mounts_complete_forest_without_recomposing() -> None:
    _Branch.compose_calls = 0
    _Leaf.compose_calls = 0
    async with _PrecomposeApp().run_test() as pilot:
        root = pilot.app.query_one("#root", Static)
        branches = [_Branch(), _Branch()]

        precompose_tree(branches)
        await root.mount(*branches)
        await pilot.pause()

        assert len(root.query(_Branch)) == 2
        assert len(root.query(_Leaf)) == 2
        assert _Branch.compose_calls == 2
        assert _Leaf.compose_calls == 2


def test_precompose_runtime_patch_is_idempotent() -> None:
    apply_runtime_patch()
    first = Widget._compose
    apply_runtime_patch()

    assert Widget._compose is first


async def test_precompose_tree_wraps_invalid_compose_result_with_widget_context() -> None:
    async with _PrecomposeApp().run_test():
        with pytest.raises(TypeError, match=r"_InvalidCompose.*compose\(\) method returned an invalid result"):
            precompose_tree([_InvalidCompose()])


async def test_precompose_tree_routes_compose_error_without_recomposing(monkeypatch: pytest.MonkeyPatch) -> None:
    _BrokenCompose.compose_calls = 0
    async with _PrecomposeApp().run_test() as pilot:
        root = pilot.app.query_one("#root", Static)
        errors: list[Exception] = []
        monkeypatch.setattr(pilot.app, "_handle_exception", errors.append)
        broken = _BrokenCompose()

        precompose_tree([broken])
        await root.mount(broken)
        await pilot.pause()

        assert len(errors) == 1
        assert str(errors[0]) == "compose failed"
        assert _BrokenCompose.compose_calls == 1
        assert broken.is_mounted

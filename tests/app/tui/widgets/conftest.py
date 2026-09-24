# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Fixtures shared by the widget test modules."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from chrys.app.tui.widgets.chat import tool_renderers


@pytest.fixture
def restore_kind_renderer_registry() -> Iterator[None]:
    """Snapshot/restore the tool renderer registries around a test.

    Tests that call ``register_kind_renderer(...)`` directly opt in (typically
    via a module-level ``pytestmark = pytest.mark.usefixtures(...)``) so the
    registration cannot leak into sibling modules under xdist ordering.
    """
    tool_renderers._ensure_loaded()
    saved_registry = dict(tool_renderers._REGISTRY)
    saved_kind_registry = dict(tool_renderers._KIND_REGISTRY)
    saved_loaded = tool_renderers._loaded
    yield
    tool_renderers._REGISTRY.clear()
    tool_renderers._REGISTRY.update(saved_registry)
    tool_renderers._KIND_REGISTRY.clear()
    tool_renderers._KIND_REGISTRY.update(saved_kind_registry)
    tool_renderers._loaded = saved_loaded

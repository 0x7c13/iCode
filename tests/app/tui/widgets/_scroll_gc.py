# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Fake GC hooks for chat-panel scroll tests."""

from __future__ import annotations

from typing import ClassVar

import pytest


def install_fake_chat_panel_gc(monkeypatch: pytest.MonkeyPatch):
    """Replace chat-panel GC hooks with a small stateful fake for scroll tests."""
    import chrys.app.tui.widgets.chat.scroll_controller as scroll_controller_module

    class _FakeGC:
        enabled = True
        disable_calls = 0
        enable_calls = 0
        collect_generations: ClassVar[list[int]] = []

        @classmethod
        def isenabled(cls) -> bool:
            return cls.enabled

        @classmethod
        def disable(cls) -> None:
            cls.disable_calls += 1
            cls.enabled = False

        @classmethod
        def enable(cls) -> None:
            cls.enable_calls += 1
            cls.enabled = True

        @classmethod
        def collect(cls, generation: int = 2) -> int:
            cls.collect_generations.append(generation)
            return 0

    monkeypatch.setattr(scroll_controller_module, "gc", _FakeGC)
    monkeypatch.setattr(scroll_controller_module, "_SCROLL_GC_PAUSE_OWNERS", 0)
    monkeypatch.setattr(scroll_controller_module, "_SCROLL_GC_WAS_ENABLED", False)
    return scroll_controller_module, _FakeGC

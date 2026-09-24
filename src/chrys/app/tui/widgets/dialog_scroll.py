# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared scrollable body for auto-height dialogs."""

from __future__ import annotations

from textual.containers import VerticalScroll


class StableAutoHeightScroll(VerticalScroll):
    """Auto-height dialog body whose content width is stable as scrolling toggles."""

    DEFAULT_CSS = """
    StableAutoHeightScroll {
        width: 100%;
        height: auto;
        max-height: 1fr;
        overflow-y: auto;
        /* Keep wrapped content from changing height when the scrollbar appears. */
        padding: 0 0 0 1;
        scrollbar-size: 1 1;
        scrollbar-gutter: stable;
    }
    """

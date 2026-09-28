# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The block-border color cache must not keep removed widgets' rendered lines alive."""

from __future__ import annotations

import gc
import weakref

from textual.app import App, ComposeResult
from textual.widgets import Static

from chrys.foundation.patches.textual_block_border import apply_runtime_patch


class _Host(App[None]):
    CSS = "Static { background: #123456; border: block #654321; }"

    def compose(self) -> ComposeResult:
        yield Static("rendered", id="probe")


async def test_rendered_lines_do_not_outlive_their_app() -> None:
    apply_runtime_patch()
    app = _Host()
    async with app.run_test() as pilot:
        await pilot.pause()
        cache = app.query_one("#probe", Static)._styles_cache
        assert cache._cache  # Rendering read the colors through the patched cache.
        released = weakref.ref(cache)
        del cache
    del app, pilot
    gc.collect()
    assert released() is None

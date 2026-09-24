# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the image compression progress dialog."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Static

from chrys.app.tui.i18n import LocaleController
from chrys.app.tui.screens.dialogs.image_compression import ImageCompressionDialog
from chrys.foundation.config.settings import Settings
from chrys.foundation.events.types import ImageAttachmentCompressionFinished, ImageAttachmentCompressionStarted
from tests.support.tui_helpers import make_backend_handler


def test_image_compression_default_title_keeps_english_and_localizes_chinese() -> None:
    assert ImageCompressionDialog()._title == "Preparing Image"
    controller = LocaleController(Settings(locale="zh-Hans"))
    assert ImageCompressionDialog(locale_controller=controller)._title == "正在准备图像"


@pytest.mark.asyncio
async def test_image_compression_finish_dismisses_once() -> None:
    class _Harness(App):
        def __init__(self) -> None:
            super().__init__()
            self.dialog = ImageCompressionDialog()
            self.results: list[None] = []

        def compose(self) -> ComposeResult:
            yield Static("base")

        def on_mount(self) -> None:
            self.push_screen(self.dialog, self.results.append)

    async with _Harness().run_test() as pilot:
        await pilot.pause()

        pilot.app.dialog.finish()
        pilot.app.dialog.finish()
        await pilot.pause()

    assert pilot.app.results == [None]


@pytest.mark.asyncio
async def test_image_compression_finish_preserves_replacement_cover_until_topmost() -> None:
    dialog = ImageCompressionDialog()
    results: list[None] = []

    class _Cover(ModalScreen[None]):
        def compose(self) -> ComposeResult:
            yield Static("cover")

    class _Harness(App):
        def compose(self) -> ComposeResult:
            yield Static("base")

    app = _Harness()
    async with app.run_test() as pilot:
        await app.push_screen(dialog, callback=results.append)
        first_cover = _Cover()
        await app.push_screen(first_cover)
        await pilot.pause()

        dialog.finish()
        assert dialog.query_one("#image-compression-container").display is False
        first_cover.dismiss(None)
        second_cover = _Cover()
        await app.push_screen(second_cover)
        await pilot.pause()

        assert app.screen is second_cover
        assert dialog in app.screen_stack
        assert results == []

        second_cover.dismiss(None)
        await pilot.pause()

        assert dialog not in app.screen_stack
        assert results == [None]


def test_image_compression_modal_opens_and_closes() -> None:
    pushed: list[object] = []
    debug_calls: list[tuple[str, str]] = []

    class _FakeApp:
        def push_screen(self, screen: object) -> None:
            pushed.append(screen)

    screen = SimpleNamespace(
        app=_FakeApp(),
        _debug=lambda key, message: debug_calls.append((key, message)),
    )
    handler = make_backend_handler(screen)
    handler._image_compression_dialog = None

    asyncio.run(handler.on_image_attachment_compression_started(ImageAttachmentCompressionStarted(image_count=2)))

    assert len(pushed) == 1
    dialog = pushed[0]
    assert dialog._title == "Preparing Images"
    assert handler._image_compression_dialog is dialog

    asyncio.run(handler.on_image_attachment_compression_finished(ImageAttachmentCompressionFinished(image_count=2)))

    assert handler._image_compression_dialog is None
    assert dialog._dismiss_pending is True
    assert debug_calls == [("ImageCompressionStarted", "2"), ("ImageCompressionFinished", "2")]

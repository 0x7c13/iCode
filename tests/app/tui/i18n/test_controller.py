# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for literal TUI localization adapters and bounded locale switching."""

from __future__ import annotations

import dataclasses
import gc
import os
import weakref
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.text import Text
from textual.content import Content

from chrys.app.tui import i18n as tui_i18n
from chrys.app.tui.i18n import (
    LocaleController,
    LocaleSwitchStatus,
    render_content,
    render_str,
    render_text,
)
from chrys.app.tui.language import LANGUAGE_PICKER_TITLE
from chrys.foundation.config.settings import Settings
from chrys.foundation.config.settings_store import LoadedSettings, SettingsHandle
from chrys.foundation.config.spec import Source
from chrys.foundation.i18n import DisplayBlock, DisplayPath, DisplaySequence, Localizer, MessageRef, msg
from chrys.orchestration.startup import catalog_load_warning

_DISPLAY_VALUE = msg("tests.tui_adapter_value", fallback="Value: {value}")
_DISPLAY_BLOCK = msg("tests.tui_adapter_block", fallback="{value}", multiline=True)


class _Surface:
    def __init__(self, calls: list[tuple[str, str, int]], controller: LocaleController) -> None:
        self._calls = calls
        self._controller = controller

    def refresh_localization(self) -> None:
        self._calls.append(("surface", self._controller.effective_locale, self._controller.revision))


class _Footer:
    def __init__(self, calls: list[tuple[str, str, int]], controller: LocaleController) -> None:
        self._calls = calls
        self._controller = controller

    def invalidate_localization(self) -> None:
        self._calls.append(("footer", self._controller.effective_locale, self._controller.revision))


class _TranscriptSentinel:
    def __init__(self) -> None:
        self.refresh_count = 0

    def refresh_localization(self) -> None:
        self.refresh_count += 1


@pytest.fixture(autouse=True)
def isolated_locale_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure direct live-environment mirroring is restored after every test."""
    monkeypatch.setenv("CHRYS_LOCALE", "")
    monkeypatch.delenv("CHRYS_LOCALE")


@pytest.fixture
def fake_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Redirect locale persistence to a temporary config directory."""
    from chrys.foundation import platform as platform_mod

    config_dir = tmp_path / "config"
    fake_platform = dataclasses.replace(platform_mod.get_platform(), config_dir=config_dir)
    monkeypatch.setattr(platform_mod, "get_platform", lambda: fake_platform)
    yield config_dir


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("before\x1b[31mafter", "before�[31mafter"),
        (DisplaySequence(("before\rafter",)), "before�after"),
        (DisplayPath("before\nafter"), "before�after"),
    ],
    ids=["escape-csi-scalar", "carriage-return-sequence", "line-feed-path"],
)
@pytest.mark.parametrize("adapter", [render_text, render_content], ids=["rich-text", "textual-content"])
def test_tui_adapters_replace_controls_in_bound_display_arguments(
    value: str | DisplaySequence | DisplayPath,
    expected: str,
    adapter: Callable[[Localizer, MessageRef], Text | Content],
) -> None:
    reference = _DISPLAY_VALUE.bind(value=value)

    rendered = adapter(Localizer("en"), reference)

    assert rendered.plain == f"Value: {expected}"


def test_plain_string_adapter_validates_multiline_display_text() -> None:
    reference = _DISPLAY_BLOCK.bind(value=DisplayBlock("first\nsecond\x1b[31m"))

    assert render_str(Localizer("en"), reference) == "first\nsecond�[31m"


def test_effective_switch_installs_bundle_before_one_registered_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(locale="en")
    controller = LocaleController(settings)
    calls: list[tuple[str, str, int]] = []
    surface = _Surface(calls, controller)
    footer = _Footer(calls, controller)
    persisted: list[str] = []

    def _persist(locale: str) -> None:
        persisted.append(locale)
        os.environ["CHRYS_LOCALE"] = locale

    monkeypatch.setattr(tui_i18n, "persist_locale", _persist)
    controller.register_surface(surface)
    controller.register_footer(footer)

    result = controller.switch_locale("zh-Hans")

    assert result.status is LocaleSwitchStatus.EFFECTIVE_CHANGED
    assert result.warning is None
    assert controller.requested_locale == controller.localizer.requested_locale == "zh-Hans"
    assert controller.effective_locale == "zh-Hans"
    assert controller.revision == 1
    assert persisted == ["zh-Hans"]
    assert calls == [("surface", "zh-Hans", 1), ("footer", "zh-Hans", 1)]


def test_same_effective_requested_update_only_persists_preference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("chrys.foundation.i18n.localizer.normalize_locale", lambda _requested: "en")
    settings = Settings(locale="system")
    controller = LocaleController(settings)
    calls: list[tuple[str, str, int]] = []
    surface = _Surface(calls, controller)
    footer = _Footer(calls, controller)
    persisted: list[str] = []

    def _persist(locale: str) -> None:
        persisted.append(locale)
        os.environ["CHRYS_LOCALE"] = locale

    monkeypatch.setattr(tui_i18n, "persist_locale", _persist)
    controller.register_surface(surface)
    controller.register_footer(footer)

    result = controller.switch_locale("en")

    assert result.status is LocaleSwitchStatus.REQUESTED_ONLY
    assert controller.requested_locale == controller.localizer.requested_locale == "en"
    assert controller.effective_locale == "en"
    assert controller.revision == 0
    assert persisted == ["en"]
    assert calls == []


def test_identical_requested_value_is_a_complete_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(locale="en")
    controller = LocaleController(settings)
    calls: list[tuple[str, str, int]] = []
    surface = _Surface(calls, controller)
    footer = _Footer(calls, controller)
    persisted: list[str] = []
    monkeypatch.setattr(tui_i18n, "persist_locale", persisted.append)
    controller.register_surface(surface)
    controller.register_footer(footer)

    result = controller.switch_locale("en")

    assert result.status is LocaleSwitchStatus.IDENTICAL_REQUEST
    assert controller.requested_locale == controller.effective_locale == "en"
    assert controller.revision == 0
    assert persisted == []
    assert calls == []


def test_catalog_build_failure_retains_old_locale_and_returns_old_locale_notice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_root = tmp_path / "catalogs"
    catalog_path = catalog_root / "zh-Hans" / "LC_MESSAGES" / "chrys.mo"
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_bytes(b"not a catalog")
    settings = Settings(locale="en")
    controller = LocaleController(settings, localizer=Localizer("en", catalog_root=catalog_root))
    calls: list[tuple[str, str, int]] = []
    surface = _Surface(calls, controller)
    persisted: list[str] = []
    monkeypatch.setattr(tui_i18n, "persist_locale", persisted.append)
    controller.register_surface(surface)

    result = controller.switch_locale("zh-Hans")

    assert result.status is LocaleSwitchStatus.LOAD_FAILED
    assert result.warning is not None
    assert controller.requested_locale == controller.localizer.requested_locale == "en"
    assert controller.effective_locale == "en"
    assert controller.revision == 0
    assert persisted == []
    assert calls == []
    notice = catalog_load_warning(result.warning)
    assert notice.display_message is not None
    assert (
        controller.localizer.render(notice.display_message)
        == "Could not load translations for locale zh-Hans; using English."
    )


def test_effective_switch_survives_persistence_write_failure(
    fake_config_dir: Path,
) -> None:
    settings = Settings(locale="en")
    controller = LocaleController(settings)
    calls: list[tuple[str, str, int]] = []
    surface = _Surface(calls, controller)
    controller.register_surface(surface)

    with patch("chrys.foundation.config.settings_store.update_yaml_doc", side_effect=OSError("disk full")):
        result = controller.switch_locale("zh-Hans")

    assert result.status is LocaleSwitchStatus.EFFECTIVE_CHANGED
    assert controller.revision == 1
    assert controller.effective_locale == "zh-Hans"
    assert controller.requested_locale == "zh-Hans"
    assert calls == [("surface", "zh-Hans", 1)]
    # The failed write left no durable or environment trace behind.
    assert not (fake_config_dir / "settings.yaml").exists()
    assert "CHRYS_LOCALE" not in os.environ


def test_same_effective_update_survives_persistence_write_failure(
    fake_config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("chrys.foundation.i18n.localizer.normalize_locale", lambda _requested: "en")
    settings = Settings(locale="system")
    controller = LocaleController(settings)
    calls: list[tuple[str, str, int]] = []
    surface = _Surface(calls, controller)
    controller.register_surface(surface)

    with patch("chrys.foundation.config.settings_store.update_yaml_doc", side_effect=OSError("disk full")):
        result = controller.switch_locale("en")

    assert result.status is LocaleSwitchStatus.REQUESTED_ONLY
    assert controller.revision == 0
    assert controller.effective_locale == "en"
    assert controller.requested_locale == "en"
    assert calls == []
    assert not (fake_config_dir / "settings.yaml").exists()
    assert "CHRYS_LOCALE" not in os.environ


def test_surface_registration_is_weak(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(locale="en")
    controller = LocaleController(settings)
    calls: list[tuple[str, str, int]] = []
    surface = _Surface(calls, controller)
    surface_reference = weakref.ref(surface)
    controller.register_surface(surface)
    del surface
    gc.collect()
    monkeypatch.setattr(tui_i18n, "persist_locale", lambda locale: os.environ.__setitem__("CHRYS_LOCALE", locale))

    result = controller.switch_locale("zh-Hans")

    assert result.status is LocaleSwitchStatus.EFFECTIVE_CHANGED
    assert surface_reference() is None
    assert calls == []


def test_refresh_pass_reaches_live_surface_without_touching_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(locale="en")
    controller = LocaleController(settings)
    calls: list[tuple[str, str, int]] = []
    surface = _Surface(calls, controller)
    transcript = _TranscriptSentinel()
    controller.register_surface(surface)
    monkeypatch.setattr(tui_i18n, "persist_locale", lambda locale: os.environ.__setitem__("CHRYS_LOCALE", locale))

    result = controller.switch_locale("zh-Hans")

    assert result.status is LocaleSwitchStatus.EFFECTIVE_CHANGED
    assert calls == [("surface", "zh-Hans", 1)]
    assert transcript.refresh_count == 0


def test_registered_surface_replaces_text_while_unregistered_transcript_stays_fixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(locale="en")
    controller = LocaleController(settings)
    transcript = _TranscriptSentinel()

    class _LocalizedSurface:
        def __init__(self) -> None:
            self.text = render_str(controller.localizer, LANGUAGE_PICKER_TITLE.bind())
            self.refresh_count = 0

        def refresh_localization(self) -> None:
            self.text = render_str(controller.localizer, LANGUAGE_PICKER_TITLE.bind())
            self.refresh_count += 1

    surface = _LocalizedSurface()
    controller.register_surface(surface)
    monkeypatch.setattr(tui_i18n, "persist_locale", lambda locale: os.environ.__setitem__("CHRYS_LOCALE", locale))

    result = controller.switch_locale("zh-Hans")

    assert result.status is LocaleSwitchStatus.EFFECTIVE_CHANGED
    assert surface.text == "显示语言"
    assert surface.refresh_count == 1
    assert transcript.refresh_count == 0


def test_reloaded_locale_is_reported_live_and_stays_switchable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The controller reads the shared handle, not a construction-time snapshot.

    A settings reload installs a new base into the handle the app shares with
    the engine. A snapshot would keep answering with the old locale while
    ``app._settings`` already claims the new one — and the old identical-value
    guard, comparing against the snapshot's successor, would then short-circuit
    the picker pick that is supposed to finish the switch.
    """
    handle = SettingsHandle(LoadedSettings(settings=Settings(locale="en"), provenance={}))
    controller = LocaleController(settings_handle=handle)
    persisted: list[str] = []
    monkeypatch.setattr(tui_i18n, "persist_locale", persisted.append)

    handle.install(handle.loaded.overlay(Source.ENV, locale="zh-Hans"))

    # Settings moved; the bundle deliberately did not (hot-applying LIVE
    # fields on reload is apply routing, not this controller's job).
    assert controller.requested_locale == "zh-Hans"
    assert controller.effective_locale == "en"

    result = controller.switch_locale("zh-Hans")

    assert result.status is LocaleSwitchStatus.EFFECTIVE_CHANGED
    assert controller.effective_locale == "zh-Hans"
    assert persisted == ["zh-Hans"]


def test_picked_locale_survives_a_later_settings_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A picker choice is a runtime override, so a reload cannot un-switch it."""
    handle = SettingsHandle(LoadedSettings(settings=Settings(locale="en"), provenance={}))
    controller = LocaleController(settings_handle=handle)
    monkeypatch.setattr(tui_i18n, "persist_locale", lambda locale: None)
    result = controller.switch_locale("zh-Hans")
    assert result.status is LocaleSwitchStatus.EFFECTIVE_CHANGED

    handle.install(LoadedSettings(settings=Settings(locale="en"), provenance={}))

    assert controller.requested_locale == "zh-Hans"
    assert controller.effective_locale == "zh-Hans"


def test_controller_rejects_two_different_settings_sources() -> None:
    handle = SettingsHandle(LoadedSettings(settings=Settings(locale="en"), provenance={}))

    with pytest.raises(ValueError, match="not two different ones"):
        LocaleController(Settings(locale="zh-Hans"), settings_handle=handle)


def test_rapid_identical_confirmations_refresh_and_persist_only_first_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = LocaleController(Settings(locale="en"))
    calls: list[tuple[str, str, int]] = []
    surface = _Surface(calls, controller)
    footer = _Footer(calls, controller)
    persisted: list[str] = []
    controller.register_surface(surface)
    controller.register_footer(footer)
    monkeypatch.setattr(tui_i18n, "persist_locale", persisted.append)

    results = [controller.switch_locale("zh-Hans") for _ in range(4)]

    assert [result.status for result in results] == [
        LocaleSwitchStatus.EFFECTIVE_CHANGED,
        LocaleSwitchStatus.IDENTICAL_REQUEST,
        LocaleSwitchStatus.IDENTICAL_REQUEST,
        LocaleSwitchStatus.IDENTICAL_REQUEST,
    ]
    assert controller.revision == 1
    assert persisted == ["zh-Hans"]
    assert calls == [("surface", "zh-Hans", 1), ("footer", "zh-Hans", 1)]


@pytest.mark.asyncio
async def test_status_mounted_after_switch_uses_event_time_locale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from textual.app import App, ComposeResult

    from chrys.app.tui.widgets.chat.messages import InterruptedMessage
    from chrys.app.tui.widgets.chat.panel import ChatPanel

    controller = LocaleController(Settings(locale="en"))
    monkeypatch.setattr(tui_i18n, "persist_locale", lambda locale: os.environ.__setitem__("CHRYS_LOCALE", locale))

    class _PanelApp(App[None]):
        def compose(self) -> ComposeResult:
            yield ChatPanel(localization=controller.localizer)

    async with _PanelApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.add_interrupted("Execution interrupted", "user")
        await pilot.pause()
        old_status = panel.query_one(InterruptedMessage)
        old_text = old_status._render_text().plain

        result = controller.switch_locale("zh-Hans")

        assert result.status is LocaleSwitchStatus.EFFECTIVE_CHANGED
        assert old_status._render_text().plain == old_text == "⚠ Interrupted\nExecution interrupted by user"

        await panel.add_interrupted("执行已中断", "user")
        await pilot.pause()
        new_status = panel.query_one(InterruptedMessage)

        assert new_status._render_text().plain == "⚠ 已中断\n用户中断：执行已中断"  # noqa: RUF001

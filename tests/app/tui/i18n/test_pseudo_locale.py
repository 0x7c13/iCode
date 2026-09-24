# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Pseudo-locale layout pressure through the real TUI localization path."""

from __future__ import annotations

from typing import ClassVar

import pytest
from scripts import i18n
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets._footer import FooterKey

from chrys.app.tui import binding_display
from chrys.app.tui.binding_display import CLOSE_BINDING, COPY_BINDING, DELETE_BINDING, localized_binding
from chrys.app.tui.i18n import LocaleController
from chrys.app.tui.widgets.chrome.footer import ChrysFooter
from chrys.app.tui.widgets.chrome.status_bar import STATUS_DETAILS_TOOLTIP, STATUS_RUNNING_TOOL
from chrys.foundation.config.settings import Settings
from chrys.foundation.i18n import Localizer
from chrys.foundation.i18n.locale import normalize_locale
from tests.support.ci import CI_LINUX_ONLY
from tests.support.waiting import wait_for

# Every test here funnels through the module fixture's full-repo extraction.
pytestmark = CI_LINUX_ONLY


@pytest.fixture(scope="module")
def pseudo_localizer(tmp_path_factory: pytest.TempPathFactory) -> Localizer:
    pseudo_output = tmp_path_factory.mktemp("pseudo-output")
    generated_mo = i18n.generate_pseudo_catalog(pseudo_output)

    catalog_root = tmp_path_factory.mktemp("pseudo-catalog-root")
    relocated_mo = catalog_root / "zh-Hans" / "LC_MESSAGES" / "chrys.mo"
    relocated_mo.parent.mkdir(parents=True)
    generated_mo.replace(relocated_mo)

    # User-facing normalization rejects en-XA by design. Load its generated
    # bundle through the real catalog-root path under a supported locale
    # identity, without a hand-built bundle or a production en-XA path.
    return Localizer("zh-Hans", catalog_root=catalog_root)


@pytest.fixture
def isolated_binding_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(binding_display, "_BINDING_DISPLAY_REGISTRY", {})


def test_generated_pseudo_catalog_wraps_expands_and_preserves_placeholders(
    pseudo_localizer: Localizer,
) -> None:
    bound_value = "shell --verbatim-flag"
    reference = STATUS_RUNNING_TOOL.bind(tool_name=bound_value)
    english = STATUS_RUNNING_TOOL.fallback.format(tool_name=bound_value)
    rendered = pseudo_localizer.render(reference)

    assert normalize_locale(i18n.PSEUDO_LOCALE) != i18n.PSEUDO_LOCALE
    assert rendered != english
    assert rendered == i18n.pseudo_localize(STATUS_RUNNING_TOOL.fallback).format(tool_name=bound_value)
    assert rendered.startswith("«") and rendered.endswith("··»")
    assert len(rendered) > len(english)
    assert bound_value in rendered


@pytest.mark.asyncio
async def test_pseudo_localized_footer_keys_use_natural_uncropped_width(
    pseudo_localizer: Localizer,
    isolated_binding_registry: None,
) -> None:
    definitions = (CLOSE_BINDING, COPY_BINDING, DELETE_BINDING)
    bindings = [
        localized_binding(key, action, definition, tooltip=STATUS_DETAILS_TOOLTIP)
        for key, action, definition in zip(("a", "b", "d"), ("close", "copy", "delete"), definitions, strict=True)
    ]
    controller = LocaleController(Settings(locale="zh-Hans"), localizer=pseudo_localizer)

    class PseudoFooterApp(App[None]):
        ENABLE_COMMAND_PALETTE = False
        BINDINGS: ClassVar[list[Binding]] = bindings

        def compose(self) -> ComposeResult:
            yield ChrysFooter(locale_controller=controller)

    app = PseudoFooterApp()
    expected_descriptions = {
        binding.action: pseudo_localizer.render(definition.bind())
        for binding, definition in zip(bindings, definitions, strict=True)
    }
    english_descriptions = {
        binding.action: definition.fallback for binding, definition in zip(bindings, definitions, strict=True)
    }
    expected_tooltip = pseudo_localizer.render(STATUS_DETAILS_TOOLTIP.bind())

    async with app.run_test(size=(120, 40)) as pilot:
        await wait_for(
            lambda: (
                len(app.query(FooterKey)) == len(bindings)
                and all(key.content_region.width > 0 for key in app.query(FooterKey))
            ),
            pilot=pilot,
            description="pseudo-localized footer geometry",
        )

        for key in app.query(FooterKey):
            assert key.description == expected_descriptions[key.action]
            assert key.description.startswith("«") and key.description.endswith("··»")
            assert len(key.description) > len(english_descriptions[key.action])
            assert key.tooltip == expected_tooltip
            assert str(key.styles.width) == "auto"
            assert key.content_region.width == key.render().cell_len


def test_pseudo_catalog_generation_leaves_repository_catalog_roots_clean(
    pseudo_localizer: Localizer,
) -> None:
    del pseudo_localizer  # Ensure the generation fixture has completed first.
    locale_directories = {entry.name for entry in (i18n.REPO_ROOT / "locales").iterdir() if entry.is_dir()}
    packaged_entries = {
        entry.name for entry in (i18n.REPO_ROOT / "src" / "chrys" / "foundation" / "i18n" / "_catalogs").iterdir()
    }

    assert "en-XA" not in locale_directories
    assert packaged_entries == {"zh-Hans"}

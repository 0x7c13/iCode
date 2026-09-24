# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Preparing settings overlays cannot partially replace the live handle."""

from __future__ import annotations

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.config.settings_store import LoadedSettings, SettingsHandle


@pytest.mark.parametrize("operation", ["prepare", "install"])
def test_failed_overlay_preserves_both_installed_settings_references(operation, monkeypatch) -> None:
    base = LoadedSettings(settings=Settings(), provenance={})
    handle = SettingsHandle(base)
    handle.override(theme="chrys-legacy")
    effective = handle.loaded
    candidate = LoadedSettings(settings=Settings(locale="zh-Hans"), provenance={})

    def fail_overlay(*args, **kwargs):
        raise ValueError("overlay rejected")

    monkeypatch.setattr(LoadedSettings, "overlay", fail_overlay)
    with pytest.raises(ValueError, match="overlay rejected"):
        if operation == "prepare":
            handle.prepare(candidate)
        else:
            handle.install(candidate)
    assert handle._base is base
    assert handle.loaded is effective


def test_install_composition_matches_prepared_install_without_recomputing_overlay(monkeypatch) -> None:
    base = LoadedSettings(settings=Settings(), provenance={})
    candidate = LoadedSettings(settings=Settings(locale="zh-Hans"), provenance={})
    combined, split = SettingsHandle(base), SettingsHandle(base)
    for handle in [combined, split]:
        handle.override(theme="chrys-legacy")
    old = split.loaded
    prepared = split.prepare(candidate)
    assert split._base is base
    assert split.loaded is old
    assert prepared.base is candidate
    combined.install(candidate)

    def fail_overlay(*args, **kwargs):
        raise AssertionError("installation recomputed the overlay")

    monkeypatch.setattr(LoadedSettings, "overlay", fail_overlay)
    split.install_prepared(prepared)
    assert split._base is combined._base is candidate
    assert split.loaded is prepared.effective
    assert split.loaded == combined.loaded
    assert split.settings.theme == "chrys-legacy"
    assert split.settings.locale == "zh-Hans"

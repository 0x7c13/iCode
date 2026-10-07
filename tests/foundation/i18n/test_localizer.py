# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for atomic Localizer bundles and safe translated rendering."""

from __future__ import annotations

import logging
import struct
from collections.abc import Mapping
from pathlib import Path

import pytest

from chrys.foundation.i18n import CatalogLoadWarning, Localizer, msg
from chrys.foundation.i18n import localizer as localizer_module

_GREETING = msg("test.localizer.greeting", fallback="Hello, {name}")
_FILES = msg("test.localizer.files", fallback="One file", plural_fallback="{count} files")
_NO_SLOT = msg("test.localizer.no_slot", fallback="English")
_VISIBLE_ENGLISH = msg("test.localizer.visible_english", fallback="Value: {value}")


_DEFAULT_METADATA = (
    "Content-Type: text/plain; charset=UTF-8\nPlural-Forms: nplurals=2; plural=(n != 1);\nLanguage: zh-Hans\n"
)


def _mo_bytes(entries: Mapping[str, str | tuple[str, ...]], *, metadata: str = _DEFAULT_METADATA) -> bytes:
    encoded: list[tuple[bytes, bytes]] = [(b"", metadata.encode())]
    for key, value in entries.items():
        original = key.encode()
        if type(value) is tuple:
            original += b"\0" + f"{key}#plural".encode()
            translation = "\0".join(value).encode()
        else:
            translation = value.encode()
        encoded.append((original, translation))
    encoded.sort(key=lambda item: item[0])

    count = len(encoded)
    originals_offset = 28
    translations_offset = originals_offset + count * 8
    strings_offset = translations_offset + count * 8
    originals = bytearray()
    translations = bytearray()
    original_table: list[tuple[int, int]] = []
    translation_table: list[tuple[int, int]] = []
    for original, _translation in encoded:
        original_table.append((len(original), strings_offset + len(originals)))
        originals.extend(original + b"\0")
    translations_base = strings_offset + len(originals)
    for _original, translation in encoded:
        translation_table.append((len(translation), translations_base + len(translations)))
        translations.extend(translation + b"\0")

    header = struct.pack(
        "<7I",
        0x950412DE,
        0,
        count,
        originals_offset,
        translations_offset,
        0,
        0,
    )
    tables = b"".join(struct.pack("<2I", *item) for item in original_table + translation_table)
    return header + tables + originals + translations


def _write_catalog(root: Path, entries: Mapping[str, str | tuple[str, ...]] | bytes) -> Path:
    path = root / "zh-Hans" / "LC_MESSAGES" / "chrys.mo"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(entries if type(entries) is bytes else _mo_bytes(entries))
    return path


_UNPARSEABLE_PLURAL_MO = _mo_bytes(
    {_GREETING.key: "你好, {name}"},
    metadata="Content-Type: text/plain; charset=UTF-8\nPlural-Forms: nplurals=2; plural=(n;\n",
)


def test_english_baseline_performs_no_catalog_io(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("English construction must not load a catalog")

    monkeypatch.setattr(localizer_module, "load_catalog", forbidden_load)

    localizer = Localizer("en")

    assert localizer.requested_locale == "en"
    assert localizer.effective_locale == "en"
    assert localizer.first_load_warning is None
    assert localizer.render(_GREETING.bind(name="Chrys")) == "Hello, Chrys"


def test_valid_catalog_loads_and_renders_translation(tmp_path: Path) -> None:
    _write_catalog(tmp_path, {_GREETING.key: "你好, {name}"})

    localizer = Localizer("zh-Hans", catalog_root=tmp_path)

    assert localizer.requested_locale == "zh-Hans"
    assert localizer.effective_locale == "zh-Hans"
    assert localizer.first_load_warning is None
    assert localizer.render(_GREETING.bind(name="Chrys")) == "你好, Chrys"


@pytest.mark.parametrize("catalog", [None, b"truncated", _UNPARSEABLE_PLURAL_MO])
def test_first_load_failure_keeps_usable_english_and_typed_warning(
    tmp_path: Path,
    catalog: bytes | None,
) -> None:
    if catalog is not None:
        _write_catalog(tmp_path, catalog)

    localizer = Localizer("zh-Hans", catalog_root=tmp_path)

    assert localizer.requested_locale == "zh-Hans"
    assert localizer.effective_locale == "en"
    assert localizer.first_load_warning == CatalogLoadWarning(requested_locale="zh-Hans")
    assert localizer.render(_GREETING.bind(name="Chrys")) == "Hello, Chrys"


def test_system_first_load_failure_preserves_system_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(localizer_module, "normalize_locale", lambda value: "zh-Hans")

    localizer = Localizer("system", catalog_root=tmp_path)

    assert localizer.requested_locale == "system"
    assert localizer.effective_locale == "en"
    assert localizer.first_load_warning == CatalogLoadWarning(requested_locale="system")


@pytest.mark.parametrize("catalog", [None, b"truncated", _UNPARSEABLE_PLURAL_MO])
def test_failed_switch_keeps_previous_bundle_and_request_usable(
    tmp_path: Path,
    catalog: bytes | None,
) -> None:
    if catalog is not None:
        _write_catalog(tmp_path, catalog)
    localizer = Localizer("en", catalog_root=tmp_path)

    warning = localizer.switch_locale("zh-Hans")

    assert warning == CatalogLoadWarning(requested_locale="zh-Hans")
    assert localizer.requested_locale == "en"
    assert localizer.effective_locale == "en"
    assert localizer.render(_GREETING.bind(name="Chrys")) == "Hello, Chrys"


def test_successful_switch_atomically_installs_requested_bundle(tmp_path: Path) -> None:
    _write_catalog(tmp_path, {_GREETING.key: "你好, {name}"})
    localizer = Localizer("en", catalog_root=tmp_path)

    warning = localizer.switch_locale("zh-Hans")

    assert warning is None
    assert localizer.requested_locale == "zh-Hans"
    assert localizer.effective_locale == "zh-Hans"
    assert localizer.render(_GREETING.bind(name="Chrys")) == "你好, Chrys"


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, "0 个文件"),
        (1, "一个文件"),
        (2, "2 个文件"),
    ],
)
def test_plural_catalog_entries_render_selected_translation(tmp_path: Path, count: int, expected: str) -> None:
    _write_catalog(tmp_path, {_FILES.key: ("一个文件", "{count} 个文件")})
    localizer = Localizer("zh-Hans", catalog_root=tmp_path)

    assert localizer.render(_FILES.bind(count=count)) == expected


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, "0 files"),
        (1, "One file"),
        (2, "2 files"),
    ],
)
def test_missing_plural_entries_never_render_echoed_lookup_ids(tmp_path: Path, count: int, expected: str) -> None:
    _write_catalog(tmp_path, {"test.localizer.other": "其他"})
    localizer = Localizer("zh-Hans", catalog_root=tmp_path)

    assert localizer.render(_FILES.bind(count=count)) == expected


def test_missing_singular_entry_renders_english_fallback(tmp_path: Path) -> None:
    _write_catalog(tmp_path, {"test.localizer.other": "其他"})

    localizer = Localizer("zh-Hans", catalog_root=tmp_path)

    assert localizer.render(_GREETING.bind(name="Chrys")) == "Hello, Chrys"


@pytest.mark.parametrize(
    "translated",
    [
        "\x1b[31m不安全",
        "[bold]不安全[/bold]",
        "第一行\n第二行",
        "other.lookup",
        "\u200bother.lookup\u200b",
    ],
)
def test_tampered_mo_templates_render_english(tmp_path: Path, translated: str) -> None:
    _write_catalog(tmp_path, {_NO_SLOT.key: translated})

    localizer = Localizer("zh-Hans", catalog_root=tmp_path)

    assert localizer.render(_NO_SLOT.bind()) == "English"


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, "0 files"),
        (1, "One file"),
        (2, "2 files"),
    ],
)
def test_tampered_plural_expression_renders_english(tmp_path: Path, count: int, expected: str) -> None:
    metadata = "Content-Type: text/plain; charset=UTF-8\nPlural-Forms: nplurals=2; plural=n%0;\nLanguage: zh-Hans\n"
    _write_catalog(tmp_path, _mo_bytes({_FILES.key: ("一个文件", "{count} 个文件")}, metadata=metadata))
    localizer = Localizer("zh-Hans", catalog_root=tmp_path)

    assert localizer.render(_FILES.bind(count=count)) == expected


def test_translation_formatting_to_zero_width_output_renders_english(tmp_path: Path) -> None:
    _write_catalog(tmp_path, {_VISIBLE_ENGLISH.key: "{value}"})
    localizer = Localizer("zh-Hans", catalog_root=tmp_path)

    assert localizer.render(_VISIBLE_ENGLISH.bind(value="\u200b")) == "Value: \u200b"


def test_catalog_template_validation_is_memoized_in_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_catalog(tmp_path, {_GREETING.key: "你好, {name}"})
    original = localizer_module.validate_authored_template
    calls = 0

    def counting_validator(template: str, *, multiline: bool) -> frozenset[str]:
        nonlocal calls
        calls += 1
        return original(template, multiline=multiline)

    monkeypatch.setattr(localizer_module, "validate_authored_template", counting_validator)
    localizer = Localizer("zh-Hans", catalog_root=tmp_path)

    assert localizer.render(_GREETING.bind(name="A")) == "你好, A"
    assert localizer.render(_GREETING.bind(name="B")) == "你好, B"
    assert calls == 1


def test_invalid_entry_diagnostic_contains_no_catalog_content(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "SECRET_CATALOG_CONTENT"
    _write_catalog(tmp_path, {_NO_SLOT.key: f"\x1b{secret}"})
    localizer = Localizer("zh-Hans", catalog_root=tmp_path)
    caplog.clear()

    with caplog.at_level(logging.WARNING, logger=localizer_module.__name__):
        assert localizer.render(_NO_SLOT.bind()) == "English"

    assert [record.message for record in caplog.records] == ["Invalid i18n catalog entry; using English fallback."]
    assert secret not in caplog.text
    assert _NO_SLOT.key not in caplog.text

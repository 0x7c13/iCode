# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for class-based GNU MO catalog loading."""

from __future__ import annotations

import gettext
import struct
from collections.abc import Mapping
from pathlib import Path

import pytest

from chrys.foundation.i18n.catalogs import CatalogLoadError, load_catalog

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


def _write_catalog(root: Path, data: bytes) -> Path:
    path = root / "zh-Hans" / "LC_MESSAGES" / "chrys.mo"
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    return path


def test_load_catalog_returns_gnu_translations_for_valid_mo(tmp_path: Path) -> None:
    _write_catalog(tmp_path, _mo_bytes({"test.catalog.greeting": "你好"}))

    catalog = load_catalog("zh-Hans", catalog_root=tmp_path)

    assert isinstance(catalog, gettext.GNUTranslations)
    assert catalog.gettext("test.catalog.greeting") == "你好"


@pytest.mark.parametrize("data", [b"not an mo", struct.pack("<I", 0x950412DE)])
def test_load_catalog_rejects_corrupt_and_truncated_mo(tmp_path: Path, data: bytes) -> None:
    _write_catalog(tmp_path, data)

    with pytest.raises(CatalogLoadError):
        load_catalog("zh-Hans", catalog_root=tmp_path)


@pytest.mark.parametrize(
    "metadata",
    [
        "Content-Type: text/plain; charset=bogus-charset\n",
        "Content-Type: text/plain; charset=UTF-8\nPlural-Forms: nplurals=2; plural=(n;\n",
    ],
)
def test_load_catalog_rejects_undecodable_or_unparseable_metadata(tmp_path: Path, metadata: str) -> None:
    _write_catalog(tmp_path, _mo_bytes({"test.catalog.greeting": "你好"}, metadata=metadata))

    with pytest.raises(CatalogLoadError):
        load_catalog("zh-Hans", catalog_root=tmp_path)


def test_load_catalog_rejects_a_missing_mo(tmp_path: Path) -> None:
    with pytest.raises(CatalogLoadError):
        load_catalog("zh-Hans", catalog_root=tmp_path)


def test_load_catalog_never_installs_gettext_into_builtins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_catalog(tmp_path, _mo_bytes({"test.catalog.greeting": "你好"}))

    def forbidden_install(*args: object, **kwargs: object) -> None:
        raise AssertionError("gettext.install() must not be used")

    monkeypatch.setattr(gettext, "install", forbidden_install)

    assert load_catalog("zh-Hans", catalog_root=tmp_path).gettext("test.catalog.greeting") == "你好"

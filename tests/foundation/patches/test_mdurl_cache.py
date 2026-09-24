# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A thread normalizing a link never sees one of mdurl's tables before it is complete."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import mdurl
import pytest
from markdown_it.common.normalize_url import normalizeLink, normalizeLinkText
from mdurl import _decode, _encode

from chrys.foundation.patches import mdurl_cache as patch


class _PublicationRecorder(dict):
    """Record how full each table is at the moment another thread could first read it."""

    def __init__(self) -> None:
        super().__init__()
        self.published: list[int] = []

    def __setitem__(self, key: str, value: Any) -> None:
        self.published.append(len(value))
        super().__setitem__(key, value)

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key not in self:
            self.published.append(len(default))
        return super().setdefault(key, default)


def _pristine(module: ModuleType) -> ModuleType:
    """A fresh, unpatched copy of an mdurl module, loaded from its installed source."""
    spec = importlib.util.spec_from_file_location(f"_pristine_{module.__name__}", Path(str(module.__file__)))
    assert spec is not None and spec.loader is not None
    copy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(copy)
    return copy


def test_the_pinned_mdurl_is_patched():
    assert mdurl.__version__ == "0.1.2"
    assert getattr(_encode.get_encode_cache, patch._RUNTIME_PATCH_MARKER, False)
    assert getattr(_decode.get_decode_cache, patch._RUNTIME_PATCH_MARKER, False)


def test_a_table_is_whole_when_it_is_published(monkeypatch):
    encode_cache, decode_cache = _PublicationRecorder(), _PublicationRecorder()
    monkeypatch.setattr(_encode, "encode_cache", encode_cache)
    monkeypatch.setattr(_decode, "decode_cache", decode_cache)
    # Unpatched, both tables are published empty and filled afterwards.
    assert normalizeLink("https://www.bbc.co.uk/news/articles/c6n9w410v3vzo") == (
        "https://www.bbc.co.uk/news/articles/c6n9w410v3vzo"
    )
    assert normalizeLinkText("https://example.com/a%20b") == "https://example.com/a b"
    assert encode_cache.published == [128]
    assert decode_cache.published == [128]
    # A later call reads the published table rather than building another.
    normalizeLink("https://example.com/")
    assert encode_cache.published == [128]


@pytest.mark.parametrize(
    "exclude", [mdurl.ENCODE_DEFAULT_CHARS, mdurl.ENCODE_COMPONENT_CHARS, "", mdurl.DECODE_DEFAULT_CHARS + "%"]
)
def test_the_tables_are_the_ones_mdurl_builds(monkeypatch, exclude):
    pristine_encode, pristine_decode = _pristine(_encode), _pristine(_decode)
    monkeypatch.setattr(_encode, "encode_cache", {})
    monkeypatch.setattr(_decode, "decode_cache", {})
    assert list(_encode.get_encode_cache(exclude)) == list(pristine_encode.get_encode_cache(exclude))
    assert list(_decode.get_decode_cache(exclude)) == list(pristine_decode.get_decode_cache(exclude))


def test_another_mdurl_version_is_left_alone(monkeypatch):
    pristine_encode, pristine_decode = _pristine(_encode), _pristine(_decode)
    monkeypatch.setattr(_encode, "get_encode_cache", pristine_encode.get_encode_cache)
    monkeypatch.setattr(_decode, "get_decode_cache", pristine_decode.get_decode_cache)
    monkeypatch.setattr(mdurl, "__version__", "0.1.3")
    patch.apply_runtime_patch()
    assert _encode.get_encode_cache is pristine_encode.get_encode_cache
    assert _decode.get_decode_cache is pristine_decode.get_decode_cache

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for cross-platform path helpers."""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path

import pytest

from chrys.foundation.platform.paths import (
    resolve_cross_platform_path,
    resolve_existing_path,
    resolve_workspace_path,
)


def test_resolve_cross_platform_path_resolves_relative_against_base(tmp_path: Path) -> None:
    assert resolve_cross_platform_path("skills/path-skill", base=tmp_path) == str(
        (tmp_path / "skills" / "path-skill").resolve()
    )


def test_resolve_cross_platform_path_resolves_native_absolute(tmp_path: Path) -> None:
    path = tmp_path / "skills" / "path-skill"

    assert resolve_cross_platform_path(str(path)) == str(path.resolve())


@pytest.mark.parametrize(
    "path",
    [
        r"C:\Users\agent\Skills\path-skill",
        r"\\server\share\Skills\path-skill",
    ],
)
@pytest.mark.skipif(os.name == "nt", reason="foreign Windows path preservation is a POSIX-host behavior")
def test_resolve_cross_platform_path_preserves_foreign_windows_absolute_on_posix(path: str) -> None:
    assert resolve_cross_platform_path(path, base="/workspace") == path


def test_resolve_workspace_path_resolves_relative_against_base(tmp_path: Path) -> None:
    assert resolve_workspace_path("src/file.py", base_cwd=tmp_path) == str(tmp_path / "src" / "file.py")


@pytest.mark.parametrize(
    "path",
    [
        r"C:\Users\agent\project\file.py",
        r"\\server\share\project\file.py",
    ],
)
@pytest.mark.skipif(os.name == "nt", reason="foreign Windows path preservation is a POSIX-host behavior")
def test_resolve_workspace_path_preserves_foreign_windows_absolute_on_posix(path: str) -> None:
    assert resolve_workspace_path(path, base_cwd="/workspace") == path


# -- resolve_existing_path -------------------------------------------------


def test_resolve_existing_path_exact_match(tmp_path: Path) -> None:
    f = tmp_path / "exact.txt"
    f.write_text("hi")

    assert resolve_existing_path(str(f)) == str(f)


def test_resolve_existing_path_returns_none_when_missing(tmp_path: Path) -> None:
    assert resolve_existing_path(str(tmp_path / "missing.txt")) is None


def test_resolve_existing_path_falls_back_to_narrow_no_break_space(tmp_path: Path) -> None:
    """macOS screenshot filenames use U+202F before AM/PM."""
    real = tmp_path / "shot 9.00.00\u202fAM.png"
    real.write_bytes(b"\x89PNG\r\n\x1a\n")
    requested = str(tmp_path / "shot 9.00.00 AM.png")

    result = resolve_existing_path(requested)

    assert result == str(real)


def test_resolve_existing_path_falls_back_to_regular_space(tmp_path: Path) -> None:
    real = tmp_path / "shot 9.00.00 AM.png"
    real.write_bytes(b"\x89PNG\r\n\x1a\n")
    requested = str(tmp_path / "shot 9.00.00\u202fAM.png")

    result = resolve_existing_path(requested)

    assert result == str(real)


def test_resolve_existing_path_falls_back_to_unicode_normalization(tmp_path: Path) -> None:
    name_nfc = unicodedata.normalize("NFC", "café.txt")
    name_nfd = unicodedata.normalize("NFD", "café.txt")

    real = tmp_path / name_nfc
    real.write_text("hi")
    requested = str(tmp_path / name_nfd)

    result = resolve_existing_path(requested)

    # Filesystems may store the name in NFD, so compare by identity under NFC/NFD.
    assert result is not None
    assert Path(result).exists()
    assert unicodedata.normalize("NFC", result) == unicodedata.normalize("NFC", str(real))


def test_resolve_existing_path_rejects_ambiguous_whitespace_matches(tmp_path: Path) -> None:
    """If multiple single-whitespace variants exist, the match is ambiguous."""
    first = tmp_path / "shot\u202f9.00.00 AM.png"
    second = tmp_path / "shot 9.00.00\u202fAM.png"
    first.write_bytes(b"\x89PNG\r\n\x1a\n")
    second.write_bytes(b"\x89PNG\r\n\x1a\n")
    requested = str(tmp_path / "shot 9.00.00 AM.png")

    assert resolve_existing_path(requested) is None


def test_resolve_existing_path_composes_normalization_and_whitespace(tmp_path: Path) -> None:
    """An NFC+U+0020 request should find an NFD+U+202F file on disk."""
    name_nfd = unicodedata.normalize("NFD", "café 9.00.00\u202fAM.png")
    name_nfc_request = unicodedata.normalize("NFC", "café 9.00.00 AM.png")

    real = tmp_path / name_nfd
    real.write_bytes(b"\x89PNG\r\n\x1a\n")
    requested = str(tmp_path / name_nfc_request)

    result = resolve_existing_path(requested)

    assert result is not None
    assert Path(result).exists()
    assert unicodedata.normalize("NFC", result) == unicodedata.normalize("NFC", str(real))


def test_resolve_existing_path_does_not_use_compatibility_normalization(tmp_path: Path) -> None:
    """NFKC/NFKD are excluded to prevent approval bypasses (fullwidth dot -> '.')."""
    real = tmp_path / ".env"
    real.write_text("secret")
    requested = str(tmp_path / "\uff0eenv")  # fullwidth dot

    assert resolve_existing_path(requested) is None


def test_resolve_existing_path_skips_whitespace_substitution_when_too_many_spaces(tmp_path: Path, monkeypatch) -> None:
    """Basenames with too many whitespace positions do not generate candidates."""
    # Lower the limit so we can test the boundary with a short, cross-platform filename.
    monkeypatch.setattr("chrys.foundation.platform.paths._MAX_WHITESPACE_SUBSTITUTION_POSITIONS", 2)

    # Real file differs from the request by only one whitespace position; without
    # the limit, single-substitution would find it. With limit=2 and 3 whitespace
    # positions, substitution is skipped and we return None.
    real = tmp_path / "x x x\u202fx"
    real.write_bytes(b"\x89PNG\r\n\x1a\n")
    requested = str(tmp_path / "x x x x")

    assert resolve_existing_path(requested) is None


def test_resolve_existing_path_does_not_resolve_traversal_via_compatibility(tmp_path: Path) -> None:
    """Fullwidth parent dots/slashes must not become real traversal sequences."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    real = tmp_path / "secret.txt"
    real.write_text("secret")
    # Fullwidth dots and slash from inside workspace: U+FF0E U+FF0E U+FF0F secret.txt
    # Would normalize to ../secret.txt and escape workspace if compatibility forms were used.
    requested = str(workspace / "\uff0e\uff0e\uff0fsecret.txt")

    assert resolve_existing_path(requested) is None

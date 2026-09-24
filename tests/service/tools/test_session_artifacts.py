# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for session-scoped document artifact handles."""

from __future__ import annotations

import os
import shutil
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from chrys.foundation.config.settings import SESSION_ROOT_DIR_ENV_VAR
from chrys.foundation.platform import get_platform
from chrys.service.tools.session_artifacts import (
    DocumentImageArtifactLimitError,
    iter_document_image_artifacts,
    make_document_artifact_handle,
    make_document_image_artifact_handle,
    reharden_document_image_artifacts,
    resolve_document_image_artifact_handle,
    resolve_document_markdown_artifact_handle,
    resolve_tool_session_dir,
)


def test_document_handle_is_readable_and_resolves_exact_filename(tmp_path) -> None:
    session_dir = tmp_path / "session"
    artifact_dir = session_dir / "doc_converter"
    artifact_dir.mkdir(parents=True)
    artifact = artifact_dir / "report 你好.md"
    artifact.write_text("content", encoding="utf-8")

    handle = make_document_artifact_handle(artifact.name)

    assert handle == "chrys-session-document:report 你好.md"
    assert resolve_document_markdown_artifact_handle(handle, session_dir) == str(artifact)


def test_document_handle_resolver_ignores_normal_paths(tmp_path) -> None:
    assert resolve_document_markdown_artifact_handle("report.md", tmp_path) is None
    assert resolve_document_image_artifact_handle("page.png", tmp_path) is None


@pytest.mark.parametrize("filename", ["page.png", "photo.jpg", "photo.jpeg", "figure.webp"])
def test_document_image_handles_resolve_supported_suffixes(tmp_path, filename: str) -> None:
    session_dir = tmp_path / "session"
    artifact_dir = session_dir / "doc_converter"
    artifact_dir.mkdir(parents=True)
    artifact = artifact_dir / filename
    artifact.write_bytes(b"image")

    handle = make_document_image_artifact_handle(filename)

    assert handle == f"chrys-session-document:{filename}"
    assert resolve_document_image_artifact_handle(handle, session_dir) == str(artifact)


def test_document_artifact_writers_enforce_output_suffixes() -> None:
    with pytest.raises(ValueError):
        make_document_artifact_handle("page.png")
    with pytest.raises(ValueError):
        make_document_image_artifact_handle("report.md")


@pytest.mark.parametrize(
    "reference",
    [
        "chrys-session-document:",
        "chrys-session-document:../report.md",
        "chrys-session-document:report.txt",
        "chrys-session-document:report?.md",
        "chrys-session-document:report-\udcff.md",
        "chrys-session-document:" + "x" * 1025 + ".md",
    ],
)
def test_document_handle_rejects_invalid_filenames(tmp_path, reference: str) -> None:
    with pytest.raises(ValueError):
        resolve_document_markdown_artifact_handle(reference, tmp_path)


def test_document_handle_requires_session_directory() -> None:
    handle = make_document_artifact_handle("report.md")

    with pytest.raises(ValueError, match="without a session directory"):
        resolve_document_markdown_artifact_handle(handle, None)


def test_tool_session_dir_fallback_is_shared_and_preserves_raw_config_text(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(SESSION_ROOT_DIR_ENV_VAR, raising=False)
    import chrys.service.tools.session_artifacts as session_artifacts

    monkeypatch.setattr(
        session_artifacts,
        "resolve_sessions_dir",
        lambda config_dir, *, create: config_dir / "sessions",
    )
    config_dir = tmp_path / "config-\udcff"

    first = resolve_tool_session_dir(config_dir, session_id="session_abc")
    second = resolve_tool_session_dir(config_dir, session_id="session_abc")

    assert first == second
    assert first is not None
    assert str(first).startswith(str(config_dir))


@pytest.mark.skipif(get_platform().is_windows, reason="symlink creation is not generally available on Windows")
def test_document_handle_rejects_redirected_artifact_root(tmp_path) -> None:
    session_dir = tmp_path / "session"
    outside = tmp_path / "outside"
    session_dir.mkdir()
    outside.mkdir()
    (session_dir / "doc_converter").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="directory must not be redirected"):
        resolve_document_markdown_artifact_handle(make_document_artifact_handle("report.md"), session_dir)


@pytest.mark.skipif(get_platform().is_windows, reason="symlink creation is not generally available on Windows")
def test_document_handle_rejects_redirected_artifact_file(tmp_path) -> None:
    session_dir = tmp_path / "session"
    artifact_dir = session_dir / "doc_converter"
    artifact_dir.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    (artifact_dir / "report.md").symlink_to(outside)

    with pytest.raises(ValueError, match="file must not be redirected"):
        resolve_document_markdown_artifact_handle(make_document_artifact_handle("report.md"), session_dir)


def test_document_handle_resolves_to_its_declared_parent(tmp_path) -> None:
    session_dir = tmp_path / "session"
    artifact_dir = session_dir / "doc_converter"
    artifact_dir.mkdir(parents=True)
    resolved = resolve_document_markdown_artifact_handle(make_document_artifact_handle("report.md"), session_dir)

    assert resolved is not None
    assert os.path.dirname(resolved) == os.path.realpath(artifact_dir)


def test_document_image_artifact_scan_stops_at_file_count_limit(tmp_path) -> None:
    artifact_dir = tmp_path / "session" / "doc_converter"
    artifact_dir.mkdir(parents=True)
    for index in range(3):
        (artifact_dir / f"image-{index}.png").write_bytes(b"x")
    (artifact_dir / "report.md").write_text("ignored", encoding="utf-8")

    with pytest.raises(DocumentImageArtifactLimitError, match="file count"):
        list(iter_document_image_artifacts(artifact_dir, max_files=2))


def test_reharden_document_images_republishes_copied_windows_artifacts(tmp_path) -> None:
    import chrys.service.tools.session_artifacts as session_artifacts

    source_session_dir = tmp_path / "source"
    source_artifact_dir = source_session_dir / "doc_converter"
    source_image = source_artifact_dir / "image-copied.png"
    session_artifacts.atomic_write_owner_only_bytes(source_image, b"copied-image")
    source_markdown = source_artifact_dir / "report.md"
    source_markdown.write_text("not an image", encoding="utf-8")
    destination_session_dir = tmp_path / "destination"
    shutil.copytree(source_session_dir, destination_session_dir)
    destination_image = destination_session_dir / "doc_converter" / source_image.name
    destination_markdown = destination_session_dir / "doc_converter" / source_markdown.name

    with (
        patch.object(session_artifacts, "get_platform", return_value=SimpleNamespace(is_windows=True)),
        patch.object(
            session_artifacts,
            "secure_open_owner_verified_binary",
            wraps=session_artifacts.secure_open_owner_verified_binary,
        ) as read_source,
        patch.object(
            session_artifacts,
            "atomic_write_owner_only_bytes",
            wraps=session_artifacts.atomic_write_owner_only_bytes,
        ) as republish,
    ):
        reharden_document_image_artifacts(source_session_dir, destination_session_dir)

    read_source.assert_called_once_with(source_image)
    republish.assert_called_once_with(destination_image, b"copied-image")
    assert destination_markdown.read_text(encoding="utf-8") == "not an image"


def test_reharden_document_images_finishes_enumeration_before_republishing(tmp_path) -> None:
    import chrys.service.tools.session_artifacts as session_artifacts

    source_session_dir = tmp_path / "source"
    source_artifact_dir = source_session_dir / "doc_converter"
    source_image = source_artifact_dir / "image-copied.png"
    session_artifacts.atomic_write_owner_only_bytes(source_image, b"copied-image")
    destination_session_dir = tmp_path / "destination"
    shutil.copytree(source_session_dir, destination_session_dir)
    destination_image = destination_session_dir / "doc_converter" / source_image.name
    scan_active = False

    def scanned_artifacts():
        nonlocal scan_active
        scan_active = True
        try:
            yield destination_image, len(b"copied-image")
        finally:
            scan_active = False

    def republish(_path, _payload) -> None:
        assert not scan_active

    with (
        patch.object(session_artifacts, "get_platform", return_value=SimpleNamespace(is_windows=True)),
        patch.object(session_artifacts, "iter_document_image_artifacts", return_value=scanned_artifacts()),
        patch.object(session_artifacts, "atomic_write_owner_only_bytes", side_effect=republish),
    ):
        reharden_document_image_artifacts(source_session_dir, destination_session_dir)

    assert destination_image.read_bytes() == b"copied-image"


def test_reharden_document_images_reports_an_artifact_root_missing_from_the_copy(tmp_path) -> None:
    import chrys.service.tools.session_artifacts as session_artifacts

    source_session_dir = tmp_path / "source"
    session_artifacts.atomic_write_owner_only_bytes(
        source_session_dir / "doc_converter" / "image-copied.png", b"copied-image"
    )
    destination_session_dir = tmp_path / "destination"
    destination_session_dir.mkdir()

    with (
        patch.object(session_artifacts, "get_platform", return_value=SimpleNamespace(is_windows=True)),
        pytest.raises(OSError, match="missing from the copy"),
    ):
        reharden_document_image_artifacts(source_session_dir, destination_session_dir)


def test_reharden_document_images_ignores_a_source_without_artifacts(tmp_path) -> None:
    import chrys.service.tools.session_artifacts as session_artifacts

    source_session_dir = tmp_path / "source"
    source_session_dir.mkdir()
    destination_session_dir = tmp_path / "destination"
    destination_session_dir.mkdir()

    with patch.object(session_artifacts, "get_platform", return_value=SimpleNamespace(is_windows=True)):
        reharden_document_image_artifacts(source_session_dir, destination_session_dir)

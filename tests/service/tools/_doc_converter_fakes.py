# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared fakes, patch targets, and artifact-path helpers for the doc_converter test modules."""

from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from chrys.service.tools.builtins.doc_converter.artifacts import (
    DocumentImageSink,
)
from chrys.service.tools.builtins.doc_converter.parsers.base import ParsedDocument
from chrys.service.tools.session_artifacts import (
    resolve_document_image_artifact_handle,
    resolve_document_markdown_artifact_handle,
)

_PATCH_REGISTRY = "chrys.service.tools.builtins.doc_converter.registry"
_PATCH_TOK = "chrys.service.tools.builtins.doc_converter._tokenizer"
_PATCH_TOOL = "chrys.service.tools.builtins.doc_converter"


@dataclass
class _FakePlatformInfo:
    config_dir: Path
    os_name: str = "macos"
    arch: str = "arm64"
    shell: object = None
    data_dir: Path | None = None
    extra_shells: tuple = ()


@dataclass
class _FakeRuntime:
    platform: _FakePlatformInfo
    cwd: str = "/tmp"
    working_dirs: list = None

    def __post_init__(self):
        if self.working_dirs is None:
            self.working_dirs = []


def _make_runtime(tmp_path: Path) -> _FakeRuntime:
    """Create a minimal fake SessionEnvironment-like object."""
    platform = _FakePlatformInfo(config_dir=tmp_path)
    # A folder that exists on every host: the tool reports a missing working directory first.
    return _FakeRuntime(platform=platform, cwd=str(tmp_path))


class _FakeParser:
    """Fake parser for testing convert_document dispatch."""

    def __init__(
        self, text_content: str = "", *, side_effect: Exception | None = None, extensions: frozenset[str] | None = None
    ):
        self._text = text_content
        self._side_effect = side_effect
        self._extensions = extensions or frozenset({".pdf"})

    @property
    def supported_extensions(self) -> frozenset[str]:
        return self._extensions

    def parse(self, path: str, *, image_sink: DocumentImageSink | None = None) -> ParsedDocument:
        if self._side_effect is not None:
            raise self._side_effect
        return ParsedDocument(markdown=self._text)


class _VisualFakeParser(_FakeParser):
    """Fake parser that emits one image only when a sink is supplied."""

    def __init__(self, image_data: bytes, text_content: str = "# Report\n\ncontent") -> None:
        super().__init__(text_content)
        self._image_data = image_data
        self.last_sink: DocumentImageSink | None = None

    def parse(self, path: str, *, image_sink: DocumentImageSink | None = None) -> ParsedDocument:
        self.last_sink = image_sink
        if image_sink is None:
            return ParsedDocument(markdown=self._text)
        assert image_sink.try_reserve_occurrence()
        occurrence = image_sink.save_image(
            self._image_data,
            location="Page 1",
            ordinal=1,
            source_name="embedded.png",
        )
        visuals = (occurrence,) if occurrence is not None else ()
        return ParsedDocument(markdown=self._text, visuals=visuals, warnings=image_sink.warnings)


def _png_bytes(color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    from PIL import Image

    output = BytesIO()
    Image.new("RGB", (8, 8), color).save(output, format="PNG")
    return output.getvalue()


def _visual_paths(result: str) -> list[str]:
    paths: list[str] = []
    for line in result.splitlines():
        marker = line.find('{"path"')
        if marker >= 0:
            payload = json.loads(line[marker:])
            paths.append(payload["path"])
    return paths


def _session_artifact_path(session_dir: Path, reference: str) -> Path:
    resolved = resolve_document_image_artifact_handle(reference, session_dir)
    assert resolved is not None
    return Path(resolved)


def _session_markdown_artifact_path(session_dir: Path, reference: str) -> Path:
    resolved = resolve_document_markdown_artifact_handle(reference, session_dir)
    assert resolved is not None
    return Path(resolved)


def _saved_markdown_path(result: str, session_dir: Path) -> Path:
    handle = result.split("Saved Markdown handle: ", 1)[1].splitlines()[0]
    return _session_markdown_artifact_path(session_dir, handle)


def _saved_markdown_absolute_path(result: str) -> Path:
    path = result.split("Saved Markdown path: ", 1)[1].splitlines()[0]
    return Path(path)

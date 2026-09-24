# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Where workflow files live and how their bytes are read once.

Three sources, highest precedence first: the project (``<cwd>/.chrys/workflows``),
the user's global directory (``<config_dir>/workflows``), and the builtin
templates shipped inside chrys. A file's stem is its workflow id; a project
file shadows a global one of the same id, which shadows a builtin. User files
are read owner-verified (no symlinks, no foreign owners) and those bytes are
the ones every later step hashes, confirms and feeds to the worker.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from chrys.foundation.models.workflow_session import WorkflowIdentity
from chrys.foundation.platform.files import secure_open_owner_verified_binary

logger = logging.getLogger(__name__)

SOURCE_KIND_BUILTIN: Final = "builtin"
SOURCE_KIND_GLOBAL: Final = "global"
SOURCE_KIND_PROJECT: Final = "project"
WORKFLOWS_DIR_NAME: Final = "workflows"
PROJECT_CONFIG_DIR_NAME: Final = ".chrys"
BUILTIN_DIR: Final = Path(__file__).resolve().parent / "builtins"
MAX_SOURCE_BYTES: Final = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class WorkflowSource:
    """One workflow file as read: identity, precedence class, and the exact bytes."""

    workflow_id: str
    source_kind: str
    canonical_path: str
    source: bytes

    @property
    def identity(self) -> WorkflowIdentity:
        return WorkflowIdentity(self.workflow_id, self.canonical_path, self.source_kind)

    @property
    def entry_sha256(self) -> str:
        return hashlib.sha256(self.source).hexdigest()


@dataclass(frozen=True, slots=True)
class SkippedSource:
    path: str
    reason: str
    source_kind: str = ""


@dataclass(frozen=True, slots=True)
class ShadowedSource:
    source: WorkflowSource
    shadowed_by: WorkflowSource


@dataclass(frozen=True, slots=True)
class Discovery:
    sources: tuple[WorkflowSource, ...]
    skipped: tuple[SkippedSource, ...]
    shadowed: tuple[ShadowedSource, ...] = ()

    def find(self, workflow_id: str) -> WorkflowSource | None:
        for source in self.sources:
            if source.workflow_id == workflow_id:
                return source
        return None


def global_workflows_dir(config_dir: Path) -> Path:
    return config_dir / WORKFLOWS_DIR_NAME


def project_workflows_dir(project_cwd: Path) -> Path:
    return project_cwd / PROJECT_CONFIG_DIR_NAME / WORKFLOWS_DIR_NAME


def discover_workflows(*, config_dir: Path, project_cwd: Path | None) -> Discovery:
    """Every runnable workflow by id (higher-precedence sources shadow lower), sorted by id."""
    by_id: dict[str, WorkflowSource] = {}
    all_sources: list[WorkflowSource] = []
    skipped: list[SkippedSource] = []
    layers: list[tuple[str, Path]] = [
        (SOURCE_KIND_BUILTIN, BUILTIN_DIR),
        (SOURCE_KIND_GLOBAL, global_workflows_dir(config_dir)),
    ]
    if project_cwd is not None:
        layers.append((SOURCE_KIND_PROJECT, project_workflows_dir(project_cwd)))
    for kind, directory in layers:
        for path in _scan(directory, kind, skipped):
            try:
                source = read_source(path, kind)
            except OSError as exc:
                skipped.append(SkippedSource(str(path.parent.resolve() / path.name), str(exc), kind))
                continue
            by_id[source.workflow_id] = source  # later layers take precedence
            all_sources.append(source)
    shadowed = tuple(
        ShadowedSource(source, by_id[source.workflow_id])
        for source in sorted(all_sources, key=lambda item: (item.workflow_id, item.source_kind))
        if source is not by_id[source.workflow_id]
    )
    return Discovery(tuple(by_id[key] for key in sorted(by_id)), tuple(skipped), shadowed)


def read_source(path: Path, kind: str) -> WorkflowSource:
    """Read one workflow file; user files owner-verified, builtins as installed (a system install is root-owned)."""
    canonical = path.parent.resolve() / path.name
    if kind == SOURCE_KIND_BUILTIN:
        payload = canonical.read_bytes()
    else:
        with secure_open_owner_verified_binary(canonical) as handle:
            payload = handle.read(MAX_SOURCE_BYTES + 1)
    if len(payload) > MAX_SOURCE_BYTES:
        raise OSError(f"{canonical} is larger than {MAX_SOURCE_BYTES} bytes")
    return WorkflowSource(canonical.stem, kind, str(canonical), payload)


def builtin_manifest_path(workflow_id: str) -> Path:
    """The pre-generated manifest shipped next to a builtin template."""
    return BUILTIN_DIR / f"{workflow_id}.manifest.json"


def read_builtin_manifest(workflow_id: str) -> dict[str, Any] | None:
    """The pre-generated manifest of builtin *workflow_id*; ``None`` when there is none or it is unreadable."""
    try:
        payload = json.loads(builtin_manifest_path(workflow_id).read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _scan(directory: Path, kind: str, skipped: list[SkippedSource]) -> list[Path]:
    """The ``*.py`` files directly under *directory*, skipping dot and underscore names; sorted."""
    found: list[Path] = []
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                name = entry.name
                if not name.endswith(".py") or name.startswith((".", "_")):
                    continue
                if not entry.is_file(follow_symlinks=False) and not entry.is_symlink():
                    continue
                found.append(Path(entry.path))
    except FileNotFoundError:
        return []
    except OSError as exc:
        skipped.append(SkippedSource(str(directory), str(exc), kind))
        logger.warning("workflow directory %s could not be listed", directory, exc_info=True)
        return []
    found.sort()
    return found

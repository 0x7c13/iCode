# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Where an approved call was confirmed to act, re-checked by the tool before it acts.

Approval resolves the physical destination of a write, or the physical working
directory of a shell command, while the request is shown or matched. The tool
re-resolves it at its own boundary and refuses to act when it changed, so an
approval (or a remembered grant) never covers a different place.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from chrys.foundation.platform.paths import resolve_workspace_path

CHANGED_AFTER_APPROVAL = "{target} changed after approval; request approval again."


@dataclass(frozen=True)
class FileWriteTarget:
    """The replaced entry and, for a symlink, the source read by write/edit previews."""

    path: str
    link_target: str | None = None
    referent: str | None = None


@dataclass(frozen=True)
class ApprovedTargets:
    """What one approval covers; None leaves that kind of target unchecked."""

    files: tuple[FileWriteTarget, ...] | None = None
    cwd: str | None = None


_approved: ContextVar[ApprovedTargets | None] = ContextVar("approved_targets", default=None)


def physical_dir(path: str) -> str:
    """The case-normalized real path a directory currently resolves to."""
    return os.path.normcase(os.path.realpath(path))


def file_write_target(path: str, *, base_cwd: str | None = None) -> FileWriteTarget:
    """Resolve the entry a write replaces, independently of whether it is reusable."""
    lexical = resolve_workspace_path(path, base_cwd=base_cwd)
    # Resolve only the parent: atomic replacement replaces the final entry,
    # not its referent. Pin this entry even when it is a final-file symlink.
    parent, name = os.path.split(lexical)
    entry = os.path.join(os.path.realpath(parent, strict=os.path.ALLOW_MISSING), name)
    if os.path.islink(entry):
        # Non-strict resolution preserves the existing ability to replace a
        # dangling or looping final link; such links still cannot mint grants.
        return FileWriteTarget(entry, os.readlink(entry), os.path.realpath(entry))
    return FileWriteTarget(entry)


@contextmanager
def approved_targets(targets: ApprovedTargets | None) -> Iterator[None]:
    """Scope the confirmed targets to one tool call, including its worker threads."""
    token = _approved.set(targets)
    try:
        yield
    finally:
        _approved.reset(token)


def approved_write_path(path: str, *, base_cwd: str | None = None) -> str:
    """Check a write against its approval and pin the path the whole operation uses."""
    approved = _approved.get()
    if approved is None or approved.files is None:
        return path
    target = file_write_target(path, base_cwd=base_cwd)
    if target not in approved.files:
        raise ValueError(CHANGED_AFTER_APPROVAL.format(target="File destination"))
    return target.path


def approved_cwd() -> str | None:
    """The physical directory an approved command must run in, if one was pinned."""
    approved = _approved.get()
    return approved.cwd if approved is not None else None

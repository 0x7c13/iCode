# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Workspace data model — session-scoped working directory configuration.

A workspace groups the primary cwd, additional working directories,
and reference files for a single session.  Changes trigger a SessionEnvironment
rebuild but preserve conversation history.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class WorkingDir:
    """A single working directory within a workspace."""

    path: str  # Absolute path
    label: str = ""  # Optional user label (e.g. "frontend", "api")
    is_primary: bool = False  # Exactly one must be True per workspace

    @property
    def exists(self) -> bool:
        """Check if the directory still exists on disk."""
        return os.path.isdir(self.path)


@dataclass
class Workspace:
    """Workspace — working directory configuration for a session.

    Groups the primary cwd, additional working directories, and reference
    files.  Can be updated mid-session (e.g. changing cwd, adding reference
    dirs/files).  Changes trigger a SessionEnvironment rebuild but preserve
    conversation history.
    """

    primary_cwd: str
    working_dirs: list[WorkingDir] = field(default_factory=list)
    reference_files: list[str] = field(default_factory=list)

    @classmethod
    def from_cwd(cls, cwd: str | None = None) -> Workspace:
        """Create from a working directory (defaults to os.getcwd())."""
        raw = cwd or os.getcwd()
        return cls(primary_cwd=os.path.abspath(os.path.expanduser(raw)))

    def missing_primary(self) -> str | None:
        """Return the primary cwd when it is no longer a directory, else ``None``.

        New work (turns, retries, restores, workflow runs) refuses to start
        while this returns a path; additional working dirs are optional and
        never block. The raw path is returned for display and filesystem use.
        """
        if self.primary_cwd and not os.path.isdir(self.primary_cwd):
            return self.primary_cwd
        return None

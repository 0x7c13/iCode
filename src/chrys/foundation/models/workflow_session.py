# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The workflow and owned workspace selected by one admitted session."""

from __future__ import annotations

from dataclasses import dataclass

from chrys.foundation.models.workspace import WorkingDir, Workspace


@dataclass(frozen=True, slots=True)
class WorkingDirSnapshot:
    path: str
    label: str = ""
    is_primary: bool = False


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshot:
    """A prepared target never retains the caller's mutable Workspace or directory lists."""

    primary_cwd: str
    working_dirs: tuple[WorkingDirSnapshot, ...] = ()
    reference_files: tuple[str, ...] = ()

    @classmethod
    def capture(cls, workspace: Workspace) -> WorkspaceSnapshot:
        return cls(
            workspace.primary_cwd,
            tuple(WorkingDirSnapshot(d.path, d.label, d.is_primary) for d in workspace.working_dirs),
            tuple(workspace.reference_files),
        )

    def materialize(self) -> Workspace:
        return Workspace(
            self.primary_cwd,
            [WorkingDir(d.path, d.label, d.is_primary) for d in self.working_dirs],
            list(self.reference_files),
        )


@dataclass(frozen=True, slots=True)
class WorkflowModelSelection:
    """A model profile identity plus durable display text; never stores credentials."""

    profile_id: str
    name: str
    model_id: str


@dataclass(frozen=True, slots=True)
class WorkflowDraft:
    workflow_id: str
    workspace: WorkspaceSnapshot
    model: WorkflowModelSelection | None = None


@dataclass(frozen=True, slots=True)
class WorkflowIdentity:
    workflow_id: str
    canonical_path: str
    source_kind: str


@dataclass(frozen=True, slots=True)
class WorkflowPins:
    """The exact source, specification and environment reviewed before submission."""

    identity: WorkflowIdentity
    spec_digest: str
    environment_fingerprint: str


@dataclass(frozen=True, slots=True)
class WorkflowSessionSelection:
    """Replace the selection as a whole; workspace belongs to this session, never to Chat."""

    session_id: str
    identity: WorkflowIdentity
    workspace: WorkspaceSnapshot
    model: WorkflowModelSelection | None = None

    @property
    def workflow_id(self) -> str:
        return self.identity.workflow_id


type WorkflowTarget = WorkflowDraft | WorkflowSessionSelection

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Workflow browsing, preview, confirmation and file operations shared by session frontends.

The catalog owns previews observed in this process. Discovery and history only
read files; preview is the explicit operation that executes a workflow module.
Deletion does not touch run history. The caller must check the coordinator's
active_source before deleting a workflow that might be running.
"""

from __future__ import annotations

import asyncio
import stat
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from chrys.foundation.events.types import WorkflowPreviewProgress
from chrys.foundation.models.workflow_session import WorkflowIdentity
from chrys.orchestration.workflows.preview import (
    REJECT_NOT_CONFIRMED,
    REJECT_SPEC_CHANGED,
    WorkflowInspection,
    WorkflowPreview,
    WorkflowPreviewError,
    WorkflowTrustDeclined,
    materialize_runtime_sdk,
    preview_workflow,
)
from chrys.service.workflows import discovery as discovery_module
from chrys.service.workflows.discovery import (
    SOURCE_KIND_BUILTIN,
    SOURCE_KIND_PROJECT,
    Discovery,
    WorkflowSource,
    discover_workflows,
    global_workflows_dir,
    project_workflows_dir,
    read_builtin_manifest,
    read_source,
)
from chrys.service.workflows.ledger import ConfirmationLedger, ledger_path

if TYPE_CHECKING:
    from chrys.foundation.events.bus import EventBus
    from chrys.service.workflows.environment import PreparedEnvironment


class WorkflowNotFoundError(KeyError):
    """No discovered workflow has the requested id."""


class WorkflowCatalog:
    """Operations scoped to the session's config directory and current project."""

    def __init__(self, *, config_dir: Path, project_cwd: Path, bus: EventBus | None = None) -> None:
        self.config_dir = config_dir
        self.project_cwd = project_cwd
        self._bus = bus
        self._previews: dict[str, WorkflowPreview] = {}

    def discover(self) -> Discovery:
        return discover_workflows(config_dir=self.config_dir, project_cwd=self.project_cwd)

    def ledger(self) -> ConfirmationLedger:
        return ConfirmationLedger(ledger_path(self.config_dir))

    def title(self, source: WorkflowSource, *, ledger: ConfirmationLedger | None = None) -> str | None:
        """Previously observed metadata, without loading the workflow on a worker."""
        if source.source_kind == SOURCE_KIND_BUILTIN:
            manifest = read_builtin_manifest(source.workflow_id)
            title = manifest.get("title") if manifest is not None else None
            return title if isinstance(title, str) else None
        preview = self._previews.get(source.canonical_path)
        if preview is not None and preview.source.entry_sha256 == source.entry_sha256:
            return preview.title
        entry = (ledger if ledger is not None else self.ledger()).recorded(source.canonical_path, source.source_kind)
        return entry.title if entry is not None else None

    async def preview(
        self,
        workflow_id: str,
        *,
        timeout: float | None = None,
        request_id: str = "",
        expected_identity: WorkflowIdentity | None = None,
        trust: bool = False,
        authorize: Callable[[WorkflowInspection], Awaitable[bool]] | None = None,
    ) -> WorkflowPreview:
        """Require source trust before any interpreter runs, then load on a throwaway worker.

        ``trust`` is an explicit caller authorization (e.g. CLI ``--trust``).
        Interactive callers may instead supply ``authorize``. Human decision time
        is excluded from the preview timeout. Timed-out filesystem awaits release
        the caller; their threads may finish later. Cancellation drains any worker.
        """
        async with asyncio.timeout(timeout) as deadline:
            source = (await asyncio.to_thread(self.discover)).find(workflow_id)
            if source is None:
                raise WorkflowNotFoundError(f"Workflow not found: {workflow_id}")
            if expected_identity is not None and expected_identity != source.identity:
                raise WorkflowPreviewError(
                    REJECT_SPEC_CHANGED, "This session belongs to another workflow source. Start a new session."
                )
            approved = trust or source.source_kind == SOURCE_KIND_BUILTIN
            recorded = None

            async def authorize_source(environment: PreparedEnvironment | None = None) -> None:
                nonlocal approved
                if authorize is None:
                    raise WorkflowPreviewError(
                        REJECT_NOT_CONFIRMED, "Trust the workflow source and environment before previewing it."
                    )
                inspection = await asyncio.to_thread(WorkflowInspection.read, source)
                inspection = replace(inspection, prepared_environment=environment)
                loop = asyncio.get_running_loop()
                paused_at, expires = loop.time(), deadline.when()
                if expires is not None and paused_at >= expires:
                    raise TimeoutError
                deadline.reschedule(None)
                try:
                    accepted = await authorize(inspection)
                finally:
                    deadline.reschedule(None if expires is None else expires + loop.time() - paused_at)
                if not accepted:
                    raise WorkflowTrustDeclined
                if (await asyncio.to_thread(self.discover)).find(workflow_id) != source:
                    raise WorkflowPreviewError(REJECT_SPEC_CHANGED, "The workflow source changed during confirmation.")
                approved = True

            if not approved:
                ledger = await asyncio.to_thread(self.ledger)
                recorded = ledger.recorded(source.canonical_path, source.source_kind)
                if recorded is None or recorded.entry_digest != source.entry_sha256:
                    await authorize_source()

            async def report(
                stage: Literal["definition", "environment", "graph", "ready"],
                *,
                title: str = "",
                node_count: int = 0,
            ) -> None:
                if self._bus is not None and request_id:
                    await self._bus.publish(
                        WorkflowPreviewProgress(
                            request_id=request_id,
                            workflow_id=workflow_id,
                            stage=stage,
                            title=title,
                            node_count=node_count,
                        )
                    )

            await report("definition")
            await report("environment")
            sdk = await materialize_runtime_sdk(self.config_dir)

            async def environment_ready(environment: PreparedEnvironment) -> None:
                if (
                    not approved
                    and recorded is not None
                    and recorded.environment_fingerprint != environment.environment_fingerprint
                ):
                    await authorize_source(environment)
                await report("graph")

            preview = await preview_workflow(
                source, sdk=sdk, workspace=self.project_cwd, on_environment_ready=environment_ready
            )
            await report("ready", title=preview.title, node_count=len(preview.manifest["nodes"]))
        self._previews[source.canonical_path] = preview
        return preview

    def confirm(self, preview: WorkflowPreview) -> None:
        self.ledger().confirm(preview.ledger_entry())

    def candidate_paths(self, source: WorkflowSource) -> tuple[Path, ...]:
        """The selected entry followed by every source that could shadow it."""
        paths = [Path(source.canonical_path)]
        if source.source_kind == SOURCE_KIND_BUILTIN:
            paths.append(global_workflows_dir(self.config_dir) / f"{source.workflow_id}.py")
        if source.source_kind != SOURCE_KIND_PROJECT:
            paths.append(project_workflows_dir(self.project_cwd) / f"{source.workflow_id}.py")
        return tuple(paths)

    def is_current(self, preview: WorkflowPreview) -> bool:
        """Check the entry bytes and precedence; a replacement or newly shadowed preview is stale."""
        source = preview.source
        try:
            if read_source(Path(source.canonical_path), source.source_kind) != source:
                return False
        except OSError:
            return False
        for path in self.candidate_paths(source)[1:]:
            try:
                read_source(path, SOURCE_KIND_PROJECT)
            except OSError:
                continue
            return False
        return True

    def delete(self, canonical_path: str) -> None:
        """Unlink a direct user workflow file (including a symlink itself), then forget its confirmation."""
        path = Path(canonical_path)
        canonical = path.parent.resolve() / path.name
        directories = {
            global_workflows_dir(self.config_dir).resolve(),
            project_workflows_dir(self.project_cwd).resolve(),
        }
        if (
            not path.is_absolute()
            or path != canonical
            or canonical.parent not in directories
            or canonical.is_relative_to(discovery_module.BUILTIN_DIR.resolve())
            or canonical.suffix != ".py"
            or canonical.name.startswith((".", "_"))
            or not canonical.stem
        ):
            raise ValueError("Only files directly in a global or project workflow directory can be deleted.")
        mode = canonical.lstat().st_mode
        if not stat.S_ISREG(mode) and not stat.S_ISLNK(mode):
            raise ValueError("Only a regular workflow file or symlink can be deleted.")
        canonical.unlink()
        self.ledger().remove(str(canonical))
        self._previews.pop(str(canonical), None)

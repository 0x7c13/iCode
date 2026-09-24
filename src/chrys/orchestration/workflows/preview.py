# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Loading a workflow file in a worker before a run: environment, manifest and digests.

A preview is the same two steps the run itself starts with, environment
preparation and a worker load, done on a throwaway worker. What comes back
is the data-only manifest plus the digests the confirmation ledger pins, so
the CLI can show and confirm exactly what a run would execute.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from chrys.foundation.events.types import WorkflowRunRequest
from chrys.foundation.models.workflow_session import WorkflowPins, WorkflowTarget
from chrys.orchestration.invoker.resources import finish_close
from chrys.orchestration.workflows.worker_client import (
    AskHandler,
    CapturedOutput,
    EmitHandler,
    LoadResult,
    WorkerLostError,
    WorkerRpcError,
    WorkerStartError,
    WorkflowWorkerClient,
)
from chrys.service.workflows.admission import spec_digest
from chrys.service.workflows.discovery import WORKFLOWS_DIR_NAME, WorkflowSource
from chrys.service.workflows.environment import (
    EnvironmentPlan,
    PreparedEnvironment,
    WorkflowEnvironmentError,
    WorkflowEnvironmentManager,
    parse_environment_request,
    plan_environment,
)
from chrys.service.workflows.graph import ManifestWarning, manifest_warnings
from chrys.service.workflows.ledger import LedgerEntry
from chrys.service.workflows.sdk_artifact import SdkArtifact, materialize_sdk_artifact

PREVIEW_ENVIRONMENT_ERROR: Final = "environment_error"
PREVIEW_WORKER_START_FAILED: Final = "worker_start_failed"
PREVIEW_LOAD_FAILED: Final = "load_failed"
PREVIEW_WORKER_LOST: Final = "worker_lost"
REJECT_SPEC_CHANGED: Final = "spec_changed"
REJECT_NOT_CONFIRMED: Final = "not_confirmed"

SDK_ARTIFACT_DIR_NAME: Final = "sdk"


class WorkflowPreviewError(Exception):
    """The workflow cannot be loaded; ``code`` is the deterministic rejection reason."""

    def __init__(self, code: str, message: str, *, traceback: str = "", stdout: CapturedOutput | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.traceback = traceback
        self.stdout = stdout or CapturedOutput("", False)


@dataclass(frozen=True, slots=True)
class WorkflowInspection:
    """Source bytes and a lexical environment plan; creating this never runs user code."""

    source: WorkflowSource
    environment: EnvironmentPlan
    prepared_environment: PreparedEnvironment | None = None

    @classmethod
    def read(cls, source: WorkflowSource) -> WorkflowInspection:
        try:
            request = parse_environment_request(source.source)
            return cls(source, plan_environment(request, entry_path=Path(source.canonical_path)))
        except WorkflowEnvironmentError as exc:
            raise WorkflowPreviewError(PREVIEW_ENVIRONMENT_ERROR, str(exc)) from exc


class WorkflowTrustDeclined(Exception):
    """The user declined execution of the inspected source; no worker was launched."""


@dataclass(frozen=True, slots=True)
class LoadedWorkflow:
    """A worker that has executed the workflow file; the caller owns ``client``."""

    client: WorkflowWorkerClient
    load: LoadResult
    manifest: dict[str, Any]
    spec_digest: str


@dataclass(frozen=True, slots=True)
class WorkflowPreview:
    """What a run of ``source`` would execute, as reported by a throwaway worker."""

    source: WorkflowSource
    environment: PreparedEnvironment
    load: LoadResult
    manifest: dict[str, Any]
    spec_digest: str

    @property
    def title(self) -> str:
        title = self.manifest.get("title")
        return title if isinstance(title, str) else ""

    @property
    def warnings(self) -> tuple[ManifestWarning, ...]:
        return manifest_warnings(self.manifest)

    def ledger_entry(self) -> LedgerEntry:
        return ledger_entry_for(self.source, self.load, self.spec_digest, self.environment, title=self.title)


@dataclass(frozen=True, slots=True)
class PreparedWorkflow:
    """A transient execution target and the preview whose pins it submits."""

    target: WorkflowTarget
    preview: WorkflowPreview

    def run_request(self, *, input_text: str, request_id: str, timeout: float = 0.0) -> WorkflowRunRequest:
        preview = self.preview
        return WorkflowRunRequest(
            target=self.target,
            pins=WorkflowPins(
                preview.source.identity, preview.spec_digest, preview.environment.environment_fingerprint
            ),
            input_text=input_text,
            request_id=request_id,
            timeout=timeout,
        )


def ledger_entry_for(
    source: WorkflowSource, load: LoadResult, digest: str, environment: PreparedEnvironment, *, title: str
) -> LedgerEntry:
    return LedgerEntry(
        title=title,
        canonical_path=source.canonical_path,
        source_kind=source.source_kind,
        workflow_id=source.workflow_id,
        entry_digest=load.entry_digest,
        manifest_digest=load.manifest_digest,
        schema_version=load.manifest["schema_version"],
        spec_digest=digest,
        environment_fingerprint=environment.environment_fingerprint,
    )


def sdk_artifact_dir(config_dir: Path) -> Path:
    """Where the injected SDK lives; discovery only scans top-level files, so the directory never collides."""
    return config_dir / WORKFLOWS_DIR_NAME / SDK_ARTIFACT_DIR_NAME


async def materialize_runtime_sdk(config_dir: Path) -> SdkArtifact:
    return await asyncio.to_thread(materialize_sdk_artifact, sdk_artifact_dir(config_dir))


async def prepare_workflow_environment(source: WorkflowSource, *, sdk: SdkArtifact) -> PreparedEnvironment:
    """Parse the file's environment request, choose the interpreter, and probe it."""
    try:
        request = parse_environment_request(source.source)
        plan = plan_environment(request, entry_path=Path(source.canonical_path))
        return await WorkflowEnvironmentManager(sdk_digest=sdk.digest).prepare(plan)
    except WorkflowEnvironmentError as exc:
        raise WorkflowPreviewError(PREVIEW_ENVIRONMENT_ERROR, str(exc)) from exc


async def load_workflow(
    source: WorkflowSource,
    *,
    environment: PreparedEnvironment,
    sdk: SdkArtifact,
    workspace: Path,
    ask_handler: AskHandler | None = None,
    emit_handler: EmitHandler | None = None,
) -> LoadedWorkflow:
    """Start a fresh worker and execute the file in it; on any failure the worker is closed before raising."""
    try:
        client = await WorkflowWorkerClient.launch(
            environment=environment, sdk=sdk, workspace=workspace, ask_handler=ask_handler, emit_handler=emit_handler
        )
    except WorkerStartError as exc:
        raise WorkflowPreviewError(PREVIEW_WORKER_START_FAILED, str(exc)) from exc
    try:
        load = await client.load(source.source, filename=source.canonical_path, workspace=workspace)
        manifest = load.manifest
        digest = spec_digest(load.entry_digest, load.manifest_digest, manifest["schema_version"])
    except WorkerRpcError as exc:
        await _close_worker(client)
        raise WorkflowPreviewError(
            PREVIEW_LOAD_FAILED,
            f"{exc.code}: {exc.message}",
            traceback=exc.traceback,
            stdout=exc.stdout,
        ) from exc
    except WorkerLostError as exc:
        await _close_worker(client)
        raise WorkflowPreviewError(PREVIEW_WORKER_LOST, str(exc)) from exc
    except BaseException:
        await _close_worker(client)
        raise
    return LoadedWorkflow(client=client, load=load, manifest=manifest, spec_digest=digest)


async def preview_workflow(
    source: WorkflowSource,
    *,
    sdk: SdkArtifact,
    workspace: Path,
    on_environment_ready: Callable[[PreparedEnvironment], Awaitable[None]] | None = None,
) -> WorkflowPreview:
    """Prepare the environment and load the file on a worker that is closed before returning."""
    environment = await prepare_workflow_environment(source, sdk=sdk)
    if on_environment_ready is not None:
        await on_environment_ready(environment)
    loaded = await load_workflow(source, environment=environment, sdk=sdk, workspace=workspace)
    await _close_worker(loaded.client)
    return WorkflowPreview(
        source=source,
        environment=environment,
        load=loaded.load,
        manifest=loaded.manifest,
        spec_digest=loaded.spec_digest,
    )


async def _close_worker(client: WorkflowWorkerClient) -> None:
    try:
        await client.close()
    except asyncio.CancelledError:
        # close owns a shielded task; drain it even if the caller is cancelled repeatedly.
        await finish_close(asyncio.create_task(client.close()))
        raise

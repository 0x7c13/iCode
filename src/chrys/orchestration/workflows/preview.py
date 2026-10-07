# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Loading a workflow file in a worker before a run: environment, manifest and digests.

A preview is the same two steps the run itself starts with, environment
preparation and a worker load, done on a throwaway worker. What comes back
is the data-only manifest plus the digests the confirmation ledger pins, so
the CLI can show and confirm exactly what a run would execute.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from chrys.foundation.events.types import WorkflowRunRequest
from chrys.foundation.models.workflow_session import WorkflowPins, WorkflowTarget
from chrys.foundation.platform.files import surrogate_safe_text
from chrys.foundation.util.once_close import finish_close
from chrys.orchestration.workflows.worker_client import (
    LOAD_TIMED_OUT,
    AskHandler,
    CapturedOutput,
    EmitHandler,
    LoadDiagnostic,
    LoadResult,
    WorkerLostError,
    WorkerRpcError,
    WorkerStartError,
    WorkflowWorkerClient,
    load_diagnostics,
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
PREVIEW_WORKING_DIR_MISSING: Final = "working_dir_missing"
REJECT_SPEC_CHANGED: Final = "spec_changed"
REJECT_NOT_CONFIRMED: Final = "not_confirmed"

SDK_ARTIFACT_DIR_NAME: Final = "sdk"
BYTECODE_CACHE_DIR_NAME: Final = ".pycache"


class WorkflowPreviewError(Exception):
    """The workflow cannot be loaded; ``code`` is the deterministic rejection reason.

    A ``diagnose`` load also says where it failed (``diagnostics``, whether
    any were left out) and whether it ran out of time (``timed_out``).
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        traceback: str = "",
        stdout: CapturedOutput | None = None,
        diagnostics: tuple[LoadDiagnostic, ...] = (),
        diagnostics_truncated: bool = False,
        timed_out: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.traceback = traceback
        self.stdout = stdout or CapturedOutput("", False)
        self.diagnostics = diagnostics
        self.diagnostics_truncated = diagnostics_truncated
        self.timed_out = timed_out


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
            raise WorkflowPreviewError(PREVIEW_ENVIRONMENT_ERROR, _located(exc)) from exc


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
        entry_digest=source.source_digest,
        manifest_digest=load.manifest_digest,
        schema_version=load.manifest["schema_version"],
        spec_digest=digest,
        environment_fingerprint=environment.environment_fingerprint,
    )


def sdk_artifact_dir(config_dir: Path) -> Path:
    """Where the injected SDK lives; discovery reserves the name ``sdk`` in the global directory, so it never collides."""
    return config_dir / WORKFLOWS_DIR_NAME / SDK_ARTIFACT_DIR_NAME


def worker_bytecode_cache_dir(config_dir: Path) -> Path:
    """The private bytecode cache of every workflow worker; discovery ignores hidden names, so it never collides."""
    return config_dir / WORKFLOWS_DIR_NAME / BYTECODE_CACHE_DIR_NAME


async def materialize_runtime_sdk(config_dir: Path) -> SdkArtifact:
    return await asyncio.to_thread(materialize_sdk_artifact, sdk_artifact_dir(config_dir))


async def prepare_workflow_environment(source: WorkflowSource, *, sdk: SdkArtifact) -> PreparedEnvironment:
    """Parse the file's environment request, choose the interpreter, and probe it."""
    try:
        request = parse_environment_request(source.source)
        plan = plan_environment(request, entry_path=Path(source.canonical_path))
        return await WorkflowEnvironmentManager(sdk_digest=sdk.digest).prepare(plan)
    except WorkflowEnvironmentError as exc:
        raise WorkflowPreviewError(PREVIEW_ENVIRONMENT_ERROR, _located(exc)) from exc


async def load_workflow(
    source: WorkflowSource,
    *,
    environment: PreparedEnvironment,
    sdk: SdkArtifact,
    workspace: Path,
    bytecode_cache: Path,
    ask_handler: AskHandler | None = None,
    emit_handler: EmitHandler | None = None,
    diagnose: bool = False,
) -> LoadedWorkflow:
    """Start a fresh worker and execute the file in it; on any failure the worker is closed before raising.

    *diagnose* (validation) also compiles a folder's other Python files first
    and places a failure in the error's ``diagnostics``.
    """
    _require_workspace(workspace)
    try:
        client = await WorkflowWorkerClient.launch(
            environment=environment,
            sdk=sdk,
            workspace=workspace,
            bytecode_cache=bytecode_cache,
            ask_handler=ask_handler,
            emit_handler=emit_handler,
        )
    except WorkerStartError as exc:
        raise WorkflowPreviewError(PREVIEW_WORKER_START_FAILED, str(exc)) from exc
    package_dir = source.package.directory if source.package is not None else None
    precompile = source.package.python_files(Path(source.canonical_path).name) if source.package is not None else ()
    try:
        load = await client.load(
            source.source,
            filename=source.canonical_path,
            workspace=workspace,
            package_dir=package_dir,
            diagnose=diagnose,
            precompile=precompile if diagnose else (),
        )
        manifest = load.manifest
        digest = spec_digest(source.source_digest, load.manifest_digest, manifest["schema_version"])
    except WorkerRpcError as exc:
        await _close_worker(client)
        diagnostics, truncated = load_diagnostics(exc.data)
        raise WorkflowPreviewError(
            PREVIEW_LOAD_FAILED,
            f"{exc.code}: {exc.message}",
            traceback=exc.traceback,
            stdout=exc.stdout,
            diagnostics=diagnostics,
            diagnostics_truncated=truncated,
            timed_out=exc.data.get("reason") == LOAD_TIMED_OUT,
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
    bytecode_cache: Path,
    on_environment_ready: Callable[[PreparedEnvironment], Awaitable[None]] | None = None,
    diagnose: bool = False,
) -> WorkflowPreview:
    """Prepare the environment and load the file on a worker that is closed before returning."""
    _require_workspace(workspace)
    environment = await prepare_workflow_environment(source, sdk=sdk)
    if on_environment_ready is not None:
        await on_environment_ready(environment)
    loaded = await load_workflow(
        source,
        environment=environment,
        sdk=sdk,
        workspace=workspace,
        bytecode_cache=bytecode_cache,
        diagnose=diagnose,
    )
    await _close_worker(loaded.client)
    return WorkflowPreview(
        source=source,
        environment=environment,
        load=loaded.load,
        manifest=loaded.manifest,
        spec_digest=loaded.spec_digest,
    )


def _require_workspace(workspace: Path) -> None:
    """Refuse before probing or spawning anything when the worker's cwd is gone."""
    if not workspace.is_dir():
        raise WorkflowPreviewError(
            PREVIEW_WORKING_DIR_MISSING,
            f"The working directory no longer exists: {surrogate_safe_text(os.fspath(workspace))}",
        )


def _located(exc: WorkflowEnvironmentError) -> str:
    """The message with its place in the file, for surfaces that show only the message."""
    if exc.line is None:
        return str(exc)
    place = f"line {exc.line}" if exc.column is None else f"line {exc.line}, column {exc.column}"
    return f"{exc} ({place})"


async def _close_worker(client: WorkflowWorkerClient) -> None:
    try:
        await client.close()
    except asyncio.CancelledError:
        # close owns a shielded task; drain it even if the caller is cancelled repeatedly.
        await finish_close(asyncio.create_task(client.close()))
        raise

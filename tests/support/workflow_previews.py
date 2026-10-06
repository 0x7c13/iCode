# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Reuse real workflow previews across UI tests, so each one does not start an interpreter probe and a worker.

A preview probes the interpreter and loads the file on a throwaway worker.
``tests/orchestration/workflows`` (``test_catalog.py``, ``test_preview_trust.py``,
``test_worker_*.py``) and ``tests/service/workflows`` cover that pipeline. UI
tests only need its result, so the first preview of a file really runs and the
later ones reuse its result:

- The key is the exact source (id, kind, canonical path, bytes and, for a
  folder, its digest), the SDK digest and the workspace; where the worker
  caches bytecode doesn't change what a preview finds. A builtin's key leaves out the workspace: builtin
  manifests are pre-generated without one (``tests/support/workflow_builtins.py``).
- Only default-interpreter previews are kept. A bring-your-own interpreter can
  change on disk under the same path.
- A reused preview still calls ``on_environment_ready``, so trust checks and
  progress reports run as they would.
- A test that replaces any step of the pipeline (to stall, fail or count it)
  gets the real preview, and the result is not kept.
"""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path

from chrys.orchestration.workflows import preview as preview_module
from chrys.orchestration.workflows.preview import WorkflowPreview
from chrys.orchestration.workflows.worker_client import WorkflowWorkerClient
from chrys.service.workflows import environment as environment_module
from chrys.service.workflows.discovery import SOURCE_KIND_BUILTIN, WorkflowSource
from chrys.service.workflows.environment import PreparedEnvironment, WorkflowEnvironmentManager
from chrys.service.workflows.sdk_artifact import SdkArtifact

_REAL_PREVIEW = preview_module.preview_workflow
_KEPT: dict[tuple[WorkflowSource, str, str], WorkflowPreview] = {}


def _pipeline() -> tuple[object, ...]:
    """Every step the real preview runs, as a test would replace it."""
    return (
        preview_module.prepare_workflow_environment,
        preview_module.load_workflow,
        preview_module._close_worker,
        preview_module.parse_environment_request,
        preview_module.plan_environment,
        preview_module.WorkflowWorkerClient,
        preview_module.WorkflowEnvironmentManager,
        WorkflowWorkerClient.__dict__["launch"],
        WorkflowWorkerClient.__dict__["load"],
        WorkflowWorkerClient.__dict__["close"],
        WorkflowEnvironmentManager.__dict__["prepare"],
        environment_module.probe_interpreter,
    )


_REAL_PIPELINE = _pipeline()


def _fresh(preview: WorkflowPreview, source: WorkflowSource) -> WorkflowPreview:
    """A copy the caller may keep and change without touching the kept preview."""
    manifest = copy.deepcopy(preview.manifest)
    return replace(preview, source=source, load=replace(preview.load, manifest=manifest), manifest=manifest)


async def reused_preview_workflow(
    source: WorkflowSource,
    *,
    sdk: SdkArtifact,
    workspace: Path,
    bytecode_cache: Path,
    on_environment_ready: Callable[[PreparedEnvironment], Awaitable[None]] | None = None,
) -> WorkflowPreview:
    """``preview_workflow`` that runs once per source, SDK and workspace in this process."""

    async def run_real() -> WorkflowPreview:
        return await _REAL_PREVIEW(
            source,
            sdk=sdk,
            workspace=workspace,
            bytecode_cache=bytecode_cache,
            on_environment_ready=on_environment_ready,
        )

    if any(step is not real for step, real in zip(_pipeline(), _REAL_PIPELINE, strict=True)):
        return await run_real()
    key = (source, sdk.digest, "" if source.source_kind == SOURCE_KIND_BUILTIN else str(workspace))
    kept = _KEPT.get(key)
    if kept is None:
        preview = await run_real()
        if preview.environment.mode == "default":
            _KEPT[key] = _fresh(preview, source)
        return preview
    if on_environment_ready is not None:
        await on_environment_ready(kept.environment)
    return _fresh(kept, source)

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A deleted working directory stops workflow preview, load and admission before any worker starts."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import create_autospec

import pytest

import chrys.orchestration.workflows.coordinator as coordinator_module
from chrys.foundation.platform import get_platform
from chrys.foundation.platform.files import atomic_write_owner_only_bytes
from chrys.orchestration.session_host import WorkflowRunRejectedError
from chrys.orchestration.workflows.coordinator import REJECT_WORKING_DIR_MISSING
from chrys.orchestration.workflows.preview import (
    PREVIEW_WORKING_DIR_MISSING,
    WorkflowPreviewError,
    load_workflow,
    materialize_runtime_sdk,
    worker_bytecode_cache_dir,
)
from chrys.orchestration.workflows.worker_client import WorkflowWorkerClient
from chrys.service.llm.mock import MockChatClient
from chrys.service.workflows.discovery import global_workflows_dir
from chrys.service.workflows.environment import WorkflowEnvironmentManager
from tests.orchestration.workflows._hosting import make_host, patch_runtime
from tests.support.workflow_workers import python_workflow

SOURCE = python_workflow("def check(text):\n    return text\n", "check")


def _write_global_workflow(workflow_id: str) -> None:
    """A user workflow outside the project, so deleting the project leaves the definition discoverable."""
    directory = global_workflows_dir(get_platform().config_dir)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_owner_only_bytes(directory / f"{workflow_id}.py", SOURCE)


def _forbid_worker_start(monkeypatch: pytest.MonkeyPatch) -> tuple[object, object]:
    prepare = create_autospec(
        WorkflowEnvironmentManager.prepare, side_effect=AssertionError("must refuse before probing")
    )
    launch = create_autospec(WorkflowWorkerClient.launch, side_effect=AssertionError("must refuse before launch"))
    monkeypatch.setattr(WorkflowEnvironmentManager, "prepare", prepare)
    monkeypatch.setattr(WorkflowWorkerClient, "launch", launch)
    return prepare, launch


async def test_run_admission_refuses_a_deleted_working_directory_until_it_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_runtime(monkeypatch, [MockChatClient(responses=[])])
    _write_global_workflow("review")
    work = tmp_path / "work"
    work.mkdir()
    host = make_host(tmp_path, project=work)
    try:
        await host.start()
        prepared = await host.preview_workflow(host.workflow_target("review", new_session=True), trust=True)
        host.confirm_workflow(prepared)
        work.rmdir()
        with monkeypatch.context() as patch:
            forbidden = create_autospec(
                coordinator_module.prepare_workflow_environment,
                side_effect=AssertionError("must reject before preparing environment"),
            )
            patch.setattr(coordinator_module, "prepare_workflow_environment", forbidden)
            with pytest.raises(WorkflowRunRejectedError) as rejected:
                await host.run_workflow_until_final(prepared)
            forbidden.assert_not_called()
        assert rejected.value.event.error == REJECT_WORKING_DIR_MISSING
        assert str(work) in rejected.value.event.message

        work.mkdir()
        result = await host.run_workflow_until_final(prepared)
        assert result.outcome.value == "completed"
    finally:
        await host.shutdown()


async def test_preview_refuses_a_deleted_working_directory_before_probing_or_launching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_runtime(monkeypatch, [MockChatClient(responses=[])])
    _write_global_workflow("review")
    work = tmp_path / "work"
    work.mkdir()
    host = make_host(tmp_path, project=work)
    try:
        target = host.workflow_target("review", new_session=True)
        work.rmdir()
        prepare, launch = _forbid_worker_start(monkeypatch)
        with pytest.raises(WorkflowPreviewError) as raised:
            await host.preview_workflow(target, trust=True)
        assert raised.value.code == PREVIEW_WORKING_DIR_MISSING
        assert str(work) in raised.value.message
        prepare.assert_not_called()
        launch.assert_not_called()
    finally:
        await host.shutdown()


async def test_load_refuses_a_deleted_working_directory_before_launching_a_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_runtime(monkeypatch, [MockChatClient(responses=[])])
    _write_global_workflow("review")
    work = tmp_path / "work"
    work.mkdir()
    host = make_host(tmp_path, project=work)
    try:
        prepared = await host.preview_workflow(host.workflow_target("review", new_session=True), trust=True)
        config_dir = get_platform().config_dir
        sdk = await materialize_runtime_sdk(config_dir)
        work.rmdir()
        _, launch = _forbid_worker_start(monkeypatch)
        with pytest.raises(WorkflowPreviewError) as raised:
            await load_workflow(
                prepared.preview.source,
                environment=prepared.preview.environment,
                sdk=sdk,
                workspace=work,
                bytecode_cache=worker_bytecode_cache_dir(config_dir),
            )
        assert raised.value.code == PREVIEW_WORKING_DIR_MISSING
        assert str(work) in raised.value.message
        launch.assert_not_called()
    finally:
        await host.shutdown()

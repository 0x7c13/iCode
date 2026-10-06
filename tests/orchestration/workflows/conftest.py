# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Worker fixtures: the SDK artifact, a prepared environment per interpreter, and a launched client per test."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from chrys.orchestration.workflows.worker_client import WorkflowWorkerClient
from chrys.service.workflows.environment import (
    DefaultPlan,
    PreparedEnvironment,
    WorkflowEnvironmentManager,
    parse_environment_request,
)
from chrys.service.workflows.sdk_artifact import SdkArtifact, materialize_sdk_artifact
from tests.support.workflow_workers import INTERPRETER_IDS, resolve_interpreter, share_worker_bytecode_cache

FAKE_WORKER = Path(__file__).with_name("fake_worker.py")


@pytest.fixture(params=INTERPRETER_IDS)
def interpreter(request: pytest.FixtureRequest) -> str:
    return resolve_interpreter(request.param)


@pytest.fixture
def sdk(tmp_path: Path) -> SdkArtifact:
    return materialize_sdk_artifact(tmp_path / "sdk")


@pytest.fixture
def sdk_dir(sdk: SdkArtifact) -> Path:
    return sdk.path


_PREPARED: dict[tuple[str, str], PreparedEnvironment] = {}


async def prepared_environment(interpreter: str, sdk: SdkArtifact) -> PreparedEnvironment:
    """The default-mode environment for *interpreter*, probed once per process."""
    key = (interpreter, sdk.digest)
    if key not in _PREPARED:
        plan = DefaultPlan(parse_environment_request(b""), interpreter)
        _PREPARED[key] = await WorkflowEnvironmentManager(sdk_digest=sdk.digest).prepare(plan)
    return _PREPARED[key]


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "workspace"
    path.mkdir()
    return path


@pytest.fixture(scope="session")
def session_bytecode_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("worker-pycache")


@pytest.fixture
def bytecode_cache(session_bytecode_cache: Path) -> Path:
    """The worker's private bytecode cache, warm across tests; ``launch`` creates it."""
    return session_bytecode_cache


@pytest.fixture(autouse=True)
def _shared_bytecode_cache(monkeypatch: pytest.MonkeyPatch, session_bytecode_cache: Path) -> None:
    share_worker_bytecode_cache(monkeypatch, session_bytecode_cache)


Launcher = Callable[..., "Any"]


@pytest.fixture
async def launch(sdk: SdkArtifact, workspace: Path, bytecode_cache: Path) -> AsyncIterator[Launcher]:
    """Launch clients that are closed when the test ends; ``interpreter=`` picks the environment, other kwargs pass through."""
    clients: list[WorkflowWorkerClient] = []

    async def _launch(**kwargs: Any) -> WorkflowWorkerClient:
        environment = await prepared_environment(kwargs.pop("interpreter", sys.executable), sdk)
        options: dict[str, Any] = {
            "environment": environment,
            "sdk": sdk,
            "workspace": workspace,
            "bytecode_cache": bytecode_cache,
        }
        options.update(kwargs)
        client = await WorkflowWorkerClient.launch(**options)
        clients.append(client)
        return client

    yield _launch
    for client in clients:
        await client.close()

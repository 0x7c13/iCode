# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared fixtures for main-screen tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from chrys.orchestration.workflows import catalog as catalog_module
from tests.support.workflow_previews import reused_preview_workflow
from tests.support.workflow_workers import share_worker_bytecode_cache


@pytest.fixture(autouse=True)
def reuse_workflow_previews(monkeypatch: pytest.MonkeyPatch) -> None:
    """Workflow screens get real previews, but each distinct preview runs its interpreter and worker only once."""
    monkeypatch.setattr(catalog_module, "preview_workflow", reused_preview_workflow)


@pytest.fixture(scope="session")
def session_bytecode_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("worker-pycache")


@pytest.fixture(autouse=True)
def _shared_bytecode_cache(monkeypatch: pytest.MonkeyPatch, session_bytecode_cache: Path) -> None:
    share_worker_bytecode_cache(monkeypatch, session_bytecode_cache)

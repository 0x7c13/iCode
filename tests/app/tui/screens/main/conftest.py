# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared fixtures for main-screen tests."""

from __future__ import annotations

import pytest

from chrys.orchestration.workflows import catalog as catalog_module
from tests.support.workflow_previews import reused_preview_workflow


@pytest.fixture(autouse=True)
def reuse_workflow_previews(monkeypatch: pytest.MonkeyPatch) -> None:
    """Workflow screens get real previews, but each distinct preview runs its interpreter and worker only once."""
    monkeypatch.setattr(catalog_module, "preview_workflow", reused_preview_workflow)

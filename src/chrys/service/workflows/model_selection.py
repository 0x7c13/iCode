# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Resolve a Workflow default model and preserve its non-secret display snapshot."""

from __future__ import annotations

from chrys.foundation.models.workflow_session import WorkflowModelSelection
from chrys.service.profiles.models.registry import ModelProfileRegistry
from chrys.service.profiles.models.resolver import resolve_selectable_profile


def resolve_workflow_model(registry: ModelProfileRegistry | None, selector: str) -> WorkflowModelSelection | None:
    if not selector:
        return None
    profile = resolve_selectable_profile(registry, selector)
    if profile is None:
        raise ValueError(f"Workflow model profile {selector!r} is unavailable. Select another model before starting.")
    return WorkflowModelSelection(profile.id, profile.name, profile.model_id)

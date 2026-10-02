# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Profile YAML scalar coercion shared by the agent and model profile loaders."""

from __future__ import annotations


def coerce_bool(value: object, *, default: bool) -> bool:
    """Coerce YAML scalar bool-ish values into a Python bool."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Profile YAML scalar coercion shared by the agent and model profile loaders."""

from __future__ import annotations

from chrys.foundation.config.coercion import FALSY


def coerce_bool(value: object, *, default: bool) -> bool:
    """Coerce a YAML scalar into a bool; a missing value (``None``) keeps ``default``.

    Text is false when blank or one of the settings grammar's false words
    (``0``, ``false``, ``no``, ``off``, in any case) and true otherwise; any
    other value follows Python truthiness.
    """
    if value is None:
        return default
    if isinstance(value, str):
        word = value.strip().casefold()
        return bool(word) and word not in FALSY
    return bool(value)

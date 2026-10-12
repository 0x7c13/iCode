# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Which edits need a confirmation before they are applied."""

from __future__ import annotations

from typing import Any

from chrys.foundation.config.settings import Settings
from chrys.foundation.config.spec import Kind, Risk, specs_by_key

APPROVAL_MODE_KEY = "approval.default_mode"


def is_dangerous_transition(key: str, old: Any, new: Any) -> bool:
    """Whether changing *key* from *old* to *new* lowers a safety posture.

    Turning a ``DANGEROUS`` boolean on, or moving the default approval mode
    onto an automatic mode (fewer human approvals), asks first; the reverse never does.
    """
    if key == APPROVAL_MODE_KEY:
        automatic_modes = {"auto", "auto-formal"}
        return new in automatic_modes and old not in automatic_modes
    entry = specs_by_key(Settings).get(key)
    if entry is None or entry.risk is not Risk.DANGEROUS or entry.kind is not Kind.BOOL:
        return False
    return bool(new) and not bool(old)

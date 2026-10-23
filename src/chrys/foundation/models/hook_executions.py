# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""What the TUI shows of the hook runs in flight."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

HookWaitPoint = Literal["call", "turn", "session", "none"]
"""When iCode waits for a run: at the call that fired it (blocking), at the
end of the turn (``drain_turn``), when the session closes (``drain_session``),
or never (``fire_and_forget`` and ``user_interrupt`` observers)."""


@dataclass(frozen=True, slots=True)
class HookExecutionView:
    """One hook run in flight (queued or running), as it is at one moment."""

    hook_id: str
    waited_at: HookWaitPoint
    queued: bool
    """Still waiting for a ``max_parallel_hooks`` slot."""
    started_at: float
    """When the run was registered, in this process's ``time.monotonic()``."""

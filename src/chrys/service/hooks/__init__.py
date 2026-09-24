# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Hook system — user-defined scripts/commands fired around agent lifecycle.

Hooks let users run external commands at well-defined points: session
start/end, turn boundaries, user prompts, and tool calls.  Each hook
is a ``script`` / ``command`` / ``shell`` invocation matched by event,
profile, and tool metadata.

Config lives at ``<config_dir>/hooks/hooks.yaml`` (or ``.yml`` / ``.json``
with the same schema).  See :mod:`chrys.service.hooks.schema` for the on-disk shape and
:mod:`chrys.service.hooks.manager` for the runtime orchestrator.
"""

from chrys.service.hooks.events import HookEvent
from chrys.service.hooks.manager import HookManager
from chrys.service.hooks.schema import (
    HookConfig,
    HookDecision,
    HookExecution,
    HookMatch,
    HookRun,
    HookSettings,
    HooksFile,
)

__all__ = [
    "HookConfig",
    "HookDecision",
    "HookEvent",
    "HookExecution",
    "HookManager",
    "HookMatch",
    "HookRun",
    "HookSettings",
    "HooksFile",
]

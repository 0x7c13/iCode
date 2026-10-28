# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The last step before a tool runs: take the right to run the version the hooks checked.

Innermost in every function-middleware chain whose outer event middleware
(``ToolEventMiddleware`` / ``SubAgentEventMiddleware``) checks calls with
``before_tool_call`` hooks. Everything before it prepares the call (file
locks, approval, an approval edit and its recheck); here the call's
``HookOperation`` is committed synchronously, so nothing changes between the
check and the state change. A call the hooks did not check through to the
end, whose arguments differ from that version, or whose operation ended
meanwhile, does not run. A call with no operation at all reached this chain
without the hook check: a wiring error, refused rather than run unchecked.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from chrys.kernel.middleware import FunctionMiddleware
from chrys.service.agent_middleware._metadata_keys import _HOOK_OPERATION_KEY
from chrys.service.agent_middleware.events.hook_dispatch import is_mutable_tool_args, reject_tool_call
from chrys.service.hooks.operation import HookOperation

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from chrys.kernel.middleware import FunctionInvocationContext
    from chrys.service.hooks.operation import CommitRefusal

logger = logging.getLogger(__name__)

_REFUSALS: dict[CommitRefusal, str] = {
    "unchecked": "Tool call was not run: the before_tool_call hooks did not finish checking it.",
    "changed": "Tool call was not run: its arguments changed after the before_tool_call hooks checked them.",
    "committed": "Tool call was not run: it already ran.",
}


class HookCommitMiddleware(FunctionMiddleware):
    """Commit the call's checked version, then run the tool; refuse a call that may not run."""

    async def process(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        metadata = context.metadata if isinstance(context.metadata, dict) else {}
        operation = metadata.get(_HOOK_OPERATION_KEY)
        if not isinstance(operation, HookOperation):
            logger.error("Tool %r reached the commit step without a hook check; refused", context.function.name)
            reject_tool_call(context, _REFUSALS["unchecked"])
            return
        arguments = context.arguments if is_mutable_tool_args(context.arguments) else {}
        refusal = operation.commit_arguments(arguments)
        if refusal == "invalidated":
            reject_tool_call(context, f"Tool call was dropped before it ran ({operation.end_reason or 'invalidated'}).")
            return
        if refusal is not None:
            reject_tool_call(context, _REFUSALS[refusal])
            return
        await call_next()

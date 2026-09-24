# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Interrupt middleware — stops execution at the next tool call boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from chrys.kernel.middleware import FunctionMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from chrys.kernel.middleware import FunctionInvocationContext


class InterruptMiddleware(FunctionMiddleware):
    """Checks for an interrupt flag between tool calls.

    When the interrupt flag is set, raises MiddlewareTermination to stop
    execution at the next tool call boundary.
    """

    def __init__(self) -> None:
        self._interrupted = False

    @property
    def is_interrupted(self) -> bool:
        """True if an interrupt has been signalled and not yet reset."""
        return self._interrupted

    def set_interrupted(self) -> None:
        """Signal that execution should stop at the next tool call."""
        self._interrupted = True

    def reset(self) -> None:
        """Clear the interrupt flag."""
        self._interrupted = False

    async def process(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        if self._interrupted:
            from chrys.kernel.middleware import MiddlewareTermination

            raise MiddlewareTermination("Execution interrupted by user")

        await call_next()

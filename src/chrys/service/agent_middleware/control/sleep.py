# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Sleep middleware — intercepts sleep tool calls so the TUI can skip them."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from chrys.foundation.events.types import SleepSkip, UserInterrupt
from chrys.foundation.tool_kinds import KIND_SLEEP, get_tool_kind
from chrys.kernel.middleware import FunctionMiddleware
from chrys.service.agent_middleware._metadata_keys import _SLEEP_INTERRUPTED_KEY, _SLEEP_SKIPPED_KEY
from chrys.service.agent_middleware.events.hook_dispatch import get_call_id
from chrys.service.tools.builtins.sleep import format_sleep_seconds, normalize_sleep_seconds, validate_sleep_seconds

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from chrys.foundation.events.bus import EventBus
    from chrys.kernel.middleware import FunctionInvocationContext


class SleepMiddleware(FunctionMiddleware):
    """Intercept ``sleep`` tool calls and race the timer against skip/interrupt events."""

    def __init__(self, event_bus: EventBus, session_id: str | None = None) -> None:
        self._bus = event_bus
        self._session_id = session_id
        self._active: dict[str, asyncio.Future[str]] = {}

    @property
    def active_call_ids(self) -> tuple[str, ...]:
        """Call ids for sleep invocations that can currently be interrupted."""
        return tuple(self._active)

    def interrupt_active(self) -> tuple[str, ...]:
        """Resolve all active sleeps as interrupted and return their call ids."""
        interrupted: list[str] = []
        for call_id, future in list(self._active.items()):
            if not future.done():
                future.set_result("interrupted")
                interrupted.append(call_id)
        return tuple(interrupted)

    async def process(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        if get_tool_kind(context.function) != KIND_SLEEP:
            await call_next()
            return

        args = context.arguments if isinstance(context.arguments, dict) else {}
        seconds = normalize_sleep_seconds(args.get("seconds"))
        error = validate_sleep_seconds(seconds)
        if error:
            context.result = error
            return
        assert seconds is not None

        call_id = get_call_id(context)
        start = time.monotonic()
        loop = asyncio.get_running_loop()
        resolved: asyncio.Future[str] = loop.create_future()
        if call_id:
            self._active[call_id] = resolved

        async def _skip_handler(event: SleepSkip) -> None:
            if call_id and event.call_id == call_id and not resolved.done():
                resolved.set_result("skipped")

        async def _interrupt_handler(event: UserInterrupt) -> None:
            if event.session_id and self._session_id and event.session_id != self._session_id:
                return
            # In engine-managed runs, Executor.interrupt() usually resolves active
            # sleeps first so they can publish ToolCallResult before task cancellation.
            # Keep this bus handler as the standalone middleware contract: callers
            # that publish UserInterrupt without an Executor still interrupt sleep.
            if not resolved.done():
                resolved.set_result("interrupted")

        await self._bus.subscribe(SleepSkip, _skip_handler)
        await self._bus.subscribe(UserInterrupt, _interrupt_handler)
        try:
            while True:
                elapsed = time.monotonic() - start
                remaining = seconds - elapsed
                if remaining <= 0:
                    context.result = f"Slept for {format_sleep_seconds(seconds)}."
                    return
                try:
                    # Wake in short ticks so cancellation and timer drift stay bounded.
                    # ``shield`` keeps the shared event Future reusable after wait_for timeouts.
                    resolution = await asyncio.wait_for(asyncio.shield(resolved), timeout=min(1.0, remaining))
                except TimeoutError:
                    continue
                elapsed_seconds = min(seconds, int(time.monotonic() - start))
                if resolution == "interrupted":
                    if isinstance(context.metadata, dict):
                        context.metadata[_SLEEP_INTERRUPTED_KEY] = True
                    context.result = (
                        f"Sleep interrupted after {format_sleep_seconds(elapsed_seconds)} "
                        f"(requested {format_sleep_seconds(seconds)})."
                    )
                else:
                    if isinstance(context.metadata, dict):
                        context.metadata[_SLEEP_SKIPPED_KEY] = True
                    context.result = (
                        f"Sleep skipped by user after {format_sleep_seconds(elapsed_seconds)} "
                        f"(requested {format_sleep_seconds(seconds)})."
                    )
                return
        finally:
            if call_id and self._active.get(call_id) is resolved:
                del self._active[call_id]
            await self._bus.unsubscribe(SleepSkip, _skip_handler)
            await self._bus.unsubscribe(UserInterrupt, _interrupt_handler)

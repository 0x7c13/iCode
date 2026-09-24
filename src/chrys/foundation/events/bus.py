# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""EventBus — async pub/sub event dispatcher."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, TypeVar

from chrys.foundation.events.types import Event

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

E = TypeVar("E", bound=Event)
logger = logging.getLogger(__name__)

# Stream queues are intentionally unbounded so a slow consumer never causes event
# loss: dropping here would silently truncate streamed LLM output on the
# headless/ACP path (see ``session_host._HEADLESS_RUN_EVENT_TYPES``, which carries
# ``InvocationMessage``/``AgentThinking``). This threshold only drives a one-shot
# diagnostic warning so a badly-lagging consumer is observable in the wild —
# delivery semantics never change (events are retained, never dropped).
_STREAM_QUEUE_HIGH_WATER = 10_000


class EventBus:
    """In-process async event bus.

    Supports both callback-based subscriptions and async iteration (streaming).
    Designed as the sole communication channel between frontend and backend.

    Two delivery mechanisms with deliberately different backpressure:

    * ``subscribe()`` callbacks are awaited inline by :meth:`publish`, so a slow
      handler backpressures the publisher. Never lossy. Downstream ordering
      contract: the TUI's tool-card routing relies on this — a tool's
      InvocationToolCallStart handler chain completes before the tool executes, so a
      call's result can never overtake its own start (see
      ``app.tui.widgets.chat.tool_registry``). Making publish fire-and-forget
      would silently re-open mid-mount routing races there.
    * ``stream()`` iterators receive events through an **unbounded** per-consumer
      queue (:class:`_EventStream`). :meth:`publish` enqueues without awaiting, so
      it never blocks on a slow consumer, and the queue is unbounded so it never
      drops. A stalled consumer therefore grows memory (bounded in practice by a
      single turn's events — the stream is opened and torn down per turn) rather
      than losing events.
    """

    def __init__(self) -> None:
        self._handlers: dict[type[Event], list[Callable[..., Awaitable[None]]]] = defaultdict(list)
        self._streams: list[_EventStream] = []

    async def publish(self, event: Event, *, raise_handler_errors: bool = False) -> None:
        """Publish an event to all subscribers."""
        event_type = type(event)
        failures: list[Exception] = []
        for handler in list(self._handlers.get(event_type, [])):
            try:
                await handler(event)
            except Exception as exc:
                failures.append(exc)
                logger.exception("EventBus handler failed for %s", event_type.__name__)
        for stream in list(self._streams):
            stream._deliver(event)
        if raise_handler_errors and failures:
            if len(failures) == 1:
                raise failures[0]
            raise ExceptionGroup(f"EventBus handlers failed for {event_type.__name__}", failures)

    async def subscribe(self, event_type: type[E], handler: Callable[[E], Awaitable[None]]) -> None:
        """Register a callback handler for a specific event type."""
        self._handlers[event_type].append(handler)  # type: ignore[arg-type]

    async def unsubscribe(self, event_type: type[E], handler: Callable[..., Awaitable[None]]) -> None:
        """Remove a previously registered handler."""
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    def stream(self, *event_types: type[Event]) -> _EventStream:
        """Create an async iterator that yields events of the given types.

        Usage::

            async for event in bus.stream(InvocationMessage, InvocationToolCallStart):
                ...
        """
        return _EventStream(self, event_types)


class _EventStream:
    """Async iterator adapter for EventBus streaming.

    Backed by an unbounded queue: events are retained, never dropped, so a slow
    consumer cannot truncate streamed output (notably LLM text on the
    headless/ACP path). See the module-level ``_STREAM_QUEUE_HIGH_WATER`` note.
    """

    def __init__(self, bus: EventBus, event_types: tuple[type[Event], ...]) -> None:
        self._bus = bus
        self._event_types = event_types
        self._queue: asyncio.Queue[Event] = asyncio.Queue()  # unbounded — lossless by design
        self._warned_high_water = False

    def _deliver(self, event: Event) -> None:
        """Enqueue an event this stream subscribed to — never drops.

        Events are filtered here (producer side) so the unbounded queue only ever
        retains events the stream actually asked for: a narrow ``stream(X)`` never
        accumulates unrelated traffic, and the high-water backlog counts only
        matching events. The queue is unbounded, so ``put_nowait`` cannot raise
        ``QueueFull``; we emit a single high-water warning if the matching backlog
        grows large, so a stalled consumer is observable without discarding an
        event. An empty ``_event_types`` means "subscribe to all".
        """
        if self._event_types and not isinstance(event, self._event_types):
            return
        self._queue.put_nowait(event)
        if not self._warned_high_water and self._queue.qsize() >= _STREAM_QUEUE_HIGH_WATER:
            self._warned_high_water = True
            logger.warning(
                "EventBus stream backlog exceeded %d events; consumer is falling behind "
                "(events are retained, not dropped)",
                _STREAM_QUEUE_HIGH_WATER,
            )

    async def __aenter__(self) -> _EventStream:
        self._bus._streams.append(self)
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._bus._streams.remove(self)

    def __aiter__(self) -> AsyncIterator[Event]:
        return self

    async def __anext__(self) -> Event:
        # Filtering happens producer-side in ``_deliver``; everything queued here
        # already matches ``_event_types``.
        return await self._queue.get()

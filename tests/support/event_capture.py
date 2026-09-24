# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Event-bus capture and display-message assertion helpers shared across engine tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import Error, Event, Warning


async def collect_events[T](events: list[T], event: T) -> None:
    """Append ``event`` to ``events``.

    Bind the list before subscribing::

        await bus.subscribe(Error, lambda event: collect_events(errors, event))
    """
    events.append(event)


async def capture_events[E](bus: EventBus, event_type: type[E]) -> list[E]:
    """Subscribe a collector for ``event_type`` on ``bus`` and return the list it appends to."""
    events: list[E] = []

    async def collect(event: E) -> None:
        events.append(event)

    await bus.subscribe(event_type, collect)
    return events


def assert_display_message(event: Error | Warning, key: str, args: Mapping[str, object] | None = None) -> None:
    """Assert ``event`` carries the localized message ``key`` with exactly ``args``."""
    reference = event.display_message
    assert reference is not None
    assert reference.definition.key == key
    assert dict(reference.args) == dict(args or {})


@asynccontextmanager
async def capture_event_sequence(bus: EventBus, *event_types: type[Event]) -> AsyncIterator[list[Event]]:
    """Capture cross-type publication order and release subscriptions on every exit.

    Keep original objects for identity/causality assertions. Sibling scheduling
    is deliberately not sorted: callers assert only the ordering they own.
    """
    events: list[Event] = []

    async def collect(event: Event) -> None:
        await collect_events(events, event)

    subscribed: list[type[Event]] = []
    try:
        for event_type in dict.fromkeys(event_types):
            await bus.subscribe(event_type, collect)
            subscribed.append(event_type)
        yield events
    finally:
        for event_type in reversed(subscribed):
            await bus.unsubscribe(event_type, collect)


class EventNormalizer:
    """Normalize explicitly selected random IDs/clocks, preserving all other data.

    One normalizer belongs to one capture, so equal IDs across event types stay
    equal. Batch IDs, attempt ordinals, counters, and source order stay intact.
    Callers name clock fields instead of accidentally hiding a new event field.
    """

    def __init__(self) -> None:
        self._identities: dict[str, str] = {}

    def identity(self, value: str) -> str:
        if not value:
            return value
        return self._identities.setdefault(value, f"id-{len(self._identities) + 1}")

    def event(
        self,
        event: Event,
        *,
        id_fields: tuple[str, ...] = ("session_id", "invocation_id", "call_id", "parent_call_id"),
        clock_fields: tuple[str, ...] = (),
    ) -> tuple[str, dict[str, object]]:
        values = asdict(event)
        origin = values.get("origin")
        while isinstance(origin, dict):
            for key in ("session_id", "invocation_id"):
                value = origin.get(key)
                if isinstance(value, str):
                    origin[key] = self.identity(value)
            origin = origin.get("parent")
        for key in id_fields:
            value = values.get(key)
            if isinstance(value, str):
                values[key] = self.identity(value)
        for key in clock_fields:
            if key in values:
                values[key] = "<clock>"
        return type(event).__name__, values

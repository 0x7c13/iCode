# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Data envelope and node context of the workflow SDK (pure stdlib, Python 3.9)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Optional, Union

# Strict recursive JSON. ``typing.Union`` is the one sanctioned exception to
# the modern-syntax rule: a runtime ``X | Y`` alias does not exist on 3.9.
JsonValue = Union[dict[str, "JsonValue"], list["JsonValue"], str, int, float, bool, None]


@dataclass(frozen=True)
class WorkflowValue:
    """The single envelope routed between nodes.

    ``text`` is what prompts, the TUI and logs consume; ``data`` is an optional
    structured payload for program consumers. Serializability is checked at
    the process boundary, not here, so the envelope stays a plain value.
    """

    text: str
    data: Optional[JsonValue] = None


@dataclass(frozen=True)
class SourceValue:
    """One resolved input of a fan-in node, in declared source order."""

    node_id: str
    activation_id: str
    value: WorkflowValue


class NodeContext:
    """Per-activation context handed to ``fn(value, ctx)`` node bodies.

    The worker host constructs it with the two callbacks wired to the wire
    protocol; user code only calls :meth:`emit` and awaits :meth:`ask`.
    """

    def __init__(self, *, emit: Callable[[str], None], ask: Callable[[str], Awaitable[str]]) -> None:
        self._emit = emit
        self._ask = ask

    def emit(self, text: str) -> None:
        """Publish a piece of process output for this activation."""
        if not isinstance(text, str):
            raise TypeError("ctx.emit() expects a str.")
        self._emit(text)

    async def ask(self, prompt: str) -> str:
        """Ask the person driving the run; only usable from an ``async def`` body."""
        if not isinstance(prompt, str):
            raise TypeError("ctx.ask() expects a str prompt.")
        return await self._ask(prompt)

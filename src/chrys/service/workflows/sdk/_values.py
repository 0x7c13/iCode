# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Data envelope and node context of the workflow SDK (pure stdlib, Python 3.9)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union, overload

from ._ask import Answer, Question, answer_from_wire, questions_to_wire

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

    def __init__(
        self,
        *,
        emit: Callable[[str], None],
        ask: Callable[[list[dict[str, Any]]], Awaitable[list[dict[str, Any]]]],
    ) -> None:
        self._emit = emit
        self._ask = ask

    def emit(self, text: str) -> None:
        """Publish a piece of process output for this activation."""
        if not isinstance(text, str):
            raise TypeError("ctx.emit() expects a str.")
        self._emit(text)

    @overload
    async def ask(self, prompt: str) -> str: ...

    @overload
    async def ask(self, prompt: Question) -> Answer: ...

    @overload
    async def ask(self, prompt: Sequence[Question]) -> tuple[Answer, ...]: ...

    async def ask(self, prompt: Union[str, Question, Sequence[Question]]) -> Union[str, Answer, tuple[Answer, ...]]:
        """Ask the person driving the run; only usable from an ``async def`` body.

        A ``str`` asks one open question and returns the typed text. A
        :class:`Question` returns one :class:`Answer`. A list or tuple of 1-5
        questions shows them in one dialog and returns a tuple of answers in
        the same order.
        """
        if isinstance(prompt, str):
            if not prompt.strip():
                raise ValueError("ctx.ask() prompt must not be blank.")
            answers = await self._ask_questions((Question(prompt),))
            return answers[0].text
        if isinstance(prompt, Question):
            return (await self._ask_questions((prompt,)))[0]
        if isinstance(prompt, (list, tuple)):
            return await self._ask_questions(tuple(prompt))
        raise TypeError("ctx.ask() expects a str, a Question, or a list or tuple of Questions.")

    async def _ask_questions(self, questions: tuple[Question, ...]) -> tuple[Answer, ...]:
        wire = questions_to_wire(questions)
        return tuple(answer_from_wire(answer) for answer in await self._ask(wire))

# Copyright (c) Microsoft. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Microsoft Agent Framework (MIT License; see NOTICE).

"""Streamed chunks read back as chat response updates.

Tool calls arrive in fragments keyed by an index and, usually, an id. One
:class:`StreamState` per stream assembles them per choice and releases each
call once, whole, in an update of its own just before the update that
finishes its choice, or at the end of a stream no chunk finished. The kernel
never sees a fragment.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from openai.types import CompletionUsage
from openai.types.chat.chat_completion_message_custom_tool_call import ChatCompletionMessageCustomToolCall
from pydantic import ValidationError

from chrys.kernel import ChatResponseUpdate, Content, FinishReason
from chrys.kernel.exceptions import ChatClientInvalidResponseException
from chrys.service.llm.openai_timestamps import normalize_openai_created_payload, openai_created_at_iso

from .decode import choice_metadata, decode_usage, response_metadata, text_contents
from .reasoning import REASONING_DETAILS_FIELD, REASONING_FORMAT_KEY, delta_reasoning, reasoning_fields
from .validation import raise_invalid_stream_event

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice

    from .client import ChatCompletionsVariant

logger = logging.getLogger(__name__)


@dataclass
class _PendingCall:
    """One function call assembled from its fragments."""

    choice: int
    index: int | None
    # Position among all calls of the stream, in the order first seen.
    order: int
    call_id: str | None = None
    name: str | None = None
    arguments: list[str] = field(default_factory=list)
    raws: list[Any] = field(default_factory=list)

    def content(self) -> Content:
        return Content.from_function_call(
            call_id=self.call_id or "",
            name=self.name or "",
            arguments="".join(self.arguments),
            raw_representation=self.raws[0] if len(self.raws) == 1 else self.raws,
        )


@dataclass
class _ChoiceState:
    """What one choice of the stream has assembled."""

    calls: list[_PendingCall] = field(default_factory=list)
    by_index: dict[int, _PendingCall] = field(default_factory=dict)
    by_id: dict[str, _PendingCall] = field(default_factory=dict)
    # An empty plaintext reasoning field was reported once.
    empty_reasoning_sent: bool = False
    # Non-empty plaintext reasoning arrived.
    reasoning_seen: bool = False
    # The choice's finish reason arrived; later fragments are ignored.
    finished: bool = False


class StreamState:
    """The assembly state of one stream."""

    def __init__(self, variant: ChatCompletionsVariant) -> None:
        self._variant = variant
        self._choices: dict[int, _ChoiceState] = {}
        self._next_order = 0

    def updates_for(self, chunk: ChatCompletionChunk) -> list[ChatResponseUpdate]:
        """The updates one chunk yields: calls its finish reasons complete, then its own."""
        choices = getattr(chunk, "choices", None)
        if not isinstance(choices, list):
            raise_invalid_stream_event(chunk, choices)
        if not choices and chunk.usage is None:
            return []
        update = self.update_for(chunk)
        finished = {choice.index: choice.finish_reason for choice in chunk.choices if choice.finish_reason is not None}
        # A consumer may stop at a finish reason, so the calls go out first.
        calls = self._release(finished) if finished else None
        return [update] if calls is None else [calls, update]

    def finish(self) -> ChatResponseUpdate | None:
        """The calls still pending at the end; some gateways never send a finish reason."""
        return self._release(None)

    def update_for(self, chunk: ChatCompletionChunk) -> ChatResponseUpdate:
        """The update for one chunk, its call fragments held back.

        A ``created`` in milliseconds is converted on a copy, which is also the
        update's raw representation; the SDK chunk stays as received.
        """
        chunk = normalize_openai_created_payload(chunk)
        metadata = response_metadata(chunk)
        contents: list[Content] = []
        finish: FinishReason | None = None
        # Usage can share a chunk with content (Gemini); both are kept.
        if usage := chunk.usage or _choice_level_usage(chunk):
            contents.append(
                Content.from_usage(usage_details=decode_usage(usage, variant=self._variant), raw_representation=chunk)
            )
        for choice in chunk.choices:
            metadata.update(choice_metadata(choice))
            if choice.finish_reason:
                finish = FinishReason(choice.finish_reason)
            # Some compatible providers finish with ``"delta": null``.
            if choice.delta is not None:
                contents.extend(self._delta_contents(choice))
        return ChatResponseUpdate(
            contents=contents,
            role="assistant",
            response_id=chunk.id,
            message_id=chunk.id,
            model=chunk.model,
            created_at=openai_created_at_iso(chunk.created),
            finish_reason=finish,
            additional_properties=metadata,
            raw_representation=chunk,
        )

    def _choice(self, index: int) -> _ChoiceState:
        return self._choices.setdefault(index, _ChoiceState())

    def _delta_contents(self, choice: ChunkChoice) -> list[Content]:
        state = self._choice(choice.index)
        fields = reasoning_fields(choice.delta)
        plain = next(((name, value) for name, value in fields.items() if name != REASONING_DETAILS_FIELD), None)
        contents: list[Content] = []
        include_plain = True
        if plain is not None and plain[1] == "":
            include_plain = False
            if not (state.empty_reasoning_sent or state.reasoning_seen):
                # The field's presence is kept once, ahead of answer text and
                # of the calls released at the finish: the text does not
                # fragment, and a call cut off at the length limit stays the
                # response's last content.
                contents.append(
                    Content.from_text_reasoning(
                        text="", additional_properties={REASONING_FORMAT_KEY: plain[0]}, raw_representation=choice.delta
                    )
                )
                state.empty_reasoning_sent = True
        elif plain is not None:
            state.reasoning_seen = True
        if choice.delta.tool_calls:
            self._collect_calls(choice.index, state, choice.delta.tool_calls)
        # A delta's fields carry no order among themselves: one that mixes
        # them reads as reasoning ending and the answer starting, so chunking
        # alone never moves reasoning into the answer text.
        contents.extend(delta_reasoning(fields, include_plain=include_plain))
        contents.extend(text_contents(choice))
        return contents

    def _collect_calls(self, choice: int, state: _ChoiceState, fragments: list[Any]) -> None:
        """Add call fragments to the calls they continue.

        A non-empty id names its call, as some gateways send several whole
        calls under one index. A fragment without one continues the call its
        index last named, or the only pending call when that is unambiguous.
        """
        if state.finished:
            logger.warning(
                "Ignoring streamed tool-call fragment received after the terminal update for choice %d", choice
            )
            return
        for fragment in fragments:
            if isinstance(fragment, ChatCompletionMessageCustomToolCall):
                continue
            index = _fragment_index(fragment)
            call_id = _fragment_call_id(fragment)
            function = getattr(fragment, "function", None)
            if call_id is None:
                pending = self._call_without_id(choice, state, index)
            else:
                pending = self._call_with_id(choice, state, call_id, index)
            if index is not None:
                # The id wins over the index: Gemini-compatible endpoints
                # reuse index 0 for distinct whole calls.
                state.by_index[index] = pending
                if pending.index is None:
                    pending.index = index
            name = getattr(function, "name", None)
            if isinstance(name, str) and name:
                if pending.name is not None and pending.name != name:
                    raise ChatClientInvalidResponseException(
                        f"Conflicting streamed tool-call names for choice {choice}, index {index!r}."
                    )
                pending.name = name
            if (arguments := getattr(function, "arguments", None)) is not None:
                if not isinstance(arguments, str):
                    raise ChatClientInvalidResponseException(
                        f"Non-string streamed tool-call arguments for choice {choice}, index {index!r}."
                    )
                pending.arguments.append(arguments)
            pending.raws.append(fragment if function is None else function)

    def _call_with_id(self, choice: int, state: _ChoiceState, call_id: str, index: int | None) -> _PendingCall:
        if (known := state.by_id.get(call_id)) is not None:
            return known
        indexed = state.by_index.get(index) if index is not None else None
        if indexed is not None and indexed.call_id is None:
            # Some endpoints send the id only after a call's first fragment.
            pending = indexed
        elif index is None and len(state.calls) == 1 and state.calls[0].call_id is None:
            # The same, with neither fragment carrying an index.
            pending = state.calls[0]
        else:
            pending = self._new_call(choice, state, index)
        pending.call_id = call_id
        state.by_id[call_id] = pending
        return pending

    def _call_without_id(self, choice: int, state: _ChoiceState, index: int | None) -> _PendingCall:
        if index is not None and (indexed := state.by_index.get(index)) is not None:
            return indexed
        only = state.calls[0] if len(state.calls) == 1 else None
        if only is not None and (index is None or only.index is None):
            return only
        if index is not None or not state.calls:
            return self._new_call(choice, state, index)
        raise ChatClientInvalidResponseException(
            f"Ambiguous streamed tool-call fragment without a call id for choice {choice}, index {index!r}."
        )

    def _new_call(self, choice: int, state: _ChoiceState, index: int | None) -> _PendingCall:
        pending = _PendingCall(choice=choice, index=index, order=self._next_order)
        self._next_order += 1
        state.calls.append(pending)
        if index is not None:
            state.by_index[index] = pending
        return pending

    def _release(self, finished: Mapping[int, str | None] | None) -> ChatResponseUpdate | None:
        """One update with the complete calls of the finished choices, or of every choice at the end."""
        contents: list[Content] = []
        for choice in sorted(self._choices if finished is None else finished):
            state = self._choices.get(choice)
            if state is None:
                if finished is not None:
                    # A choice finished (``delta: null``) before sending
                    # anything. Remembering it keeps a late fragment from
                    # starting a call the end of the stream would release.
                    self._choice(choice).finished = True
                continue
            contents.extend(_complete_calls(choice, state.calls, None if finished is None else finished.get(choice)))
            state.calls.clear()
            state.by_index.clear()
            state.by_id.clear()
            if finished is not None:
                state.finished = True
        return ChatResponseUpdate(contents=contents) if contents else None


def _complete_calls(choice: int, calls: list[_PendingCall], finish_reason: str | None) -> list[Content]:
    """The calls of one choice as contents, in index order.

    A call without an id keeps the kernel's id-less behavior. A call without a
    name is an error, except in a choice cut off at the length limit: the
    model never finished it, and the finish reason already says why.
    """
    ordered = sorted(calls, key=lambda call: (call.order if call.index is None else call.index, call.order))
    if len(ordered) > 1:
        _assign_local_ids(choice, ordered)
    contents: list[Content] = []
    for call in ordered:
        if not call.name:
            if finish_reason != "length":
                raise ChatClientInvalidResponseException(
                    f"Streamed tool call ended without a function name for choice {choice}, index {call.index!r}."
                )
            logger.warning(
                "Discarding truncated streamed tool call without a name for choice %d, index %r", choice, call.index
            )
            continue
        if not call.call_id:
            logger.debug(
                "OpenAI-compatible stream emitted a tool call without an id for choice %d, index %r; "
                "preserving legacy id-less behavior",
                choice,
                call.index,
            )
        contents.append(call.content())
    return contents


def _assign_local_ids(choice: int, calls: list[_PendingCall]) -> None:
    """Give the id-less calls of a parallel batch request-local ids.

    The kernel merges adjacent fragments that share an id, so two id-less
    calls would fuse; distinct ids keep them apart and pair each with its
    result.
    """
    taken = {call.call_id for call in calls if call.call_id}
    for call in calls:
        if call.call_id:
            continue
        base = f"call_chrys_{choice}_{call.order}"
        candidate, suffix = base, 1
        while candidate in taken:
            candidate = f"{base}_{suffix}"
            suffix += 1
        call.call_id = candidate
        taken.add(candidate)
        logger.debug("Assigned a local id to an id-less parallel tool call for choice %d, index %r", choice, call.index)


def _fragment_index(fragment: Any) -> int | None:
    """The fragment's index when usable; the SDK does not validate it on construction."""
    index = getattr(fragment, "index", None)
    return index if isinstance(index, int) and not isinstance(index, bool) and index >= 0 else None


def _fragment_call_id(fragment: Any) -> str | None:
    """The fragment's call id; blank and a stringified JSON ``null`` count as none."""
    call_id = getattr(fragment, "id", None)
    return call_id if isinstance(call_id, str) and call_id not in ("", "null") else None


def _choice_level_usage(chunk: ChatCompletionChunk) -> CompletionUsage | None:
    """Usage a provider (Kimi) streams on a choice instead of on the chunk."""
    for choice in chunk.choices:
        if raw := (choice.model_extra or {}).get("usage"):
            try:
                return CompletionUsage.model_validate(raw)
            except ValidationError:
                logger.debug("Ignoring malformed choice-level usage on stream chunk %s", chunk.id)
    return None

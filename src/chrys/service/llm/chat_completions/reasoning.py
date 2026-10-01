# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The reasoning fields Chat Completions dialects use, read and replayed.

Compatible endpoints put the model's reasoning in one of three message
fields: ``reasoning_details`` (OpenRouter; structured), ``reasoning_content``
(DeepSeek, GLM, Kimi; plaintext) or vLLM's newer ``reasoning`` (plaintext,
sometimes a mirror of one of the others). Each captured value is stamped
with the field it came from, and replay sends it back under that field only.
Replay depends on the dialect, never on the provider: Kimi runs on the plain
``openai`` provider.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from chrys.kernel import Content, Message
from chrys.kernel._content import _ANTHROPIC_REDACTED_THINKING_KEY

if TYPE_CHECKING:
    from .client import ChatCompletionsVariant

logger = logging.getLogger(__name__)

REASONING_DETAILS_FIELD = "reasoning_details"
REASONING_CONTENT_FIELD = "reasoning_content"
REASONING_FIELD = "reasoning"
REASONING_FORMAT_KEY = "openai_reasoning_format"

REASONING_FIELDS: tuple[str, ...] = (REASONING_DETAILS_FIELD, REASONING_CONTENT_FIELD, REASONING_FIELD)


def reasoning_fields(source: Any) -> dict[str, Any]:
    """The reasoning fields an SDK message or delta carries, by wire name.

    ``reasoning_details`` and ``reasoning_content`` come first, as they may
    hold state the provider needs back. A gateway can mirror the same
    plaintext into ``reasoning`` too, so that field counts only when neither
    of the others is present.
    """
    found = {
        name: value
        for name in (REASONING_DETAILS_FIELD, REASONING_CONTENT_FIELD)
        if (value := getattr(source, name, None)) is not None
    }
    if not found and (mirror := getattr(source, REASONING_FIELD, None)) is not None:
        found[REASONING_FIELD] = mirror
    return found


# --- capture ----------------------------------------------------------------


def message_reasoning(message: Any) -> list[Content]:
    """A whole choice message's reasoning: one protected JSON payload per field."""
    return [
        Content.from_text_reasoning(
            protected_data=json.dumps(value), additional_properties={REASONING_FORMAT_KEY: name}
        )
        for name, value in reasoning_fields(message).items()
    ]


def message_reasoning_props(message: Any) -> dict[str, Any]:
    """The same fields as message properties, the leading one named as the format.

    They duplicate the contents :func:`message_reasoning` returns; replay
    prefers the contents.
    """
    props = reasoning_fields(message)
    if props:
        props[REASONING_FORMAT_KEY] = next(iter(props))
    return props


def delta_reasoning(fields: Mapping[str, Any], *, include_plain: bool) -> list[Content]:
    """The reasoning contents of one streamed delta's fields.

    ``reasoning_details`` stays a protected payload. A plaintext field is the
    display text itself, left out when *include_plain* is false.
    """
    contents: list[Content] = []
    for name, value in fields.items():
        if name == REASONING_DETAILS_FIELD:
            contents.append(
                Content.from_text_reasoning(
                    protected_data=json.dumps(value), additional_properties={REASONING_FORMAT_KEY: name}
                )
            )
        elif not include_plain:
            continue
        elif isinstance(value, str):
            contents.append(Content.from_text_reasoning(text=value, additional_properties={REASONING_FORMAT_KEY: name}))
        else:
            logger.debug("Ignoring non-string Chat Completions reasoning delta of type %s", type(value).__name__)
    return contents


# --- replay -----------------------------------------------------------------


def replays_reasoning(messages: Sequence[Message], *, variant: ChatCompletionsVariant, request_has_tools: bool) -> bool:
    """Whether this request sends historical reasoning back.

    By default it always does: endpoints with preserved thinking (GLM) need it
    on every multi-turn request, and Kimi needs every assistant message's
    ``reasoning_content``. Scoping replay by provider is impossible (Kimi and
    OpenAI share ``openai``, differing only by base URL) and unnecessary:
    OpenAI's Chat Completions API drops unknown message-level fields before
    tokenization (checked live: no 400, no prompt tokens), so any history
    replays safely. Its strict validation covers top-level request
    parameters only.

    A variant that replays reasoning only with tools (DeepSeek) does so on a
    request that sends tools or follows a tool interaction.
    """
    if not variant.reasoning_with_tools or request_has_tools:
        return True
    return any(
        message.role == "tool"
        or any(content.type in ("function_call", "function_result") for content in message.contents)
        for message in messages
    )


def pad_reasoning_content(wire: list[dict[str, Any]]) -> None:
    """Give every assistant wire message a ``reasoning_content``, empty where it had none.

    Once tools are in play DeepSeek wants one on each assistant message, and
    history another model wrote (or DeepSeek with thinking off) has none.
    Only the wire copy changes.
    """
    for message in wire:
        if message.get("role") == "assistant":
            message.setdefault(REASONING_CONTENT_FIELD, "")


def contribution(content: Content) -> tuple[str, Any] | None:
    """The field and value one reasoning content replays as, or ``None``.

    The format stamp names the field. A stamp naming none of these fields
    belongs to another dialect, whose state would be forged by replaying it
    here. A protected payload replays when it decodes as JSON to something
    other than ``null``; anything else is another provider's opaque state
    (Responses encrypted reasoning, Anthropic signatures) and silences the
    content's text as well. Without a payload, a stamped content replays its
    text. An unstamped content, from before the stamp existed, replays only a
    decodable payload, as ``reasoning_details``.
    """
    properties = content.additional_properties
    if properties.get(_ANTHROPIC_REDACTED_THINKING_KEY):
        return None
    stamp = properties.get(REASONING_FORMAT_KEY)
    field = stamp if stamp in REASONING_FIELDS else None
    if field is None and REASONING_FORMAT_KEY in properties:
        return None
    if content.protected_data is not None:
        payload = _decoded_payload(content.protected_data)
        if payload is None:
            return None
        return field or REASONING_DETAILS_FIELD, payload
    if field is not None and content.text is not None:
        return field, content.text
    return None


def fold(fields: dict[str, Any], field: str, payload: Any) -> None:
    """Add one contribution to a per-field aggregate.

    Plaintext fields concatenate strings and ``reasoning_details`` extends
    lists; any other pairing keeps the later value.
    """
    if field in fields:
        earlier = fields[field]
        if field == REASONING_DETAILS_FIELD:
            if isinstance(earlier, list) and isinstance(payload, list):
                payload = [*cast("list[Any]", earlier), *cast("list[Any]", payload)]
        elif isinstance(earlier, str) and isinstance(payload, str):
            payload = earlier + payload
    fields[field] = payload


def replayable_fields(message: Message) -> dict[str, Any]:
    """The reasoning fields a message replays, aggregated in capture order.

    Its contents decide (see :func:`contribution`); the copy kept in the
    message properties fills in only fields no content supplied.
    """
    fields: dict[str, Any] = {}
    for content in message.contents:
        if content.type == "text_reasoning" and (found := contribution(content)) is not None:
            fold(fields, *found)
    fields.update(stored_reasoning(message, skip=fields))
    return fields


def stored_reasoning(message: Message, *, skip: Collection[str] = ()) -> dict[str, Any]:
    """The reasoning fields kept in the message properties, except those in *skip*."""
    properties = message.additional_properties
    return {
        name: properties[name] for name in REASONING_FIELDS if name not in skip and properties.get(name) is not None
    }


def _decoded_payload(protected_data: str) -> Any | None:
    """A Chat Completions client's own JSON payload; ``None`` for anything else."""
    try:
        return json.loads(protected_data)
    except ValueError:
        return None

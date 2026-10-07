# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""What every wire case is made of: the case type, scripted replies and payload builders.

Payloads are built from fixed values.  A streamed reply is generated from the
same final payload as its non-streamed twin, so the two differ only in how
the provider sent it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from chrys.kernel import FunctionTool, Message, tool
from chrys.service.profiles.models.schema import API_STYLE_CHAT_COMPLETIONS, ModelProfile

CREATED = 1_767_225_600
# Streamed text and arguments arrive in deltas of this many characters.
_SPLIT = 6

# ---------------------------------------------------------------------------
# Scripted replies
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Reply:
    """One scripted HTTP answer."""

    status: int = 200
    body: bytes = b""
    headers: tuple[tuple[str, str], ...] = ()
    # The connection is lost after the body: reading on fails as a reset connection.
    breaks_off: bool = False


def json_reply(payload: Any) -> Reply:
    return Reply(200, json.dumps(payload).encode("utf-8"), (("content-type", "application/json"),))


def sse_reply(events: Sequence[tuple[str | None, Any]]) -> Reply:
    """A ``text/event-stream`` body; each event is ``(name or None, data)``, data JSON-encoded unless a str."""
    chunks = []
    for name, data in events:
        text = data if isinstance(data, str) else json.dumps(data)
        chunks.append((f"event: {name}\n" if name else "") + f"data: {text}\n\n")
    return Reply(200, "".join(chunks).encode("utf-8"), (("content-type", "text/event-stream"),))


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Case:
    """One scripted exchange: a profile, the input, the options and the replies."""

    provider: str
    replies: tuple[Reply, ...]
    messages: Callable[[], list[Message]]
    options: Callable[[], dict[str, Any]]
    api_style: str = API_STYLE_CHAT_COMPLETIONS
    stream: bool = False
    base_url: str = ""
    profile_fields: Mapping[str, Any] = field(default_factory=dict)

    def profile(self) -> ModelProfile:
        return ModelProfile(
            id="golden-profile",
            name="golden",
            provider=self.provider,
            api_style=self.api_style,
            model_id="golden-model",
            api_key="sk-golden",
            base_url=self.base_url,
            stream=self.stream,
            **self.profile_fields,
        )


# ---------------------------------------------------------------------------
# Shared inputs
# ---------------------------------------------------------------------------


def _lookup(city: str) -> str:
    return f"Sunny in {city}, 21 degrees."


def lookup_tool() -> FunctionTool:
    """A fresh ``lookup`` tool: a client may write into a tool's cached schema, so no two cases share one."""
    return tool(name="lookup", description="Look up the weather in a city.")(_lookup)


class WeatherReport(BaseModel):
    city: str
    summary: str
    celsius: int


WEATHER_REPORT_JSON = '{"city": "Paris", "summary": "Sunny", "celsius": 21}'
LOOKUP_ARGUMENTS = '{"city": "Paris"}'


def weather_question() -> list[Message]:
    return [
        Message("system", ["You are a concise assistant."]),
        Message("user", ["What is the weather in Paris?"]),
    ]


def with_lookup() -> dict[str, Any]:
    return {"tools": [lookup_tool()]}


# ---------------------------------------------------------------------------
# Chat Completions payloads
# ---------------------------------------------------------------------------

CC_USAGE_1 = {
    "prompt_tokens": 40,
    "completion_tokens": 12,
    "total_tokens": 52,
    "prompt_tokens_details": {"cached_tokens": 8},
    "completion_tokens_details": {"reasoning_tokens": 5},
}
CC_USAGE_2 = {"prompt_tokens": 70, "completion_tokens": 9, "total_tokens": 79}


def cc_completion(
    *,
    response_id: str,
    message: Mapping[str, Any],
    finish_reason: str,
    usage: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": CREATED,
        "model": "golden-model-2026",
        "system_fingerprint": "fp_golden",
        "choices": [{"index": 0, "message": dict(message), "finish_reason": finish_reason, "logprobs": None}],
        "usage": dict(usage),
    }


def cc_chunks(completion: Mapping[str, Any]) -> list[tuple[str | None, Any]]:
    """The chunk stream a provider sends for *completion*.

    The role opens, reasoning and text follow in short deltas,
    then each tool call's head and argument fragments, the finish chunk, the
    usage chunk and ``[DONE]``.
    """
    head = {key: completion[key] for key in ("id", "created", "model", "system_fingerprint")}
    choice = completion["choices"][0]
    message = choice["message"]

    def chunk(delta: Mapping[str, Any], finish: str | None = None) -> tuple[None, dict[str, Any]]:
        return None, {
            **head,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}],
        }

    events = [chunk({"role": "assistant", "content": ""})]
    for key in ("reasoning_content", "content"):
        text = message.get(key)
        if text:
            events.extend(chunk({key: text[start : start + _SPLIT]}) for start in range(0, len(text), _SPLIT))
    for index, call in enumerate(message.get("tool_calls") or ()):
        arguments = call["function"]["arguments"]
        head_delta = {
            "index": index,
            "id": call["id"],
            "type": "function",
            "function": {"name": call["function"]["name"], "arguments": ""},
        }
        events.append(chunk({"tool_calls": [head_delta]}))
        events.extend(
            chunk({"tool_calls": [{"index": index, "function": {"arguments": arguments[start : start + _SPLIT]}}]})
            for start in range(0, len(arguments), _SPLIT)
        )
    events.append(chunk({}, choice["finish_reason"]))
    events.append((None, {**head, "object": "chat.completion.chunk", "choices": [], "usage": completion["usage"]}))
    events.append((None, "[DONE]"))
    return events


def cc_lookup_call() -> dict[str, Any]:
    return {"id": "call_lookup_1", "type": "function", "function": {"name": "lookup", "arguments": LOOKUP_ARGUMENTS}}


def cc_weather_turns(
    *, reasoning: Mapping[str, Any] | None = None, final_reasoning: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """A text-plus-tool-call answer, then the final answer after the tool result."""
    first: dict[str, Any] = {
        "role": "assistant",
        "content": "Let me check the weather.",
        "tool_calls": [cc_lookup_call()],
        **(reasoning or {}),
    }
    second: dict[str, Any] = {
        "role": "assistant",
        "content": "It is sunny in Paris, 21 degrees.",
        **(final_reasoning or {}),
    }
    return (
        cc_completion(response_id="chatcmpl-golden-1", message=first, finish_reason="tool_calls", usage=CC_USAGE_1),
        cc_completion(response_id="chatcmpl-golden-2", message=second, finish_reason="stop", usage=CC_USAGE_2),
    )


def cc_text(text: str, *, response_id: str) -> dict[str, Any]:
    return cc_completion(
        response_id=response_id,
        message={"role": "assistant", "content": text},
        finish_reason="stop",
        usage=CC_USAGE_2,
    )


def cc_replies(completions: Sequence[Mapping[str, Any]], *, stream: bool) -> tuple[Reply, ...]:
    if stream:
        return tuple(sse_reply(cc_chunks(completion)) for completion in completions)
    return tuple(json_reply(completion) for completion in completions)


# ---------------------------------------------------------------------------
# Responses payloads
# ---------------------------------------------------------------------------

RESP_USAGE_1 = {
    "input_tokens": 40,
    "input_tokens_details": {"cached_tokens": 8},
    "output_tokens": 12,
    "output_tokens_details": {"reasoning_tokens": 5},
    "total_tokens": 52,
}
RESP_USAGE_2 = {
    "input_tokens": 70,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens": 9,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 79,
}


def resp_response(
    *,
    response_id: str,
    output: Sequence[Mapping[str, Any]],
    usage: Mapping[str, Any] = RESP_USAGE_2,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": CREATED,
        "status": "completed",
        "model": "golden-model-2026",
        "output": [dict(item) for item in output],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "error": None,
        "incomplete_details": None,
        "usage": dict(usage),
    }


def resp_reasoning(item_id: str, *summaries: str, encrypted: str | None = None) -> dict[str, Any]:
    """A reasoning item: summary parts and optional encrypted content."""
    item: dict[str, Any] = {
        "type": "reasoning",
        "id": item_id,
        "summary": [{"type": "summary_text", "text": summary} for summary in summaries],
    }
    if encrypted is not None:
        item["encrypted_content"] = encrypted
    return item


def resp_message(item_id: str, text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "id": item_id,
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def resp_function_call(item_id: str, call_id: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": item_id,
        "call_id": call_id,
        "name": "lookup",
        "arguments": LOOKUP_ARGUMENTS,
        "status": "completed",
    }


def resp_events(response: Mapping[str, Any]) -> list[tuple[str | None, Any]]:
    """The typed event stream a provider sends for *response*, from ``created`` to its terminal event."""
    events: list[tuple[str | None, Any]] = []

    def emit(event_type: str, **fields: Any) -> None:
        events.append((event_type, {"type": event_type, "sequence_number": len(events), **fields}))

    in_progress = {**response, "status": "in_progress", "output": [], "usage": None}
    emit("response.created", response=in_progress)
    emit("response.in_progress", response=in_progress)
    for output_index, item in enumerate(response["output"]):
        kind = item["type"]
        item_id = item["id"]
        if kind == "message":
            emit(
                "response.output_item.added",
                output_index=output_index,
                item={**item, "status": "in_progress", "content": []},
            )
            for content_index, part in enumerate(item["content"]):
                where = {"item_id": item_id, "output_index": output_index, "content_index": content_index}
                emit("response.content_part.added", **where, part={**part, "text": "", "annotations": []})
                text = part["text"]
                for start in range(0, len(text), _SPLIT):
                    emit("response.output_text.delta", **where, delta=text[start : start + _SPLIT], logprobs=[])
                emit("response.output_text.done", **where, text=text, logprobs=[])
                emit("response.content_part.done", **where, part=part)
        elif kind == "reasoning":
            emit("response.output_item.added", output_index=output_index, item={**item, "summary": []})
            for summary_index, summary in enumerate(item["summary"]):
                where = {"item_id": item_id, "output_index": output_index, "summary_index": summary_index}
                emit("response.reasoning_summary_part.added", **where, part={"type": "summary_text", "text": ""})
                text = summary["text"]
                for start in range(0, len(text), _SPLIT):
                    emit("response.reasoning_summary_text.delta", **where, delta=text[start : start + _SPLIT])
                emit("response.reasoning_summary_text.done", **where, text=text)
                emit("response.reasoning_summary_part.done", **where, part=summary)
        elif kind == "function_call":
            emit(
                "response.output_item.added",
                output_index=output_index,
                item={**item, "arguments": "", "status": "in_progress"},
            )
            arguments = item["arguments"]
            for start in range(0, len(arguments), _SPLIT):
                emit(
                    "response.function_call_arguments.delta",
                    item_id=item_id,
                    output_index=output_index,
                    delta=arguments[start : start + _SPLIT],
                )
            emit(
                "response.function_call_arguments.done",
                item_id=item_id,
                output_index=output_index,
                arguments=arguments,
            )
        else:
            raise ValueError(f"unscripted output item type: {kind}")
        emit("response.output_item.done", output_index=output_index, item=item)
    emit("response.completed", response=response)
    return events


def resp_weather_turns(*, encrypted: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    first = resp_response(
        response_id="resp_golden_1",
        output=[
            resp_reasoning(
                "rs_golden_1", "The user wants the weather.", encrypted="enc-golden-1" if encrypted else None
            ),
            resp_message("msg_golden_1", "Let me check the weather."),
            resp_function_call("fc_golden_1", "call_lookup_1"),
        ],
        usage=RESP_USAGE_1,
    )
    second = resp_response(
        response_id="resp_golden_2",
        output=[resp_message("msg_golden_2", "It is sunny in Paris, 21 degrees.")],
    )
    return first, second


def resp_replies(responses: Sequence[Mapping[str, Any]], *, stream: bool) -> tuple[Reply, ...]:
    if stream:
        return tuple(sse_reply(resp_events(response)) for response in responses)
    return tuple(json_reply(response) for response in responses)


# ---------------------------------------------------------------------------
# Anthropic payloads
# ---------------------------------------------------------------------------

ANTH_USAGE_1 = {
    "input_tokens": 40,
    "output_tokens": 12,
    "cache_creation_input_tokens": 6,
    "cache_read_input_tokens": 8,
}
ANTH_USAGE_2 = {"input_tokens": 70, "output_tokens": 9, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


def anth_message(
    *,
    message_id: str,
    content: Sequence[Mapping[str, Any]],
    stop_reason: str = "end_turn",
    usage: Mapping[str, Any] = ANTH_USAGE_2,
) -> dict[str, Any]:
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": "golden-model-2026",
        "content": [dict(block) for block in content],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": dict(usage),
    }


def anth_events(message: Mapping[str, Any]) -> list[tuple[str | None, Any]]:
    """The event stream a provider sends for *message*: start, one block at a time, delta, stop."""
    events: list[tuple[str | None, Any]] = []

    def emit(event_type: str, **fields: Any) -> None:
        events.append((event_type, {"type": event_type, **fields}))

    usage = message["usage"]
    emit(
        "message_start", message={**message, "content": [], "stop_reason": None, "usage": {**usage, "output_tokens": 1}}
    )
    for index, block in enumerate(message["content"]):
        kind = block["type"]
        if kind == "text":
            emit("content_block_start", index=index, content_block={"type": "text", "text": ""})
            for citation in block.get("citations") or ():
                emit("content_block_delta", index=index, delta={"type": "citations_delta", "citation": citation})
            text = block["text"]
            for start in range(0, len(text), _SPLIT):
                emit(
                    "content_block_delta",
                    index=index,
                    delta={"type": "text_delta", "text": text[start : start + _SPLIT]},
                )
        elif kind == "thinking":
            emit(
                "content_block_start", index=index, content_block={"type": "thinking", "thinking": "", "signature": ""}
            )
            text = block["thinking"]
            # Omitted thinking still streams one empty delta, as the service does.
            for start in range(0, len(text) or 1, _SPLIT):
                emit(
                    "content_block_delta",
                    index=index,
                    delta={"type": "thinking_delta", "thinking": text[start : start + _SPLIT]},
                )
            emit("content_block_delta", index=index, delta={"type": "signature_delta", "signature": block["signature"]})
        elif kind == "tool_use":
            emit("content_block_start", index=index, content_block={**block, "input": {}})
            arguments = json.dumps(block["input"])
            for start in range(0, len(arguments), _SPLIT):
                emit(
                    "content_block_delta",
                    index=index,
                    delta={"type": "input_json_delta", "partial_json": arguments[start : start + _SPLIT]},
                )
        else:
            raise ValueError(f"unscripted content block type: {kind}")
        emit("content_block_stop", index=index)
    emit(
        "message_delta",
        delta={"stop_reason": message["stop_reason"], "stop_sequence": message["stop_sequence"]},
        usage={"output_tokens": usage["output_tokens"]},
    )
    emit("message_stop")
    return events


def anth_weather_turns() -> tuple[dict[str, Any], dict[str, Any]]:
    first = anth_message(
        message_id="msg_golden_1",
        content=[
            {"type": "thinking", "thinking": "The user wants the weather.", "signature": "sig-golden-1"},
            {"type": "text", "text": "Let me check the weather."},
            {"type": "tool_use", "id": "toolu_golden_1", "name": "lookup", "input": {"city": "Paris"}},
        ],
        stop_reason="tool_use",
        usage=ANTH_USAGE_1,
    )
    second = anth_message(
        message_id="msg_golden_2", content=[{"type": "text", "text": "It is sunny in Paris, 21 degrees."}]
    )
    return first, second


def anth_text(text: str, *, message_id: str) -> dict[str, Any]:
    return anth_message(message_id=message_id, content=[{"type": "text", "text": text}])


def anth_replies(messages: Sequence[Mapping[str, Any]], *, stream: bool) -> tuple[Reply, ...]:
    if stream:
        return tuple(sse_reply(anth_events(message)) for message in messages)
    return tuple(json_reply(message) for message in messages)

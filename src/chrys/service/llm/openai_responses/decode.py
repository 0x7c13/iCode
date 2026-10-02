# Copyright (c) Microsoft. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Microsoft Agent Framework (MIT License; see NOTICE).

"""Responses read back as chat responses.

:func:`decode_response` reads a blocking response. The stream (:mod:`.stream`)
reuses the pieces that describe whole items or the response itself: reasoning
items, client-executed tool calls, usage, the conversation handle, the
continuation token and the finish reason.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, cast

from openai.types.responses.parsed_response import ParsedResponse

from chrys.foundation.hosted_tools import OPENAI_HOSTED_WIRE_ITEM_KEY
from chrys.kernel import (
    OPENAI_OUTPUT_MESSAGE_ENVELOPE_KEY,
    Annotation,
    ChatResponse,
    Content,
    ContinuationToken,
    Message,
    TextSpanRegion,
    UsageDetails,
)
from chrys.service.llm.chat_completions.decode import add_deepseek_cache_usage
from chrys.service.profiles.models.options import effective_store_option

from .hosted import decode_hosted_item, to_payload

if TYPE_CHECKING:
    from openai.types.responses.response import Response
    from openai.types.responses.response_usage import ResponseUsage
    from pydantic import BaseModel

    from .client import ResponsesVariant

logger = logging.getLogger(__name__)

# The output-message fields its text replays with, so a replayed message
# keeps the identity it was produced under.
ENVELOPE_FIELDS = ("id", "status", "phase")
# Statuses of a response that is still running, so a token can resume it.
RUNNING_STATUSES = ("in_progress", "queued")


class OpenAIContinuationToken(ContinuationToken):
    """Where to pick up a background response: the id to retrieve."""

    response_id: str


def decode_response(
    response: Response | ParsedResponse[BaseModel], options: Mapping[str, Any], *, variant: ResponsesVariant
) -> ChatResponse:
    """A blocking response as one assistant message plus response metadata."""
    # ParsedResponse's type argument is erased at runtime; requests only ever
    # ask the SDK to parse into Pydantic models.
    parsed = cast("BaseModel | None", response.output_parsed) if isinstance(response, ParsedResponse) else None
    # Log probabilities of the text parts collect here.
    metadata: dict[str, Any] = dict(response.metadata or {})
    try:
        output = response.output
    except AttributeError:
        output = []
    contents: list[Content] = []
    for item in output:
        contents.extend(_decode_item(item, metadata, variant.hosted_provider))

    fields: dict[str, Any] = {
        "response_id": response.id,
        "created_at": timestamp(response.created_at),
        "messages": Message(role="assistant", contents=contents),
        "model": response.model,
        "additional_properties": metadata,
        "raw_representation": response,
    }
    store = effective_store_option(options)
    if conversation_id := conversation_handle(response, store=store, variant=variant):
        fields["conversation_id"] = conversation_id
    if response.usage and (usage := decode_usage(response.usage, variant=variant)):
        fields["usage_details"] = usage
    if parsed:
        fields["value"] = parsed
    elif response_format := options.get("response_format"):
        fields["response_format"] = response_format
    if response.status in RUNNING_STATUSES and (token := continuation_token(response.id, store=store, variant=variant)):
        fields["continuation_token"] = token
    if reason := finish_reason(response):
        fields["finish_reason"] = reason
    return ChatResponse(**fields)


def _decode_item(item: Any, metadata: dict[str, Any], provider: str) -> list[Content]:
    match item.type:
        case "message":
            return _message_parts(item, metadata)
        case "reasoning":
            return decode_reasoning_item(item, streamed=False)
        case "function_call":
            return [
                Content.from_function_call(
                    call_id=item.call_id,
                    name=item.name,
                    arguments=item.arguments,
                    additional_properties={"fc_id": item.id, "status": item.status},
                    raw_representation=item,
                )
            ]
        case "custom_tool_call":
            return [decode_client_tool_call(item, name=item.name, arguments=item.input)]
        case "apply_patch_call":
            return [decode_client_tool_call(item, name="apply_patch", arguments=getattr(item, "operation", None))]
        case _:
            return decode_hosted_item(item, provider) or []


def _message_parts(item: Any, metadata: dict[str, Any]) -> list[Content]:
    envelope = output_message_envelope(item)
    properties = {OPENAI_OUTPUT_MESSAGE_ENVELOPE_KEY: envelope} if envelope else None
    parts: list[Content] = []
    for part in item.content:
        if part.type == "output_text":
            text = Content.from_text(text=part.text, raw_representation=part, additional_properties=properties)
            metadata.update(logprobs_metadata(part))
            if part.annotations:
                text.annotations = [
                    annotation for annotation in map(_citation, part.annotations) if annotation is not None
                ]
            parts.append(text)
        elif part.type == "refusal":
            parts.append(
                Content.from_text(text=part.refusal, raw_representation=part, additional_properties=properties)
            )
    return parts


def _citation(annotation: Any) -> Annotation | None:
    """A citation annotation of a blocking response's text part.

    Streamed parts describe the same citations in another shape
    (:func:`.stream.streamed_citation`); the two are kept as they are.
    """
    kind = annotation.type
    if kind == "url_citation":
        return Annotation(
            type="citation",
            title=annotation.title,
            url=annotation.url,
            annotated_regions=[_span(annotation)],
            raw_representation=annotation,
        )
    if kind == "container_file_citation":
        return Annotation(
            type="citation",
            file_id=annotation.file_id,
            url=annotation.filename,
            additional_properties={"container_id": annotation.container_id},
            annotated_regions=[_span(annotation)],
            raw_representation=annotation,
        )
    if kind == "file_citation":
        return Annotation(
            type="citation",
            url=annotation.filename,
            file_id=annotation.file_id,
            raw_representation=annotation,
            additional_properties={"index": annotation.index},
        )
    if kind == "file_path":
        return Annotation(
            type="citation",
            file_id=annotation.file_id,
            additional_properties={"index": annotation.index},
            raw_representation=annotation,
        )
    logger.debug("Unparsed annotation type: %s", kind)
    return None


def _span(annotation: Any) -> TextSpanRegion:
    return TextSpanRegion(type="text_span", start_index=annotation.start_index, end_index=annotation.end_index)


def output_message_envelope(item: Any) -> dict[str, str]:
    """The envelope fields an output message carries."""
    return {name: value for name in ENVELOPE_FIELDS if isinstance(value := getattr(item, name, None), str) and value}


def decode_reasoning_item(item: Any, *, streamed: bool) -> list[Content]:
    """A reasoning item's text parts, then its summaries.

    An item with neither still yields one empty content, so its encrypted
    payload and its place among the outputs survive. Two differences between
    the blocking and the streamed shape stay as they are: a blocking item
    puts its payload on the first content only and pairs each text part with
    the summary at its index; a streamed one puts the payload on every
    content and pairs nothing.
    """
    item_id = (getattr(item, "id", None) or None) if streamed else item.id
    payload = getattr(item, "encrypted_content", None)
    summaries = getattr(item, "summary", None) or []
    contents: list[Content] = []

    def add(text: str, raw: Any, properties: dict[str, Any] | None = None) -> None:
        contents.append(
            Content.from_text_reasoning(
                id=item_id,
                text=text,
                protected_data=payload if streamed or not contents else None,
                raw_representation=raw,
                additional_properties=properties,
            )
        )

    for index, part in enumerate(getattr(item, "content", None) or []):
        properties: dict[str, Any] = {"reasoning_text": True}
        if not streamed and index < len(summaries):
            properties["summary"] = summaries[index]
        add(part.text, part, properties)
    for summary in summaries:
        add(summary.text, summary)
    if not contents:
        add("", item)
    return contents


def decode_client_tool_call(item: Any, *, name: str, arguments: Any) -> Content:
    """A custom or apply-patch call, which the client would have to run.

    Nothing here runs it, so it is informational: history sends the wire
    item back and replay answers an apply-patch call with a failure.
    """
    item_type = str(getattr(item, "type", ""))
    properties: dict[str, Any] = {"item_type": item_type, OPENAI_HOSTED_WIRE_ITEM_KEY: to_payload(item)}
    if item_type == "custom_tool_call":
        call_id = getattr(item, "call_id", "") or ""
        if item_id := getattr(item, "id", None):
            properties["item_id"] = item_id
        if namespace := getattr(item, "namespace", None):
            properties["namespace"] = namespace
    else:
        item_id = getattr(item, "id", "") or ""
        call_id = getattr(item, "call_id", None) or item_id
        properties["item_id"] = item_id
        properties["status"] = getattr(item, "status", None)
        properties["execution"] = to_payload(getattr(item, "execution", None))
        if created_by := getattr(item, "created_by", None):
            properties["created_by"] = created_by
    return Content.from_function_call(
        call_id=call_id,
        name=name,
        arguments=to_payload(arguments),
        informational_only=True,
        additional_properties=properties,
        raw_representation=item,
    )


def decode_usage(usage: ResponseUsage, *, variant: ResponsesVariant) -> UsageDetails:
    """Token counts, with cache reads and writes and reasoning tokens when reported (zero included)."""
    details = UsageDetails(
        input_token_count=usage.input_tokens,
        output_token_count=usage.output_tokens,
        total_token_count=usage.total_tokens,
    )
    if inputs := usage.input_tokens_details:
        if (cached := getattr(inputs, "cached_tokens", None)) is not None:
            details["openai.cached_input_tokens"] = cached  # type: ignore[typeddict-unknown-key]
            details["cache_read_input_token_count"] = cached
        # Not in the SDK's model (kept as an extra field): billed explicit
        # prompt caching reports it.
        if (written := getattr(inputs, "cache_write_tokens", None)) is not None:
            details["openai.cache_write_tokens"] = written  # type: ignore[typeddict-unknown-key]
            details["cache_creation_input_token_count"] = written
    outputs = usage.output_tokens_details
    if outputs and (reasoning := getattr(outputs, "reasoning_tokens", None)) is not None:
        details["openai.reasoning_tokens"] = reasoning  # type: ignore[typeddict-unknown-key]
        details["reasoning_output_token_count"] = reasoning
    if variant.reports_prompt_cache_hits:
        add_deepseek_cache_usage(details, usage)
    return details


def logprobs_metadata(source: Any) -> dict[str, Any]:
    """The response metadata a text part or delta adds: its log probabilities."""
    logprobs = getattr(source, "logprobs", None)
    return {"logprobs": logprobs} if logprobs else {}


def conversation_handle(response: Any, *, store: Any, variant: ResponsesVariant) -> str | None:
    """The id the next request continues from: the conversation, else the response.

    None when the service keeps nothing, as when the request opted out of
    storing it.
    """
    if variant.stateless or store is False:
        return None
    if response.conversation and response.conversation.id:
        return response.conversation.id
    return response.id


def continuation_token(response_id: str, *, store: Any, variant: ResponsesVariant) -> OpenAIContinuationToken | None:
    """A token to resume an unfinished response, if it can be retrieved.

    An unstored response cannot: retrieving it fails, so a token would turn
    every reconnect into an error instead of a new request.
    """
    if variant.stateless or store is False:
        return None
    return OpenAIContinuationToken(response_id=response_id)


def finish_reason(response: Any) -> Literal["length"] | None:
    """``length`` for a response cut off at its output cap.

    Truncation handling then treats it like the other protocols' cutoff, not
    as an empty response to retry.
    """
    details = getattr(response, "incomplete_details", None)
    if response.status == "incomplete" and getattr(details, "reason", None) == "max_output_tokens":
        return "length"
    return None


def timestamp(created_at: float) -> str:
    """A response's creation time in the chat response's UTC format."""
    return datetime.fromtimestamp(created_at, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

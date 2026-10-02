# Copyright (c) Microsoft. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Microsoft Agent Framework (MIT License; see NOTICE).

"""Decode Messages API responses: content blocks, citations, usage and stop reasons.

The same block decoder reads a blocking response's content and a stream's
block starts and deltas; :mod:`.stream` adds what only the stream knows.
Server-tool and MCP blocks are decoded in :mod:`.server_tools`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from chrys.kernel import (
    Annotation,
    ChatResponse,
    Content,
    FinishReason,
    FinishReasonLiteral,
    Message,
    TextSpanRegion,
    UsageDetails,
)
from chrys.kernel._content import _ANTHROPIC_REDACTED_THINKING_KEY

from . import server_tools

if TYPE_CHECKING:
    from anthropic.types.beta import BetaMessage, BetaMessageDeltaUsage, BetaUsage

logger = logging.getLogger(__name__)

_FINISH_REASONS: Final[Mapping[str, FinishReasonLiteral]] = {
    "end_turn": "stop",
    "pause_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    # Cut off, without the window-filled marker Chat Completions sets
    # (``CONTEXT_WINDOW_FILLED_KEY``): a LAST_WORDS note cut off here is kept.
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}

# Server tools that run in the code container; tools they call report results
# nested in it.
_CONTAINER_TOOLS: Final = frozenset({"code_execution", "bash_code_execution", "text_editor_code_execution"})


def decode_message(message: BetaMessage, *, response_format: Any) -> ChatResponse:
    """A blocking response as one assistant message."""
    usage = decode_usage(message.usage)
    if usage is not None and (estimate := blocking_context_estimate(message.usage, message.content)) is not None:
        usage["context_input_token_estimate"] = estimate
    return ChatResponse(
        response_id=message.id,
        messages=[Message(role="assistant", contents=decode_blocks(message.content), raw_representation=message)],
        usage_details=usage,
        model=message.model,
        finish_reason=decode_stop_reason(message.stop_reason),
        response_format=response_format,
        raw_representation=message,
    )


def decode_stop_reason(stop_reason: str | None) -> FinishReason | None:
    """The finish reason of *stop_reason*; one Chrys does not map is kept as it is."""
    if not stop_reason:
        return None
    return FinishReason(_FINISH_REASONS.get(stop_reason, stop_reason))


def decode_blocks(blocks: Sequence[Any]) -> list[Content]:
    """Decode content blocks, or the deltas of streamed ones; unsupported blocks are skipped."""
    return [content for block in blocks if (content := _decode_block(block)) is not None]


def _decode_block(block: Any) -> Content | None:
    match block.type:
        case "text" | "text_delta":
            return Content.from_text(text=block.text, raw_representation=block, annotations=decode_citations(block))
        case "thinking" | "thinking_delta":
            # A thinking delta carries no signature; a later signature delta does.
            signature = getattr(block, "signature", None)
            return Content.from_text_reasoning(text=block.thinking, protected_data=signature, raw_representation=block)
        case "signature_delta":
            return Content.from_text_reasoning(text=None, protected_data=block.signature, raw_representation=block)
        case "redacted_thinking":
            return Content.from_text_reasoning(
                text=None,
                protected_data=block.data,
                additional_properties={_ANTHROPIC_REDACTED_THINKING_KEY: True},
                raw_representation=block,
            )
        case "tool_use":
            return Content.from_function_call(
                call_id=block.id, name=block.name, arguments=block.input, raw_representation=block
            )
        case "server_tool_use":
            return server_tools.decode_server_tool_use(block)
        case "mcp_tool_use":
            return server_tools.decode_mcp_tool_use(block)
        case "mcp_tool_result":
            return server_tools.decode_mcp_tool_result(block, output=_mcp_output(block))
        case "input_json_delta":
            # Only the stream event knows which block the input belongs to.
            logger.debug("Ignoring Anthropic input_json_delta without stream event context")
            return None
        case kind if kind != "tool_result" and kind.endswith("_tool_result"):
            return server_tools.decode_server_tool_result(block)
        case kind:
            logger.debug("Ignoring unsupported content type: %s", kind)
            return None


def _mcp_output(block: Any) -> list[Content] | None:
    output = block.content
    if not output:
        return None
    if isinstance(output, list):
        return decode_blocks(output)
    if isinstance(output, (str, bytes)):
        return [Content.from_text(text=str(output), raw_representation=block)]
    return decode_blocks([output])


@dataclass(frozen=True, slots=True)
class _CitationShape:
    """Where one citation type keeps the fields an annotation reads."""

    title: str
    span: tuple[str, str] | None
    """The start and end attributes of the cited region."""
    url: str | None = None
    names_file: bool = False
    """The citation may name the uploaded file it cites."""


_CITATION_SHAPES: Final[Mapping[str, _CitationShape]] = {
    "char_location": _CitationShape("document_title", ("start_char_index", "end_char_index"), names_file=True),
    "page_location": _CitationShape("document_title", ("start_page_number", "end_page_number"), names_file=True),
    "content_block_location": _CitationShape(
        "document_title", ("start_block_index", "end_block_index"), names_file=True
    ),
    "web_search_result_location": _CitationShape("title", None, url="url"),
    "search_result_location": _CitationShape("title", ("start_block_index", "end_block_index"), url="source"),
}


def decode_citations(block: Any) -> list[Annotation] | None:
    """The citations of a text block as annotations; None when it cites nothing."""
    citations = getattr(block, "citations", None)
    if not citations:
        return None
    return [_decode_citation(citation) for citation in citations] or None


def _decode_citation(citation: Any) -> Annotation:
    annotation = Annotation(type="citation", raw_representation=citation)
    shape = _CITATION_SHAPES.get(citation.type)
    if shape is None:
        logger.debug("Unknown citation type encountered: %s", citation.type)
        return annotation
    if (title := getattr(citation, shape.title, None)) is not None:
        annotation["title"] = title
    annotation["snippet"] = citation.cited_text
    if shape.url is not None:
        annotation["url"] = getattr(citation, shape.url)
    if shape.names_file and citation.file_id:
        annotation["file_id"] = citation.file_id
    if shape.span is not None:
        start, end = shape.span
        annotation["annotated_regions"] = [
            TextSpanRegion(type="text_span", start_index=getattr(citation, start), end_index=getattr(citation, end))
        ]
    return annotation


def token_count(value: Any) -> int | None:
    """*value* when it is a token count: a non-negative int that is not a bool."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def decode_usage(usage: BetaUsage | BetaMessageDeltaUsage | None) -> UsageDetails | None:
    """Usage details of a response, or of a stream's start or message delta.

    Cache creation and reads are reported under the shared keys and under
    ``anthropic.``-prefixed ones. Both are also added to ``input_token_count``:
    cached prompt tokens still occupy the context window.
    """
    if not usage:
        return None
    details = UsageDetails(output_token_count=usage.output_tokens)
    if usage.input_tokens is not None:
        details["input_token_count"] = usage.input_tokens
    created, read = usage.cache_creation_input_tokens, usage.cache_read_input_tokens
    if created is not None:
        details["anthropic.cache_creation_input_tokens"] = created  # type: ignore[typeddict-unknown-key]
        details["cache_creation_input_token_count"] = created
    if read is not None:
        details["anthropic.cache_read_input_tokens"] = read  # type: ignore[typeddict-unknown-key]
        details["cache_read_input_token_count"] = read
    if (floor := context_floor(usage)) is not None:
        details["context_input_token_floor"] = floor
    if (cached := int(created or 0) + int(read or 0)) > 0 and "input_token_count" in details:
        details["input_token_count"] = int(details["input_token_count"] or 0) + cached
    return details


def context_floor(usage: BetaUsage | BetaMessageDeltaUsage | None) -> int | None:
    """Context tokens *usage* proves: uncached input plus cache creation.

    Cache reads are left out because a hosted-tool loop reports them summed
    over its sampling passes.
    """
    if usage is None:
        return None
    uncached, created = token_count(usage.input_tokens), token_count(usage.cache_creation_input_tokens)
    if uncached is None or created is None:
        return None
    return created + uncached


def stream_context_input(usage: BetaMessageDeltaUsage | None, *, first_cache_read: int | None) -> int | None:
    """Context occupancy at the end of a streamed response.

    The final usage of a hosted-tool loop sums cache reads over every sampling
    pass; of those, only the first request's read (*first_cache_read*, from
    the message start) is still in the context.
    """
    if first_cache_read is None or (floor := context_floor(usage)) is None:
        return None
    return first_cache_read + floor


def blocking_context_estimate(usage: BetaUsage | None, blocks: Sequence[Any]) -> int | None:
    """Estimate the context occupancy of a blocking response.

    A blocking response reports only the cache reads summed over its sampling
    passes. Their mean estimates the cached prefix still held, and counts
    only when it exceeds the cache creation.
    """
    floor = context_floor(usage)
    if usage is None or floor is None:
        return None
    read, created = token_count(usage.cache_read_input_tokens), token_count(usage.cache_creation_input_tokens)
    passes = _sampling_passes(blocks)
    if read is None or created is None or passes is None:
        return floor
    mean_read = -(-read // passes)
    return floor + mean_read if mean_read > created else floor


def _sampling_passes(blocks: Sequence[Any]) -> int | None:
    """Sampling passes of a blocking response; None when no server tool returned.

    Every top-level server-tool result starts another pass. A result for a
    tool called inside a running container tool is part of that tool's pass.
    """
    running_containers: list[str] = []
    results = 0
    for block in blocks:
        kind = str(getattr(block, "type", ""))
        if kind == "server_tool_use":
            call_id = getattr(block, "id", None)
            if getattr(block, "name", None) in _CONTAINER_TOOLS and isinstance(call_id, str):
                running_containers.append(call_id)
        elif kind != "tool_result" and kind.endswith("_tool_result"):
            answered = getattr(block, "tool_use_id", None)
            if isinstance(answered, str) and answered in running_containers:
                position = running_containers.index(answered)
                del running_containers[position]
                if position == 0:
                    results += 1
            elif not running_containers:
                results += 1
    return results + 1 if results else None

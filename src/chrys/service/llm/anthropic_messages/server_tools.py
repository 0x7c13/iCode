# Copyright (c) Microsoft. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Microsoft Agent Framework (MIT License; see NOTICE).

"""Decode server-tool and MCP blocks as hosted-tool contents.

Each content keeps the block it was decoded from, serialized under
``ANTHROPIC_HOSTED_WIRE_BLOCK_KEY``, and history sends that block back
unchanged (:mod:`.history`). A failed result also carries its error payload.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

from anthropic.types.beta.beta_bash_code_execution_tool_result_error import BetaBashCodeExecutionToolResultError
from anthropic.types.beta.beta_code_execution_result_block import BetaCodeExecutionResultBlock
from anthropic.types.beta.beta_code_execution_tool_result_error import BetaCodeExecutionToolResultError
from anthropic.types.beta.beta_encrypted_code_execution_result_block import BetaEncryptedCodeExecutionResultBlock
from pydantic import BaseModel

from chrys.foundation.hosted_tools import (
    ANTHROPIC_HOSTED_WIRE_BLOCK_KEY,
    HostedRetrySafety,
    HostedToolFamily,
    HostedToolPhase,
)
from chrys.kernel import Annotation, Content, TextSpanRegion


class _HostedFields(TypedDict):
    """Provider facts every hosted content decoded from one block shares."""

    hosted_provider: str
    provider_item_type: str
    provider_item_id: str
    provider_phase: str
    provider_status: str
    retry_safety: str
    additional_properties: dict[str, Any]
    raw_representation: Any


def _serialize(value: Any) -> Any:
    """*value* as JSON-safe data; SDK models leave out unset and None fields."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True, exclude_unset=True)
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_serialize(item) for item in value]
    if hasattr(value, "__dict__"):
        public = {key: item for key, item in vars(value).items() if not key.startswith("_") and item is not None}
        return {str(key): _serialize(item) for key, item in public.items()}
    return value


def _hosted_fields(
    block: Any, item_id: str, *, phase: str, status: str, safety: str, error: Any = None
) -> _HostedFields:
    # The persisted replay block, plus the error payload of a failed result.
    properties: dict[str, Any] = {ANTHROPIC_HOSTED_WIRE_BLOCK_KEY: _serialize(block)}
    if error is not None:
        properties["is_error"] = True
        properties["error"] = _serialize(error)
    return _HostedFields(
        hosted_provider="anthropic",
        provider_item_type=block.type,
        provider_item_id=item_id,
        provider_phase=phase,
        provider_status=status,
        retry_safety=safety,
        additional_properties=properties,
        raw_representation=block,
    )


def _failure(payload: Any) -> Any:
    """*payload* when it is an error block, else None."""
    return payload if str(getattr(payload, "type", "")).endswith(("_error", "_error_block")) else None


def _shell_commands(arguments: Any) -> list[str]:
    command = arguments.get("command") if isinstance(arguments, Mapping) else arguments
    if isinstance(command, str):
        return [command]
    if isinstance(command, Sequence) and not isinstance(command, (bytes, bytearray)):
        return [str(part) for part in command]
    return []


def _code_source(arguments: Any) -> str:
    code = arguments.get("code") if isinstance(arguments, Mapping) else None
    return code if isinstance(code, str) else json.dumps(arguments, ensure_ascii=False)


_TOOL_SEARCH_NAMES = frozenset({"tool_search", "tool_search_tool_regex", "tool_search_tool_bm25"})


def decode_server_tool_use(block: Any) -> Content:
    """A ``server_tool_use`` block as the hosted call of its tool family."""
    name = block.name or ""
    arguments = _serialize(block.input)

    def fields(safety: str) -> _HostedFields:
        return _hosted_fields(block, block.id, phase=HostedToolPhase.START, status="running", safety=safety)

    if name == "web_search":
        return Content.from_search_tool_call(
            block.id, tool_name=name, arguments=arguments, status="running", **fields(HostedRetrySafety.READ_ONLY)
        )
    if name == "web_fetch":
        return Content.from_search_tool_call(
            block.id,
            tool_name=name,
            arguments=arguments,
            status="running",
            hosted_family=HostedToolFamily.FETCH,
            **fields(HostedRetrySafety.READ_ONLY),
        )
    if name == "code_execution":
        return Content.from_code_interpreter_tool_call(
            call_id=block.id,
            inputs=[Content.from_text(_code_source(arguments))],
            **fields(HostedRetrySafety.SANDBOXED),
        )
    if name == "bash_code_execution":
        return Content.from_shell_tool_call(
            call_id=block.id,
            commands=_shell_commands(arguments),
            status="running",
            **fields(HostedRetrySafety.SIDE_EFFECTFUL),
        )
    if name == "text_editor_code_execution":
        family, safety = HostedToolFamily.FILE_OPERATION, HostedRetrySafety.SIDE_EFFECTFUL
    elif name in _TOOL_SEARCH_NAMES:
        # The regex and BM25 variants are one tool to Chrys.
        name, family, safety = "tool_search", HostedToolFamily.TOOL_DISCOVERY, HostedRetrySafety.READ_ONLY
    else:
        name, family, safety = name or "anthropic_server_tool", HostedToolFamily.GENERIC, HostedRetrySafety.UNKNOWN
    return Content.from_hosted_tool_call(
        block.id, tool_name=name, arguments=arguments, status="running", hosted_family=family, **fields(safety)
    )


def decode_mcp_tool_use(block: Any) -> Content:
    """An ``mcp_tool_use`` block as a remote MCP call."""
    return Content.from_mcp_server_tool_call(
        block.id,
        block.name,
        server_name=block.server_name,
        arguments=block.input,
        **_hosted_fields(
            block, block.id, phase=HostedToolPhase.START, status="running", safety=HostedRetrySafety.SIDE_EFFECTFUL
        ),
    )


def decode_mcp_tool_result(block: Any, *, output: list[Content] | None) -> Content:
    """An ``mcp_tool_result`` block whose content decoded to *output*."""
    failed = bool(getattr(block, "is_error", False))
    return Content.from_mcp_server_tool_result(
        block.tool_use_id,
        output=output,
        **_hosted_fields(
            block,
            block.tool_use_id,
            phase=HostedToolPhase.TERMINAL,
            status="failed" if failed else "completed",
            safety=HostedRetrySafety.SIDE_EFFECTFUL,
            error=block.content if failed else None,
        ),
    )


def decode_server_tool_result(block: Any) -> Content:
    """A server tool's ``*_tool_result`` block as the hosted result of its family."""
    payload = block.content
    error = _failure(payload)
    status = "completed" if error is None else "failed"

    def fields(safety: str) -> _HostedFields:
        return _hosted_fields(
            block, block.tool_use_id, phase=HostedToolPhase.TERMINAL, status=status, safety=safety, error=error
        )

    match block.type:
        case "web_search_tool_result" | "web_fetch_tool_result":
            searched = block.type == "web_search_tool_result"
            return Content.from_search_tool_result(
                block.tool_use_id,
                tool_name="web_search" if searched else "web_fetch",
                result=_serialize(payload),
                status=status,
                hosted_family=HostedToolFamily.SEARCH if searched else HostedToolFamily.FETCH,
                **fields(HostedRetrySafety.READ_ONLY),
            )
        case "code_execution_tool_result":
            return Content.from_code_interpreter_tool_result(
                call_id=block.tool_use_id, outputs=_code_outputs(payload), **fields(HostedRetrySafety.SANDBOXED)
            )
        case "bash_code_execution_tool_result":
            return Content.from_shell_tool_result(
                call_id=block.tool_use_id, outputs=_bash_outputs(payload), **fields(HostedRetrySafety.SIDE_EFFECTFUL)
            )
        case "text_editor_code_execution_tool_result":
            return Content.from_hosted_tool_result(
                block.tool_use_id,
                tool_name="text_editor_code_execution",
                result=_serialize(payload),
                items=_text_editor_outputs(payload),
                status=status,
                hosted_family=HostedToolFamily.FILE_OPERATION,
                **fields(HostedRetrySafety.SIDE_EFFECTFUL),
            )
        case "tool_search_tool_result":
            family, safety, tool_name = HostedToolFamily.TOOL_DISCOVERY, HostedRetrySafety.READ_ONLY, "tool_search"
        case _:
            family, safety = HostedToolFamily.GENERIC, HostedRetrySafety.UNKNOWN
            tool_name = block.type.removesuffix("_tool_result")
    return Content.from_hosted_tool_result(
        block.tool_use_id,
        tool_name=tool_name,
        result=_serialize(payload),
        status=status,
        hosted_family=family,
        **fields(safety),
    )


def _code_outputs(payload: Any) -> list[Content]:
    if not payload:
        return []
    if isinstance(payload, BetaCodeExecutionToolResultError):
        return [Content.from_error(message=payload.error_code, raw_representation=payload)]
    outputs: list[Content] = []
    if isinstance(payload, BetaCodeExecutionResultBlock) and payload.stdout:
        outputs.append(Content.from_text(text=payload.stdout, raw_representation=payload))
    if isinstance(payload, BetaEncryptedCodeExecutionResultBlock) and payload.encrypted_stdout:
        outputs.append(Content.from_text(text=payload.encrypted_stdout, raw_representation=payload))
    if payload.stderr:
        outputs.append(Content.from_error(message=payload.stderr, raw_representation=payload))
    outputs.extend(Content.from_hosted_file(file_id=file.file_id, raw_representation=file) for file in payload.content)
    return outputs


def _bash_outputs(payload: Any) -> list[Content]:
    if not payload:
        return []
    if isinstance(payload, BetaBashCodeExecutionToolResultError):
        return [
            Content.from_shell_command_output(
                stderr=payload.error_code,
                timed_out=payload.error_code == "execution_time_exceeded",
                raw_representation=payload,
            )
        ]
    run = Content.from_shell_command_output(
        stdout=payload.stdout or None,
        stderr=payload.stderr or None,
        exit_code=int(payload.return_code),
        timed_out=False,
        raw_representation=payload,
    )
    return [run, *(Content.from_hosted_file(file_id=file.file_id, raw_representation=file) for file in payload.content)]


def _line_region(start: int | None, count: int | None) -> TextSpanRegion | None:
    """The span of *count* lines from line *start*; None unless both are known."""
    if start is None or count is None:
        return None
    return TextSpanRegion(type="text_span", start_index=start, end_index=start + count)


def _text_editor_outputs(payload: Any) -> list[Content]:
    match payload.type:
        case "text_editor_code_execution_tool_result_error":
            message = payload.error_message or payload.error_code
            return [Content.from_error(message=message, error_code=payload.error_code, raw_representation=payload)]
        case "text_editor_code_execution_view_result":
            shown = _line_region(payload.start_line, payload.num_lines)
            annotations = (
                [Annotation(type="citation", raw_representation=payload, annotated_regions=[shown])]
                if shown is not None
                else None
            )
            return [Content.from_text(text=payload.content, annotations=annotations, raw_representation=payload)]
        case "text_editor_code_execution_str_replace_result":
            new_text = "\n".join(payload.lines) if payload.lines else None
            citations: list[Annotation] = []
            if (replaced := _line_region(payload.old_start, payload.old_lines)) is not None:
                citations.append(Annotation(type="citation", raw_representation=payload, annotated_regions=[replaced]))
            if (inserted := _line_region(payload.new_start, payload.new_lines)) is not None:
                # Only the inserted lines are quoted, as a snippet that may be None.
                citations.append(
                    Annotation(
                        type="citation", raw_representation=payload, snippet=new_text, annotated_regions=[inserted]
                    )
                )
            return [Content.from_text(text=new_text or "", annotations=citations or None, raw_representation=payload)]
        case "text_editor_code_execution_create_result":
            return [Content.from_text(text=f"File update: {payload.is_file_update}", raw_representation=payload)]
        case _:
            return []


def apply_streamed_input(call: Content, arguments: Any) -> None:
    """Refresh a streamed hosted *call*, and its replay block, from its input so far."""
    if call.type == "code_interpreter_tool_call":
        call.inputs = [Content.from_text(_code_source(arguments))]
    elif call.type == "shell_tool_call":
        call.commands = _shell_commands(arguments)
    else:
        call.arguments = arguments
    replay = call.additional_properties.get(ANTHROPIC_HOSTED_WIRE_BLOCK_KEY)
    if isinstance(replay, dict):
        replay["input"] = arguments

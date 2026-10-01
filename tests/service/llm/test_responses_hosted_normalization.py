# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Fixture-driven hosted-tool normalization tests for the Responses dialect."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from chrys.foundation.hosted_tools import OPENAI_HOSTED_WIRE_ITEM_KEY
from chrys.kernel import ChatResponse, Content, Message
from chrys.kernel.compaction import (
    GROUP_ANNOTATION_KEY,
    GROUP_HAS_REASONING_KEY,
    GROUP_ID_KEY,
    GROUP_INDEX_KEY,
    GROUP_KIND_KEY,
)
from chrys.service.llm.openai_responses.client import OPENAI_RESPONSES
from chrys.service.llm.openai_responses.decode import decode_response
from chrys.service.llm.openai_responses.history import answer_apply_patch_calls
from chrys.service.llm.openai_responses.hosted import OPENAI_HOSTED_REPLAY_SHADOW_KEY
from chrys.service.llm.openai_responses.replay import encode_input
from chrys.service.llm.openai_responses.stream import StreamState


def _response(output: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(
        metadata=None,
        output=output,
        id="resp_1",
        created_at=0,
        model="gpt-test",
        conversation=None,
        usage=None,
        status="completed",
        incomplete_details=None,
    )


def _start_snapshot(item: SimpleNamespace) -> SimpleNamespace:
    values = dict(vars(item))
    if "status" in values:
        values["status"] = "in_progress"
    match item.type:
        case "web_search_call":
            values["action"] = None
        case "file_search_call":
            values["results"] = None
        case "mcp_call":
            values["output"] = None
            values["error"] = None
        case "code_interpreter_call":
            values["code"] = None
            values["outputs"] = None
        case "image_generation_call":
            values["result"] = None
        case _:
            if "output" in values:
                values["output"] = None
    return SimpleNamespace(**values)


def _stream_parse(
    items: list[SimpleNamespace],
    *,
    options: dict[str, Any] | None = None,
) -> ChatResponse:
    updates = []
    state = StreamState(options or {}, model="gpt-test", variant=OPENAI_RESPONSES)
    for output_index, item in enumerate(items):
        added = SimpleNamespace(
            type="response.output_item.added",
            item=_start_snapshot(item),
            output_index=output_index,
        )
        updates.append(state.update_for(added))
        if item.type == "web_search_call":
            for status in ("searching", "completed"):
                updates.append(
                    state.update_for(
                        SimpleNamespace(
                            type=f"response.web_search_call.{status}",
                            output_index=output_index,
                        )
                    )
                )
        elif item.type == "code_interpreter_call" and item.code:
            midpoint = max(1, len(item.code) // 2)
            for sequence_number, delta in enumerate((item.code[:midpoint], item.code[midpoint:])):
                updates.append(
                    state.update_for(
                        SimpleNamespace(
                            type="response.code_interpreter_call_code.delta",
                            item_id=item.id,
                            output_index=output_index,
                            sequence_number=sequence_number,
                            delta=delta,
                        )
                    )
                )
            updates.append(
                state.update_for(
                    SimpleNamespace(
                        type="response.code_interpreter_call_code.done",
                        item_id=item.id,
                        output_index=output_index,
                        sequence_number=2,
                        code=item.code,
                    )
                )
            )
        elif item.type == "image_generation_call" and item.result:
            for partial_index, partial in enumerate(("aW1nLTE=", "aW1nLTI=")):
                updates.append(
                    state.update_for(
                        SimpleNamespace(
                            type="response.image_generation_call.partial_image",
                            item_id=item.id,
                            output_index=output_index,
                            partial_image_index=partial_index,
                            partial_image_b64=partial,
                        )
                    )
                )
        updates.append(
            state.update_for(SimpleNamespace(type="response.output_item.done", item=item, output_index=output_index))
        )
    return ChatResponse.from_updates(updates)


def _contents(response: ChatResponse) -> list[dict[str, Any]]:
    return [content.to_dict() for content in response.messages[0].contents]


def _assert_blocking_streaming_parity(items: list[SimpleNamespace]) -> ChatResponse:
    blocking = decode_response(_response(items), {"store": False}, variant=OPENAI_RESPONSES)
    streaming = _stream_parse(items)
    assert _contents(streaming) == _contents(blocking)
    return blocking


@pytest.mark.parametrize(
    ("status", "expected_error"),
    [("completed", False), ("failed", True)],
)
def test_web_search_success_and_failure_normalize_with_streaming_parity(
    status: str,
    expected_error: bool,
) -> None:
    item = SimpleNamespace(
        type="web_search_call",
        id=f"ws_{status}",
        status=status,
        action=SimpleNamespace(type="search", query="Chrys"),
    )

    parsed = _assert_blocking_streaming_parity([item])
    call, result = parsed.messages[0].contents

    assert (call.type, result.type) == ("search_tool_call", "search_tool_result")
    assert call.retry_safety == "read_only"
    assert result.provider_status == status
    assert result.additional_properties.get("is_error", False) is expected_error


@pytest.mark.parametrize(
    ("status", "expected_phase"),
    [
        ("in_progress", "snapshot"),
        ("searching", "snapshot"),
        ("queued", "snapshot"),
        ("completed", "terminal"),
        ("failed", "terminal"),
        (None, "terminal"),
    ],
)
def test_hosted_item_phase_tracks_provider_status(status: str | None, expected_phase: str) -> None:
    # Background-mode responses surface still-running items; stamping those
    # terminal would flip live cards to a final state prematurely.
    item = SimpleNamespace(
        type="web_search_call",
        id="ws_phase",
        action=SimpleNamespace(type="search", query="Chrys"),
    )
    if status is not None:
        item.status = status

    parsed = decode_response(_response([item]), {"store": False}, variant=OPENAI_RESPONSES)
    call, result = parsed.messages[0].contents

    assert call.provider_phase == expected_phase
    assert result.provider_phase == expected_phase


def test_file_search_normalizes_results_and_round_trips() -> None:
    item = SimpleNamespace(
        type="file_search_call",
        id="fs_1",
        queries=["provider normalization"],
        results=[SimpleNamespace(file_id="file_1", filename="plan.md", score=0.98, text="match")],
        status="completed",
    )

    parsed = _assert_blocking_streaming_parity([item])
    replayed = encode_input(parsed.messages, service_side=False, variant=OPENAI_RESPONSES)

    assert replayed == [
        {
            "type": "file_search_call",
            "id": "fs_1",
            "queries": ["provider normalization"],
            "results": [{"file_id": "file_1", "filename": "plan.md", "score": 0.98, "text": "match"}],
            "status": "completed",
        }
    ]


@pytest.mark.parametrize(
    ("output", "error", "status"),
    [("found", None, "completed"), (None, "remote failed", "failed")],
)
def test_mcp_output_and_error_normalize_with_streaming_parity(
    output: str | None,
    error: str | None,
    status: str,
) -> None:
    item = SimpleNamespace(
        type="mcp_call",
        id=f"mcp_{status}",
        arguments='{"query":"chrys"}',
        name="lookup",
        server_label="docs",
        output=output,
        error=error,
        status=status,
    )

    parsed = _assert_blocking_streaming_parity([item])
    call, result = parsed.messages[0].contents

    assert (call.type, result.type) == ("mcp_server_tool_call", "mcp_server_tool_result")
    assert call.retry_safety == "side_effectful"
    assert result.provider_status == status
    assert result.additional_properties.get("is_error", False) is (error is not None)


def test_mcp_replay_consumes_reused_id_by_occurrence() -> None:
    items = [
        SimpleNamespace(
            type="mcp_call",
            id="mcp_reused",
            arguments=f'{{"query":"{query}"}}',
            name="lookup",
            server_label="docs",
            output=output,
            error=None,
            status="completed",
        )
        for query, output in (("first", "first result"), ("second", "second result"))
    ]
    parsed = decode_response(_response(items), {"store": False}, variant=OPENAI_RESPONSES)

    replayed = encode_input(parsed.messages, service_side=False, variant=OPENAI_RESPONSES)

    assert [item["output"] for item in replayed] == ["first result", "second result"]


def test_mcp_sibling_calls_share_exchange_result_scope() -> None:
    call1 = Content.from_mcp_server_tool_call("mcp_1", "first", server_name="docs", arguments="{}")
    call2 = Content.from_mcp_server_tool_call("mcp_2", "second", server_name="docs", arguments="{}")
    result1 = Content.from_mcp_server_tool_result("mcp_1", output=[Content.from_text("first result")])
    result2 = Content.from_mcp_server_tool_result("mcp_2", output=[Content.from_text("second result")])

    replayed = encode_input(
        [
            Message("assistant", [call1]),
            Message("assistant", [call2]),
            Message("tool", [result1, result2]),
        ],
        service_side=False,
        variant=OPENAI_RESPONSES,
    )

    assert [item["output"] for item in replayed] == ["first result", "second result"]


def test_code_delta_logs_image_and_file_fold_to_one_pair() -> None:
    item = SimpleNamespace(
        type="code_interpreter_call",
        id="ci_1",
        call_id="call_ci_1",
        container_id="container_1",
        code="print('hi')",
        outputs=[
            SimpleNamespace(type="logs", logs="hi\n"),
            SimpleNamespace(type="image", url="https://example.test/plot.png"),
            # doc-derived fixture; confirm via probe P-2
            SimpleNamespace(type="hosted_file", file_id="file_1", filename="report.csv"),
        ],
        status="completed",
    )

    parsed = _assert_blocking_streaming_parity([item])
    call, result = parsed.messages[0].contents

    assert (call.type, result.type) == ("code_interpreter_tool_call", "code_interpreter_tool_result")
    assert call.inputs[0].text == "print('hi')"
    assert [output.type for output in result.outputs] == ["text", "uri", "hosted_file"]
    assert call.retry_safety == "sandboxed"


def test_image_partial_snapshots_fold_to_one_final_pair() -> None:
    item = SimpleNamespace(
        type="image_generation_call",
        id="img_1",
        result="aW1nLWZpbmFs",
        status="completed",
    )

    parsed = _assert_blocking_streaming_parity([item])
    call, result = parsed.messages[0].contents

    assert (call.type, result.type) == ("image_generation_tool_call", "image_generation_tool_result")
    assert len(result.outputs) == 1
    assert result.outputs[0].uri.endswith("aW1nLWZpbmFs")
    assert "result" not in call.additional_properties[OPENAI_HOSTED_WIRE_ITEM_KEY]
    persisted = json.dumps([call.to_dict(), result.to_dict()])
    assert persisted.count("aW1nLWZpbmFs") == 1

    replayed = encode_input([Message("assistant", [call, result])], service_side=False, variant=OPENAI_RESPONSES)
    assert replayed[0]["type"] == "message"
    assert "Status: completed" in replayed[0]["content"][0]["text"]
    assert "Images: 1" in replayed[0]["content"][0]["text"]
    assert "aW1nLWZpbmFs" not in json.dumps(replayed)


def test_image_result_terminalizes_stale_generating_status() -> None:
    item = SimpleNamespace(
        type="image_generation_call",
        id="img_stale_status",
        result="aW1nLWZpbmFs",
        status="generating",
        action="generate",
        background="opaque",
        output_format="png",
        quality="medium",
        revised_prompt="A revised prompt",
        size="1024x1024",
    )

    parsed = _assert_blocking_streaming_parity([item])
    call, result = parsed.messages[0].contents
    replayed = encode_input(parsed.messages, service_side=False, variant=OPENAI_RESPONSES)

    assert call.provider_status == result.provider_status == "generating"
    assert call.provider_phase == result.provider_phase == "terminal"
    assert replayed[0]["type"] == "message"
    assert "Status: completed" in replayed[0]["content"][0]["text"]
    assert "img_stale_status" not in json.dumps(replayed)
    assert "aW1nLWZpbmFs" not in json.dumps(replayed)


def test_image_replay_consumes_reused_id_by_occurrence() -> None:
    items = [
        SimpleNamespace(
            type="image_generation_call",
            id="img_reused",
            result=result,
            status="completed",
        )
        for result in ("aW1nLWZpcnN0", "aW1nLXNlY29uZA==")
    ]
    parsed = decode_response(_response(items), {"store": False}, variant=OPENAI_RESPONSES)

    replayed = encode_input(parsed.messages, service_side=False, variant=OPENAI_RESPONSES)

    assert [item["type"] for item in replayed] == ["message", "message"]
    assert all("Images: 1" in item["content"][0]["text"] for item in replayed)
    assert "img_reused" not in json.dumps(replayed)
    assert "aW1n" not in json.dumps(replayed)


def test_image_sibling_calls_share_exchange_result_scope() -> None:
    items = [
        SimpleNamespace(
            type="image_generation_call",
            id=image_id,
            result=result,
            status="completed",
        )
        for image_id, result in (("img_1", "aW1nLWZpcnN0"), ("img_2", "aW1nLXNlY29uZA=="))
    ]
    parsed = decode_response(_response(items), {"store": False}, variant=OPENAI_RESPONSES)
    call1, result1, call2, result2 = parsed.messages[0].contents

    replayed = encode_input(
        [
            Message("assistant", [call1]),
            Message("assistant", [call2]),
            Message("tool", [result1, result2]),
        ],
        service_side=False,
        variant=OPENAI_RESPONSES,
    )

    assert [item["type"] for item in replayed] == ["message", "message"]
    assert all("Images: 1" in item["content"][0]["text"] for item in replayed)
    assert "img_1" not in json.dumps(replayed)
    assert "img_2" not in json.dumps(replayed)


def test_result_only_image_with_conflicting_annotation_replays_payload() -> None:
    item = SimpleNamespace(
        type="image_generation_call",
        id="img_grouped",
        result="aW1nLWdyb3VwZWQ=",
        status="completed",
    )
    parsed = decode_response(_response([item]), {"store": False}, variant=OPENAI_RESPONSES)
    call, result = parsed.messages[0].contents

    def _annotated(message: Message, group_id: str) -> Message:
        message.additional_properties[GROUP_ANNOTATION_KEY] = {
            GROUP_ID_KEY: group_id,
            GROUP_KIND_KEY: "tool_call",
            GROUP_INDEX_KEY: 0,
            GROUP_HAS_REASONING_KEY: False,
        }
        return message

    call_message = _annotated(Message("assistant", [call]), "call_group")
    result_message = _annotated(Message("assistant", [result]), "result_group")

    replayed = encode_input([call_message, result_message], service_side=False, variant=OPENAI_RESPONSES)

    assert len(replayed) == 1
    assert replayed[0]["type"] == "message"
    assert "Images: 1" in replayed[0]["content"][0]["text"]
    assert "img_grouped" not in json.dumps(replayed)
    assert "aW1nLWdyb3VwZWQ=" not in json.dumps(replayed)


def test_image_partial_snapshots_terminalize_when_done_has_no_result() -> None:
    state = StreamState({}, model="gpt-test", variant=OPENAI_RESPONSES)
    partials = ["aW1nLTE=", "aW1nLTI="]
    updates = [
        state.update_for(
            SimpleNamespace(
                type="response.image_generation_call.partial_image",
                item_id="img_failed",
                output_index=0,
                partial_image_index=partial_index,
                partial_image_b64=partial,
            )
        )
        for partial_index, partial in enumerate(partials)
    ]
    done_item = SimpleNamespace(
        type="image_generation_call",
        id="img_failed",
        result=None,
        status="failed",
    )
    updates.append(state.update_for(SimpleNamespace(type="response.output_item.done", item=done_item, output_index=0)))

    parsed = ChatResponse.from_updates(updates)
    call, result = parsed.messages[0].contents
    replayed = encode_input(parsed.messages, service_side=False, variant=OPENAI_RESPONSES)

    assert (call.type, result.type) == ("image_generation_tool_call", "image_generation_tool_result")
    assert call.provider_phase == result.provider_phase == "terminal"
    assert result.provider_status == "failed"
    assert [output.uri.rsplit(",", 1)[-1] for output in result.outputs] == partials
    assert result.additional_properties[OPENAI_HOSTED_REPLAY_SHADOW_KEY] is True
    assert [item["type"] for item in replayed] == ["message"]
    assert "Status: failed" in replayed[0]["content"][0]["text"]
    assert "Images: 2" in replayed[0]["content"][0]["text"]
    assert "img_failed" not in json.dumps(replayed)


@pytest.mark.parametrize(
    ("outcome", "expected_status", "expected_error"),
    [
        (SimpleNamespace(type="exit", exit_code=0), "completed", False),
        (SimpleNamespace(type="exit", exit_code=7), "incomplete", True),
        (SimpleNamespace(type="timeout"), "incomplete", True),
    ],
)
def test_hosted_shell_success_nonzero_and_timeout(
    outcome: SimpleNamespace,
    expected_status: str,
    expected_error: bool,
) -> None:
    call = SimpleNamespace(
        type="shell_call",
        id="sh_1",
        call_id="call_sh_1",
        action=SimpleNamespace(commands=["echo hi"], timeout_ms=1000, max_output_length=2000),
        status="completed",
    )
    output = SimpleNamespace(
        type="shell_call_output",
        id="sho_1",
        call_id="call_sh_1",
        output=[SimpleNamespace(stdout="ok", stderr="boom" if expected_error else "", outcome=outcome)],
        max_output_length=2000,
        status=expected_status,
    )

    parsed = _assert_blocking_streaming_parity([call, output])
    shell_call, shell_result = parsed.messages[0].contents

    assert (shell_call.type, shell_result.type) == ("shell_tool_call", "shell_tool_result")
    assert shell_call.retry_safety == "side_effectful"
    assert shell_result.additional_properties.get("is_error", False) is expected_error


def test_tool_search_terminal_call_and_output_normalize_and_round_trip() -> None:
    # probe-confirmed fixture (P-1, 2026-08-07)
    call = SimpleNamespace(
        type="tool_search_call",
        id="tsc_1",
        call_id="call_ts_1",
        arguments={"query": "weather"},
        execution="server",
        status="completed",
        created_by=None,
    )
    # probe-confirmed fixture (P-1, 2026-08-07)
    output = SimpleNamespace(
        type="tool_search_output",
        id="tso_1",
        call_id="call_ts_1",
        execution="server",
        status="completed",
        tools=[{"type": "function", "name": "get_weather", "parameters": {"type": "object"}}],
        created_by=None,
    )

    call_only = _assert_blocking_streaming_parity([call])
    parsed = _assert_blocking_streaming_parity([call, output])
    replayed = encode_input(parsed.messages, service_side=False, variant=OPENAI_RESPONSES)

    assert [content.type for content in call_only.messages[0].contents] == ["hosted_tool_call"]
    assert [content.type for content in parsed.messages[0].contents] == ["hosted_tool_call", "hosted_tool_result"]
    assert all(content.hosted_family == "tool_discovery" for content in parsed.messages[0].contents)
    assert [item["type"] for item in replayed] == ["tool_search_call", "tool_search_output"]


def test_apply_patch_call_preserves_informational_call_and_completes_replay() -> None:
    # probe-confirmed fixture (P-2, 2026-08-07): client-executed Codex-style
    # patch call; replay without an output item is rejected by the API.
    item = SimpleNamespace(
        type="apply_patch_call",
        id="apc_1",
        call_id="call_patch_1",
        status="completed",
        operation=SimpleNamespace(type="create_file", path="hello.txt", diff="+hi there\n"),
    )

    parsed = _assert_blocking_streaming_parity([item])
    content = parsed.messages[0].contents[0]

    assert content.type == "function_call"
    assert content.informational_only is True
    assert content.name == "apply_patch"
    assert content.call_id == "call_patch_1"
    assert content.arguments == {"type": "create_file", "path": "hello.txt", "diff": "+hi there\n"}

    replayed = encode_input(parsed.messages, service_side=False, variant=OPENAI_RESPONSES)

    assert [item["type"] for item in replayed] == ["apply_patch_call", "apply_patch_call_output"]
    assert replayed[0]["call_id"] == "call_patch_1"
    synthesized = replayed[1]
    assert synthesized["call_id"] == "call_patch_1"
    assert synthesized["status"] == "failed"
    assert synthesized["output"]


def test_complete_unanswered_apply_patch_calls_skips_answered_calls() -> None:
    items = [
        {"type": "apply_patch_call", "call_id": "c1"},
        {"type": "apply_patch_call_output", "call_id": "c1", "status": "completed", "output": "ok"},
        {"type": "apply_patch_call", "call_id": "c2"},
        {"type": "message", "role": "assistant", "content": "done"},
    ]

    completed = answer_apply_patch_calls(items)

    assert [item.get("type") for item in completed] == [
        "apply_patch_call",
        "apply_patch_call_output",
        "apply_patch_call",
        "apply_patch_call_output",
        "message",
    ]
    assert completed[3] == {
        "type": "apply_patch_call_output",
        "call_id": "c2",
        "status": "failed",
        "output": "The client did not execute this apply_patch call.",
    }
    assert completed[1]["status"] == "completed"


@pytest.mark.parametrize("answered_exchange", [None, 0, 1])
def test_apply_patch_completion_is_scoped_to_each_exchange(answered_exchange: int | None) -> None:
    messages: list[Message] = []
    for index in range(2):
        call = SimpleNamespace(
            type="apply_patch_call",
            id=f"apc_{index}",
            call_id="reused",
            status="completed",
            operation=SimpleNamespace(type="create_file", path=f"file{index}.txt", diff="+content\n"),
        )
        messages.append(Message(role="user", contents=[Content.from_text(f"request {index}")]))
        messages.extend(decode_response(_response([call]), {"store": False}, variant=OPENAI_RESPONSES).messages)
        if index == answered_exchange:
            result = Content.from_function_result(
                "reused",
                result="executed",
                additional_properties={
                    OPENAI_HOSTED_WIRE_ITEM_KEY: {
                        "type": "apply_patch_call_output",
                        "call_id": "reused",
                        "status": "completed",
                        "output": "executed",
                    }
                },
            )
            result.hosted_provider = "openai"
            messages.append(Message(role="tool", contents=[result]))
    replay = encode_input(messages, service_side=False, variant=OPENAI_RESPONSES)
    assert [item["type"] for item in replay if item.get("type", "").startswith("apply_patch")] == [
        "apply_patch_call",
        "apply_patch_call_output",
        "apply_patch_call",
        "apply_patch_call_output",
    ]
    results = [item for item in replay if item.get("type") == "apply_patch_call_output"]
    assert [item["status"] for item in results] == [
        "completed" if index == answered_exchange else "failed" for index in range(2)
    ]
    assert all(item["call_id"] == "reused" for item in results)


def test_unknown_server_item_uses_fail_closed_generic_pair() -> None:
    # doc-derived fixture; confirm via probe P-2
    item = SimpleNamespace(
        type="skills_call",
        id="skill_1",
        call_id="call_skill_1",
        name="spreadsheet_skill",
        arguments={"file_id": "file_1"},
        execution="server",
        output={"artifact_id": "artifact_1"},
        status="completed",
    )

    parsed = _assert_blocking_streaming_parity([item])
    call, result = parsed.messages[0].contents

    assert (call.type, result.type) == ("hosted_tool_call", "hosted_tool_result")
    assert call.hosted_family == "generic"
    assert call.retry_safety == "unknown"


def test_custom_computer_and_local_shell_are_not_misclassified() -> None:
    custom = SimpleNamespace(
        type="custom_tool_call",
        id="ctc_1",
        call_id="call_custom_1",
        name="python",
        input="print('hi')",
        namespace="container",
    )
    computer = SimpleNamespace(
        type="computer_call",
        id="computer_1",
        call_id="call_computer_1",
        action=SimpleNamespace(type="click", x=1, y=2),
        status="completed",
    )

    parsed = decode_response(_response([custom, computer]), {"store": False}, variant=OPENAI_RESPONSES)
    streamed = _stream_parse([custom, computer])

    assert len(parsed.messages[0].contents) == 1
    assert _contents(streamed) == _contents(parsed)
    assert parsed.messages[0].contents[0].type == "function_call"
    assert parsed.messages[0].contents[0].informational_only is True

    local_shell = SimpleNamespace(
        type="local_shell_call",
        id="lsc_1",
        call_id="call_local_1",
        action=SimpleNamespace(command=["echo", "hi"], timeout_ms=1000),
        status="completed",
    )
    blocking = decode_response(_response([local_shell]), {"store": False}, variant=OPENAI_RESPONSES)
    streaming = _stream_parse([local_shell])

    assert _contents(streaming) == _contents(blocking)
    assert blocking.messages[0].contents[0].type == "shell_tool_call"
    assert blocking.messages[0].contents[0].provider_hosted is True


def test_openai_specialized_hosted_items_same_provider_round_trip() -> None:
    items = [
        SimpleNamespace(
            type="web_search_call",
            id="ws_1",
            status="completed",
            action=SimpleNamespace(type="search", query="Chrys"),
        ),
        SimpleNamespace(
            type="mcp_call",
            id="mcp_1",
            arguments='{"query":"Chrys"}',
            name="lookup",
            server_label="docs",
            output="found",
            error=None,
            status="completed",
        ),
        SimpleNamespace(
            type="code_interpreter_call",
            id="ci_1",
            container_id="container_1",
            code="print('hi')",
            outputs=[SimpleNamespace(type="logs", logs="hi\n")],
            status="completed",
        ),
        SimpleNamespace(
            type="image_generation_call",
            id="img_1",
            result="aW1n",
            status="completed",
        ),
        SimpleNamespace(
            type="shell_call",
            id="sh_1",
            call_id="call_sh_1",
            action=SimpleNamespace(commands=["pwd"], timeout_ms=1000, max_output_length=2000),
            status="completed",
            created_by=None,
        ),
        SimpleNamespace(
            type="shell_call_output",
            id="sho_1",
            call_id="call_sh_1",
            output=[
                SimpleNamespace(
                    stdout="/tmp",
                    stderr="",
                    outcome=SimpleNamespace(type="exit", exit_code=0),
                )
            ],
            max_output_length=2000,
            status="completed",
            created_by=None,
        ),
    ]
    parsed = decode_response(_response(items), {"store": False}, variant=OPENAI_RESPONSES)

    replayed = encode_input(parsed.messages, service_side=False, variant=OPENAI_RESPONSES)

    assert [item["type"] for item in replayed] == [
        "web_search_call",
        "mcp_call",
        "code_interpreter_call",
        "message",
        "shell_call",
        "shell_call_output",
    ]
    assert replayed[0]["action"] == {"type": "search", "query": "Chrys"}
    assert replayed[1]["output"] == "found"
    assert replayed[2]["outputs"] == [{"type": "logs", "logs": "hi\n"}]
    assert "Images: 1" in replayed[3]["content"][0]["text"]
    assert "aW1n" not in json.dumps(replayed[3])
    assert replayed[4]["action"]["commands"] == ["pwd"]
    assert "status" not in replayed[5]
    assert replayed[5]["output"][0]["outcome"] == {"type": "exit", "exit_code": 0}


def test_unrecognized_non_execution_items_are_skipped_with_counter_friendly_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    items = [
        SimpleNamespace(type="mcp_list_tools", id="list_1", tools=[], server_label="docs"),
        SimpleNamespace(type="mcp_approval_request", id="approval_1", arguments="{}", name="x"),
    ]

    with caplog.at_level("DEBUG", logger="chrys.service.llm.openai_responses"):
        parsed = decode_response(_response(items), {"store": False}, variant=OPENAI_RESPONSES)

    assert parsed.messages[0].contents == []
    assert caplog.text.count("responses_parser_unrecognized_server_item") == 2


def test_openai_generic_same_provider_round_trip_replays_one_wire_item() -> None:
    # doc-derived fixture; confirm via probe P-2
    item = SimpleNamespace(
        type="skills_call",
        id="skill_1",
        call_id="call_skill_1",
        name="document_skill",
        arguments={"task": "summarize"},
        execution="server",
        output={"text": "done"},
        status="completed",
    )
    parsed = decode_response(_response([item]), {"store": False}, variant=OPENAI_RESPONSES)

    replayed = encode_input(
        [Message("assistant", parsed.messages[0].contents)], service_side=False, variant=OPENAI_RESPONSES
    )

    assert replayed == [vars(item)]


@pytest.mark.parametrize("source_provider", ["anthropic", "deepseek-openai"])
@pytest.mark.parametrize("family", ["search", "generic"])
def test_openai_cross_provider_hosted_history_degrades_to_assistant_context(
    source_provider: str,
    family: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    if family == "search":
        call = Content.from_search_tool_call(
            "foreign-1",
            tool_name="web_search",
            arguments={"query": "Chrys"},
            status="running",
            hosted_provider=source_provider,
            provider_phase="start",
        )
        result = Content.from_search_tool_result(
            "foreign-1",
            tool_name="web_search",
            result="found",
            status="completed",
            hosted_provider=source_provider,
            provider_phase="terminal",
        )
    else:
        call = Content.from_hosted_tool_call(
            "foreign-1",
            tool_name="remote_task",
            arguments={"task": "inspect"},
            status="running",
            hosted_provider=source_provider,
            provider_phase="start",
        )
        result = Content.from_hosted_tool_result(
            "foreign-1",
            tool_name="remote_task",
            result="found",
            status="completed",
            hosted_provider=source_provider,
            provider_phase="terminal",
        )

    with caplog.at_level("DEBUG", logger="chrys.service.agent_middleware.events.hosted_tools"):
        replayed = encode_input(
            [Message("assistant", [call]), Message("assistant", [result])], service_side=False, variant=OPENAI_RESPONSES
        )

    assert len(replayed) == 1
    assert replayed[0]["type"] == "message"
    assert replayed[0]["role"] == "assistant"
    summary = replayed[0]["content"][0]["text"]
    assert "[Provider-hosted tool context]" in summary
    assert "Status: completed" in summary
    assert 'Result: "found"' in summary
    assert "Degrading provider-hosted history to assistant context" in caplog.text

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Order-preserving serialization tests for OpenAI-compatible Responses input."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from chrys.kernel import Annotation, ChatResponse, Content, Message, TextSpanRegion
from chrys.service.llm.openai_responses.client import DEEPSEEK_RESPONSES, OPENAI_RESPONSES, ResponsesVariant
from chrys.service.llm.openai_responses.decode import decode_response
from chrys.service.llm.openai_responses.history import encode_message
from chrys.service.llm.openai_responses.replay import encode_input
from chrys.service.llm.openai_responses.stream import StreamState
from chrys.service.session.history import SessionHistoryManager

_MODEL = "test-model"


def _encode(
    messages: list[Message], *, service_side: bool = True, variant: ResponsesVariant = OPENAI_RESPONSES
) -> list[dict[str, object]]:
    return encode_input(messages, service_side=service_side, variant=variant)


def _reasoning(*, encrypted: bool = True, reasoning_id: str = "rs_1") -> Content:
    return Content.from_text_reasoning(
        id=reasoning_id,
        text="thinking",
        protected_data="encrypted" if encrypted else None,
    )


def _call(index: int, *, fc_id: str | None = None) -> Content:
    return Content.from_function_call(
        call_id=f"call_{index}",
        name=f"tool_{index}",
        arguments="{}",
        additional_properties={"fc_id": fc_id or f"fc_{index}"},
    )


def _result(index: int) -> Content:
    return Content.from_function_result(call_id=f"call_{index}", result=f"result {index}")


def _types(items: list[dict[str, object]]) -> list[str]:
    return [str(item["type"]) for item in items]


def test_reasoning_text_parallel_calls_and_outputs_preserve_non_degraded_source_order() -> None:
    messages = [
        Message("assistant", [_reasoning(), Content.from_text("I will check."), _call(1), _call(2), _call(3)]),
        Message("tool", [_result(1), _result(2), _result(3)]),
    ]

    prepared = _encode(messages, service_side=False)

    assert _types(prepared) == [
        "reasoning",
        "message",
        "function_call",
        "function_call",
        "function_call",
        "function_call_output",
        "function_call_output",
        "function_call_output",
    ]
    assert prepared[1]["content"] == [{"type": "output_text", "text": "I will check.", "annotations": []}]
    assert [item.get("id") for item in prepared[2:5]] == ["fc_1", "fc_2", "fc_3"]


def test_reasoning_text_parallel_calls_and_outputs_preserve_degraded_source_order() -> None:
    messages = [
        Message(
            "assistant",
            [_reasoning(encrypted=False), Content.from_text("I will check."), _call(1), _call(2), _call(3)],
        ),
        Message("tool", [_result(1), _result(2), _result(3)]),
    ]

    prepared = _encode(messages, service_side=False)

    assert _types(prepared) == [
        "message",
        "function_call",
        "function_call",
        "function_call",
        "function_call_output",
        "function_call_output",
        "function_call_output",
    ]
    assert all("id" not in item for item in prepared[1:4])


def test_interleaved_message_runs_and_calls_preserve_source_order() -> None:
    message = Message(
        "assistant",
        [Content.from_text("first"), _call(1), Content.from_text("second"), _call(2)],
    )

    prepared = _encode([message], service_side=False)

    assert _types(prepared) == ["message", "function_call", "message", "function_call"]
    assert prepared[0]["content"][0]["text"] == "first"
    assert prepared[2]["content"][0]["text"] == "second"


def test_multimodal_message_runs_split_at_calls_without_reordering() -> None:
    message = Message(
        "assistant",
        [
            Content.from_text("caption"),
            Content.from_uri("https://example.test/image.png", media_type="image/png"),
            _call(1),
            Content.from_text("after"),
        ],
    )

    prepared = _encode([message], service_side=False)

    assert _types(prepared) == ["message", "function_call", "message"]
    assert [part["type"] for part in prepared[0]["content"]] == ["output_text", "input_image"]
    assert prepared[2]["content"][0]["text"] == "after"


@pytest.mark.parametrize("service_storage", [False, True])
@pytest.mark.parametrize("text_first", [False, True])
def test_assistant_text_and_function_result_preserve_both_content_orders(
    service_storage: bool,
    text_first: bool,
) -> None:
    text = Content.from_text("answer")
    result = _result(1)
    message = Message("assistant", [text, result] if text_first else [result, text])

    prepared = _encode(
        [message],
        service_side=service_storage,
    )

    expected = ["message", "function_call_output"] if text_first else ["function_call_output", "message"]
    assert _types(prepared) == expected


def test_canonical_single_kind_histories_keep_their_wire_shapes() -> None:
    assert _encode([Message("assistant", [Content.from_text("answer")])]) == [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "answer", "annotations": []}],
        }
    ]
    assert _types(
        _encode(
            [Message("assistant", [_call(1), _call(2)])],
            service_side=False,
        )
    ) == ["function_call", "function_call"]
    assert _encode([Message("tool", [_result(1)])]) == [
        {"type": "function_call_output", "call_id": "call_1", "output": "result 1"}
    ]


def test_dropped_and_empty_preparations_do_not_split_a_message_run() -> None:
    dropped = _call(1)
    unsupported_media = Content.from_uri("https://example.test/video.mp4", media_type="video/mp4")
    message = Message(
        "assistant", [Content.from_text("before"), dropped, unsupported_media, Content.from_text("after")]
    )

    prepared = encode_message(
        message,
        provider="openai",
        service_side=False,
        dropped={id(dropped)},
    )

    assert _types(prepared) == ["message"]
    assert [part["text"] for part in prepared[0]["content"]] == ["before", "after"]


def test_degraded_mcp_group_with_surrounding_text_has_no_marker_or_empty_message() -> None:
    call = Content.from_mcp_server_tool_call("mcp_1", "search", server_name="remote", arguments="{}")
    result = Content.from_mcp_server_tool_result("mcp_1", output=[Content.from_text("found")])
    messages = [
        Message(
            "assistant", [Content.from_text("before"), _reasoning(encrypted=False), call, Content.from_text("after")]
        ),
        Message("tool", [result]),
    ]

    prepared = _encode(messages, service_side=False)

    assert _types(prepared) == ["message"]
    assert [part["text"] for part in prepared[0]["content"]] == ["before", "after"]


def test_mcp_coalescing_keeps_interposed_message_after_mutated_call() -> None:
    call = Content.from_mcp_server_tool_call("mcp_1", "search", server_name="remote", arguments="{}")
    result = Content.from_mcp_server_tool_result("mcp_1", output=[Content.from_text("found")])

    prepared = _encode(
        [Message("assistant", [call, Content.from_text("note")]), Message("tool", [result])],
        service_side=False,
    )

    assert _types(prepared) == ["mcp_call", "message"]
    assert prepared[0]["output"] == "found"


def test_unmatched_mcp_result_marker_is_removed_without_reordering_message_content() -> None:
    result = Content.from_mcp_server_tool_result("missing", output=[Content.from_text("orphan")])

    prepared = _encode(
        [Message("tool", [Content.from_text("before"), result, Content.from_text("after")])],
        service_side=False,
    )

    assert _types(prepared) == ["message", "message"]
    assert [item["content"][0]["text"] for item in prepared] == ["before", "after"]


def test_duplicate_mcp_ids_still_degrade_around_inserted_message_runs() -> None:
    calls = [
        Content.from_mcp_server_tool_call("mcp_dup", "first", server_name="remote", arguments="{}"),
        Content.from_mcp_server_tool_call("mcp_dup", "second", server_name="remote", arguments="{}"),
    ]

    prepared = _encode(
        [Message("assistant", [_reasoning(), calls[0], Content.from_text("kept"), calls[1]])],
        service_side=False,
    )

    assert _types(prepared) == ["message"]
    assert prepared[0]["content"][0]["text"] == "kept"


@pytest.mark.parametrize("collision", ["reasoning", "function"])
def test_item_id_collisions_still_degrade_with_inserted_message_runs(collision: str) -> None:
    reasoning_id = "fc_dup" if collision == "reasoning" else "rs_1"
    second_fc_id = "fc_2" if collision == "reasoning" else "fc_dup"
    first_fc_id = "fc_dup"
    message = Message(
        "assistant",
        [
            _reasoning(reasoning_id=reasoning_id),
            Content.from_text("kept"),
            _call(1, fc_id=first_fc_id),
            _call(2, fc_id=second_fc_id),
        ],
    )

    prepared = _encode([message], service_side=False)

    assert _types(prepared) == ["message", "function_call", "function_call"]
    assert all("id" not in item for item in prepared[1:])


def test_synthetic_response_output_parse_then_replay_preserves_normalized_item_order() -> None:
    response = SimpleNamespace(
        id="resp_1",
        created_at=0,
        model="test-model",
        metadata={},
        output=[
            SimpleNamespace(
                type="reasoning",
                id="rs_1",
                content=[SimpleNamespace(text="private")],
                summary=[],
                encrypted_content="encrypted",
            ),
            SimpleNamespace(
                type="message",
                id="msg_1",
                status="incomplete",
                phase="commentary",
                content=[SimpleNamespace(type="output_text", text="working", annotations=[])],
            ),
            SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="call_1",
                name="lookup",
                arguments="{}",
                status="completed",
            ),
        ],
        usage=None,
        conversation=None,
        status="completed",
        incomplete_details=None,
    )
    parsed = decode_response(response, {}, variant=OPENAI_RESPONSES)
    restored = ChatResponse.from_dict(parsed.to_dict())

    prepared = _encode(
        restored.messages,
        service_side=False,
    )

    assert _types(prepared) == ["reasoning", "message", "function_call"]
    assert {key: prepared[1][key] for key in ("id", "status", "phase")} == {
        "id": "msg_1",
        "status": "incomplete",
        "phase": "commentary",
    }


def test_streamed_output_message_replays_the_done_envelope_verbatim() -> None:
    state = StreamState({}, model=_MODEL, variant=OPENAI_RESPONSES)
    added_item = SimpleNamespace(
        type="message",
        id="msg_stream",
        status="in_progress",
        phase="commentary",
    )
    done_item = SimpleNamespace(
        type="message",
        id="msg_stream",
        status="incomplete",
        phase="commentary",
    )

    updates = [
        state.update_for(SimpleNamespace(type="response.output_item.added", item=added_item, output_index=0)),
        state.update_for(
            SimpleNamespace(
                type="response.output_text.delta",
                delta="streamed",
                item_id="msg_stream",
                output_index=0,
                content_index=0,
            )
        ),
        state.update_for(SimpleNamespace(type="response.output_item.done", item=done_item, output_index=0)),
    ]

    response = ChatResponse.from_updates(updates)
    prepared = _encode(response.messages)

    assert prepared == [
        {
            "type": "message",
            "role": "assistant",
            "id": "msg_stream",
            "status": "incomplete",
            "phase": "commentary",
            "content": [{"type": "output_text", "text": "streamed", "annotations": []}],
        }
    ]


@pytest.mark.parametrize("persisted", [False, True], ids=["in_memory", "persisted"])
def test_consecutive_streamed_output_messages_replay_as_distinct_items(persisted: bool) -> None:
    state = StreamState({}, model=_MODEL, variant=OPENAI_RESPONSES)
    first = SimpleNamespace(
        type="message",
        id="msg_1",
        status="completed",
        phase="commentary",
    )
    second = SimpleNamespace(
        type="message",
        id="msg_2",
        status="completed",
        phase="final_answer",
    )

    events = [
        SimpleNamespace(type="response.output_item.added", item=first, output_index=0),
        SimpleNamespace(
            type="response.output_text.delta",
            delta="working",
            item_id="msg_1",
            output_index=0,
            content_index=0,
        ),
        SimpleNamespace(type="response.output_item.done", item=first, output_index=0),
        SimpleNamespace(type="response.output_item.added", item=second, output_index=1),
        SimpleNamespace(
            type="response.output_text.delta",
            delta="done",
            item_id="msg_2",
            output_index=1,
            content_index=0,
        ),
        SimpleNamespace(type="response.output_item.done", item=second, output_index=1),
    ]
    updates = [state.update_for(event) for event in events]

    response = ChatResponse.from_updates(updates)
    if persisted:
        response = ChatResponse.from_dict(response.to_dict())
    prepared = _encode(response.messages)

    assert prepared == [
        {
            "type": "message",
            "role": "assistant",
            "id": "msg_1",
            "status": "completed",
            "phase": "commentary",
            "content": [{"type": "output_text", "text": "working", "annotations": []}],
        },
        {
            "type": "message",
            "role": "assistant",
            "id": "msg_2",
            "status": "completed",
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": "done", "annotations": []}],
        },
    ]


_DIALECTS = pytest.mark.parametrize("variant", [OPENAI_RESPONSES, DEEPSEEK_RESPONSES], ids=["openai", "deepseek"])


def _function_call_items(variant: ResponsesVariant, messages: list[Message]) -> list[dict[str, object]]:
    prepared = _encode(messages, service_side=False, variant=variant)
    return [item for item in prepared if item["type"] == "function_call"]


def _exchange(arguments: str | dict[str, object]) -> list[Message]:
    return [
        Message("user", ["read it"]),
        Message("assistant", [Content.from_function_call(call_id="call_1", name="read_file", arguments=arguments)]),
        Message("tool", [Content.from_function_result(call_id="call_1", result="ok")]),
    ]


@_DIALECTS
@pytest.mark.parametrize("persisted", [False, True], ids=["live", "persisted"])
def test_dict_function_call_arguments_go_out_as_a_json_string(variant: ResponsesVariant, persisted: bool) -> None:
    """Dict arguments (an Anthropic tool_use input) replay as the string the Responses API requires."""
    messages = _exchange({"path": "a.py", "limit": 20})
    if persisted:
        messages = [Message.from_dict(message.to_dict()) for message in messages]

    [item] = _function_call_items(variant, messages)

    assert isinstance(item["arguments"], str)
    assert json.loads(item["arguments"]) == {"path": "a.py", "limit": 20}


@_DIALECTS
def test_string_function_call_arguments_go_out_byte_for_byte(variant: ResponsesVariant) -> None:
    raw = '{ "path":"a.py" }'

    [item] = _function_call_items(variant, _exchange(raw))

    assert item["arguments"] == raw


@_DIALECTS
def test_approval_edited_arguments_replay_as_a_json_string(variant: ResponsesVariant) -> None:
    """An approval edit writes dict arguments into history; the next Responses request still sends a string."""
    messages = _exchange('{"prompt": "old prompt"}')
    history = SessionHistoryManager()
    history.bind({"messages": messages})
    history.persist_approval_decisions(
        [
            {
                "request_id": "req-1",
                "tool_name": "read_file",
                "status": "user_approved",
                "modified_args": '{"prompt": "new prompt"}',
            }
        ]
    )

    [item] = _function_call_items(variant, messages)

    assert json.loads(item["arguments"]) == {"prompt": "new prompt"}


def _container_citation(filename: str | None) -> Annotation:
    return Annotation(
        type="citation",
        file_id="cfile_1",
        url=filename,
        additional_properties={"container_id": "cntr_1"},
        annotated_regions=[
            TextSpanRegion(type="text_span", start_index=0, end_index=3),
            TextSpanRegion(type="text_span", start_index=4, end_index=7),
        ],
    )


def test_container_citation_replays_one_entry_per_region_with_its_filename_last() -> None:
    texts = [
        Content.from_text("abc def", annotations=[_container_citation("a.csv")]),
        Content.from_text("abc def", annotations=[_container_citation(None)]),
    ]

    [message] = _encode([Message("assistant", texts)], service_side=False)

    named, unnamed = (part["annotations"] for part in message["content"])
    base = [("type", "container_file_citation"), ("container_id", "cntr_1"), ("file_id", "cfile_1")]
    assert [list(entry.items()) for entry in named] == [
        [*base, ("start_index", 0), ("end_index", 3), ("filename", "a.csv")],
        [*base, ("start_index", 4), ("end_index", 7), ("filename", "a.csv")],
    ]
    assert [list(entry.items()) for entry in unnamed] == [
        [*base, ("start_index", 0), ("end_index", 3)],
        [*base, ("start_index", 4), ("end_index", 7)],
    ]


def test_mcp_call_and_result_without_call_id_are_not_replayed() -> None:
    messages = [
        Message("user", [Content.from_text("hi")]),
        Message("assistant", [Content.from_mcp_server_tool_call("", "search", server_name="remote", arguments="{}")]),
        Message("tool", [Content.from_mcp_server_tool_result("", output=[Content.from_text("found")])]),
        Message("user", [Content.from_text("next")]),
    ]

    prepared = _encode(messages, service_side=False)

    assert _types(prepared) == ["message", "message"]
    assert [item["role"] for item in prepared] == ["user", "user"]

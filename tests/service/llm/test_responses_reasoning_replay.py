# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Request-boundary coverage for OpenAI Responses reasoning replay."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import AsyncOpenAI, BadRequestError

from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.kernel import ChatResponse, ChatResponseUpdate, Content, Message, annotate_message_groups
from chrys.kernel.compaction import (
    GROUP_ANNOTATION_KEY,
    GROUP_HAS_REASONING_KEY,
    GROUP_ID_KEY,
    GROUP_INDEX_KEY,
    GROUP_KIND_KEY,
)
from chrys.kernel.exceptions import ChatClientException
from chrys.service.llm.openai_responses import ResponsesApiClient
from chrys.service.llm.openai_responses.client import OPENAI_RESPONSES
from chrys.service.llm.openai_responses.history import (
    OPENAI_SHELL_OUTPUT_TYPE_KEY,
    OPENAI_SHELL_OUTPUT_TYPE_LOCAL_SHELL_CALL,
    OPENAI_SHELL_OUTPUT_TYPE_SHELL_CALL,
)
from chrys.service.llm.openai_responses.replay import partition_groups
from chrys.service.llm.openai_responses.request import build_request
from chrys.service.llm.openai_responses.stream import StreamState


class _FakeAsyncOpenAI:
    base_url = "https://api.test"


_USER_ITEM = {
    "type": "message",
    "role": "user",
    "content": [{"type": "input_text", "text": "Start"}],
}


_MODEL = "gpt-test"


def _client(sdk_client: object | None = None) -> ResponsesApiClient:
    return ResponsesApiClient(model=_MODEL, sdk_client=sdk_client or _FakeAsyncOpenAI())


def _build(messages: list[Message], options: dict[str, object]) -> dict[str, object]:
    return build_request(messages, options, model=_MODEL, variant=OPENAI_RESPONSES)


def _fresh_update(event: object) -> ChatResponseUpdate:
    """The update *event* makes as the first event of its own stream."""
    return StreamState({}, model=_MODEL, variant=OPENAI_RESPONSES).update_for(event)


def _reasoning(
    reasoning_id: str | None = "rs_1",
    *,
    text: str = "Plan",
    encrypted_content: str | None = "encrypted",
    reasoning_text: bool | str = False,
    legacy_payload: bool = False,
) -> Content:
    additional_properties: dict[str, object] = {}
    if reasoning_text:
        additional_properties["reasoning_text"] = reasoning_text
    if legacy_payload and encrypted_content:
        additional_properties["encrypted_content"] = encrypted_content
    return Content.from_text_reasoning(
        id=reasoning_id,
        text=text,
        protected_data=None if legacy_payload else encrypted_content,
        additional_properties=additional_properties,
    )


def _function_call(call_id: str, name: str, *, fc_id: str | None = None) -> Content:
    additional_properties = {"fc_id": fc_id} if fc_id else None
    return Content.from_function_call(
        call_id,
        name,
        arguments="{}",
        additional_properties=additional_properties,
    )


def _mcp_call(call_id: str, name: str = "search") -> Content:
    return Content.from_mcp_server_tool_call(
        call_id,
        name,
        server_name="remote",
        arguments='{"q":"cats"}',
    )


def _mcp_result(call_id: str, output: str) -> Content:
    return Content.from_mcp_server_tool_result(call_id, output=[Content.from_text(output)])


def _prepare(
    messages: list[Message],
    options: dict[str, object] | None = None,
) -> dict[str, object]:
    request_options: dict[str, object] = {"store": False}
    if options:
        request_options.update(options)
    return _build(
        [Message("user", ["Start"]), *messages],
        request_options,
    )


def test_completed_reasoning_function_group_replays_exact_items() -> None:
    prepared = _prepare(
        [
            Message(
                "assistant",
                [_reasoning(), _function_call("call_1", "lookup", fc_id="fc_live")],
            ),
            Message("tool", [Content.from_function_result("call_1", result="done")]),
        ]
    )

    assert prepared["input"] == [
        _USER_ITEM,
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "Plan"}],
            "encrypted_content": "encrypted",
        },
        {
            "call_id": "call_1",
            "id": "fc_live",
            "type": "function_call",
            "name": "lookup",
            "arguments": "{}",
        },
        {
            "call_id": "call_1",
            "type": "function_call_output",
            "output": "done",
        },
    ]


def test_active_reasoning_mcp_group_replays_without_output() -> None:
    prepared = _prepare(
        [
            Message("assistant", [_reasoning(), _mcp_call("mcp_1")]),
        ]
    )

    assert prepared["input"] == [
        _USER_ITEM,
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "Plan"}],
            "encrypted_content": "encrypted",
        },
        {
            "type": "mcp_call",
            "id": "mcp_1",
            "server_label": "remote",
            "name": "search",
            "arguments": '{"q":"cats"}',
        },
    ]


def test_parallel_reasoning_function_group_replays_all_calls_and_results() -> None:
    prepared = _prepare(
        [
            Message(
                "assistant",
                [
                    _reasoning(),
                    _function_call("call_1", "first", fc_id="fc_first"),
                    _function_call("call_2", "second", fc_id="fc_second"),
                ],
            ),
            Message(
                "tool",
                [
                    Content.from_function_result("call_1", result="one"),
                    Content.from_function_result("call_2", result="two"),
                ],
            ),
        ]
    )

    assert prepared["input"] == [
        _USER_ITEM,
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "Plan"}],
            "encrypted_content": "encrypted",
        },
        {
            "call_id": "call_1",
            "id": "fc_first",
            "type": "function_call",
            "name": "first",
            "arguments": "{}",
        },
        {
            "call_id": "call_2",
            "id": "fc_second",
            "type": "function_call",
            "name": "second",
            "arguments": "{}",
        },
        {
            "call_id": "call_1",
            "type": "function_call_output",
            "output": "one",
        },
        {
            "call_id": "call_2",
            "type": "function_call_output",
            "output": "two",
        },
    ]


def test_encrypted_only_reasoning_replays_empty_visible_arrays() -> None:
    prepared = _prepare(
        [Message("assistant", [_reasoning(text=""), _function_call("call_1", "lookup", fc_id="fc_live")])]
    )

    assert prepared["input"] == [
        _USER_ITEM,
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "encrypted_content": "encrypted",
        },
        {
            "call_id": "call_1",
            "id": "fc_live",
            "type": "function_call",
            "name": "lookup",
            "arguments": "{}",
        },
    ]


def test_ordinary_function_group_without_payload_degrades_without_item_id() -> None:
    prepared = _prepare(
        [
            Message(
                "assistant",
                [_reasoning(encrypted_content=None), _function_call("call_1", "lookup", fc_id="fc_live")],
            ),
            Message("tool", [Content.from_function_result("call_1", result="done")]),
        ]
    )

    assert prepared["input"] == [
        _USER_ITEM,
        {
            "call_id": "call_1",
            "type": "function_call",
            "name": "lookup",
            "arguments": "{}",
        },
        {
            "call_id": "call_1",
            "type": "function_call_output",
            "output": "done",
        },
    ]


def test_hosted_mcp_group_without_payload_degrades_atomically() -> None:
    prepared = _prepare(
        [
            Message("assistant", [_reasoning(encrypted_content=None), _mcp_call("mcp_1")]),
            Message("tool", [_mcp_result("mcp_1", "done")]),
        ]
    )

    assert prepared["input"] == [_USER_ITEM]


def test_replayable_mcp_group_survives_same_call_id_in_degraded_group() -> None:
    prepared = _prepare(
        [
            Message("assistant", [_reasoning("rs_bad", encrypted_content=None), _mcp_call("shared")]),
            Message("tool", [_mcp_result("shared", "discarded")]),
            Message("assistant", [_reasoning("rs_good"), _mcp_call("shared", "lookup")]),
            Message("tool", [_mcp_result("shared", "kept")]),
        ]
    )

    assert prepared["input"] == [
        _USER_ITEM,
        {
            "type": "reasoning",
            "id": "rs_good",
            "summary": [{"type": "summary_text", "text": "Plan"}],
            "encrypted_content": "encrypted",
        },
        {
            "type": "mcp_call",
            "id": "shared",
            "server_label": "remote",
            "name": "lookup",
            "arguments": '{"q":"cats"}',
            "output": "kept",
        },
    ]


def test_duplicate_call_id_results_coalesce_only_inside_their_group() -> None:
    prepared = _prepare(
        [
            Message("assistant", [_mcp_call("shared", "first")]),
            Message("tool", [_mcp_result("shared", "first output")]),
            Message("assistant", [_reasoning("rs_bad", encrypted_content=None), _mcp_call("shared", "second")]),
            Message("tool", [_mcp_result("shared", "second output")]),
        ]
    )

    assert prepared["input"] == [
        _USER_ITEM,
        {
            "type": "mcp_call",
            "id": "shared",
            "server_label": "remote",
            "name": "first",
            "arguments": '{"q":"cats"}',
            "output": "first output",
        },
    ]


def test_reused_ordinary_call_id_keeps_each_positional_result() -> None:
    prepared = _prepare(
        [
            Message(
                "assistant",
                [_reasoning("rs_good"), _function_call("shared", "first", fc_id="fc_first")],
            ),
            Message("tool", [Content.from_function_result("shared", result="first output")]),
            Message(
                "assistant",
                [
                    _reasoning("rs_bad", encrypted_content=None),
                    _function_call("shared", "second", fc_id="fc_second"),
                ],
            ),
            Message("tool", [Content.from_function_result("shared", result="second output")]),
        ]
    )

    assert prepared["input"] == [
        _USER_ITEM,
        {
            "type": "reasoning",
            "id": "rs_good",
            "summary": [{"type": "summary_text", "text": "Plan"}],
            "encrypted_content": "encrypted",
        },
        {
            "call_id": "shared",
            "id": "fc_first",
            "type": "function_call",
            "name": "first",
            "arguments": "{}",
        },
        {
            "call_id": "shared",
            "type": "function_call_output",
            "output": "first output",
        },
        {
            "call_id": "shared",
            "type": "function_call",
            "name": "second",
            "arguments": "{}",
        },
        {
            "call_id": "shared",
            "type": "function_call_output",
            "output": "second output",
        },
    ]


def test_idless_reasoning_with_payload_degrades_function_group() -> None:
    prepared = _prepare(
        [
            Message(
                "assistant",
                [_reasoning(None), _function_call("call_1", "lookup", fc_id="fc_live")],
            )
        ]
    )

    assert prepared["input"] == [
        _USER_ITEM,
        {
            "call_id": "call_1",
            "type": "function_call",
            "name": "lookup",
            "arguments": "{}",
        },
    ]


def _hosted_content(shape: str) -> Content:
    match shape:
        case "code_call":
            return Content.from_code_interpreter_tool_call(
                call_id="code_1",
                inputs=[Content.from_text("print(1)")],
            )
        case "code_result":
            return Content.from_code_interpreter_tool_result(call_id="code_1")
        case "search_call":
            return Content.from_search_tool_call(
                "search_1",
                tool_name="web_search",
                arguments={"query": "cats"},
            )
        case "search_result":
            return Content.from_search_tool_result(
                "search_1",
                tool_name="web_search",
                result={"matches": []},
            )
        case "image_call":
            return Content.from_image_generation_tool_call(image_id="image_1")
        case "image_result":
            return Content.from_image_generation_tool_result(image_id="image_1", outputs=[])
        case "shell_call":
            return Content.from_shell_tool_call(call_id="shell_1", commands=["pwd"])
        case "shell_result":
            return Content.from_shell_tool_result(call_id="shell_1", outputs=[])
        case _:
            raise AssertionError(f"Unknown hosted content shape: {shape}")


@pytest.mark.parametrize(
    "shape",
    [
        "code_call",
        "code_result",
        "search_call",
        "search_result",
        "image_call",
        "image_result",
        "shell_call",
        "shell_result",
    ],
)
def test_non_replayable_hosted_shapes_degrade_with_reasoning(shape: str) -> None:
    prepared = _prepare([Message("assistant", [_reasoning(), _hosted_content(shape)])])

    assert prepared["input"] == [_USER_ITEM]


@pytest.mark.parametrize("item_type", [None, "custom_tool_call"])
def test_informational_function_calls_degrade_by_semantic_field(item_type: str | None) -> None:
    additional_properties = {"item_type": item_type} if item_type else None
    informational_call = Content.from_function_call(
        "hosted_1",
        "provider_tool",
        arguments="{}",
        informational_only=True,
        additional_properties=additional_properties,
    )

    prepared = _prepare(
        [
            Message("assistant", [_reasoning(), informational_call]),
            Message("tool", [Content.from_function_result("hosted_1", result="provider output")]),
        ]
    )

    assert prepared["input"] == [_USER_ITEM]


@pytest.mark.parametrize(
    "shell_marker",
    [OPENAI_SHELL_OUTPUT_TYPE_SHELL_CALL, OPENAI_SHELL_OUTPUT_TYPE_LOCAL_SHELL_CALL],
)
def test_marker_bearing_local_shell_groups_degrade(shell_marker: str) -> None:
    marker = {OPENAI_SHELL_OUTPUT_TYPE_KEY: shell_marker}
    prepared = _prepare(
        [
            Message(
                "assistant",
                [
                    _reasoning(),
                    Content.from_function_call(
                        "shell_1",
                        "bash",
                        arguments='{"command":"pwd"}',
                        additional_properties=marker,
                    ),
                ],
            ),
            Message(
                "tool",
                [
                    Content.from_function_result(
                        "shell_1",
                        result="ok",
                        additional_properties=marker,
                    )
                ],
            ),
        ]
    )

    assert prepared["input"] == [_USER_ITEM]


@pytest.mark.integration
@pytest.mark.skip(reason="Requires a live reasoning model and shell tool endpoint")
def test_local_shell_reasoning_replay_live_validity_gate() -> None:
    """Keep local-shell reasoning replay disabled until this live contract passes."""


def test_duplicate_message_and_group_ids_do_not_merge_positional_twins() -> None:
    messages = [
        Message(
            "assistant",
            [_reasoning("rs_good", text="First"), _function_call("call_1", "first", fc_id="fc_first")],
            message_id="duplicate",
        ),
        Message("tool", [Content.from_function_result("call_1", result="one")], message_id="result_1"),
        Message(
            "assistant",
            [
                _reasoning("rs_bad", text="Second", encrypted_content=None),
                _function_call("call_2", "second", fc_id="fc_second"),
            ],
            message_id="duplicate",
        ),
        Message("tool", [Content.from_function_result("call_2", result="two")], message_id="result_2"),
    ]
    annotate_message_groups(messages, force_reannotate=True)
    first_group_id = messages[0].additional_properties["_group"]["id"]
    second_group_id = messages[2].additional_properties["_group"]["id"]
    assert first_group_id == second_group_id

    prepared = _prepare(messages)

    assert prepared["input"] == [
        _USER_ITEM,
        {
            "type": "reasoning",
            "id": "rs_good",
            "summary": [{"type": "summary_text", "text": "First"}],
            "encrypted_content": "encrypted",
        },
        {
            "call_id": "call_1",
            "id": "fc_first",
            "type": "function_call",
            "name": "first",
            "arguments": "{}",
        },
        {
            "call_id": "call_1",
            "type": "function_call_output",
            "output": "one",
        },
        {
            "call_id": "call_2",
            "type": "function_call",
            "name": "second",
            "arguments": "{}",
        },
        {
            "call_id": "call_2",
            "type": "function_call_output",
            "output": "two",
        },
    ]


def test_annotated_tool_group_reasoning_only_survivor_is_dropped() -> None:
    complete_group = [
        Message("assistant", [_reasoning()]),
        Message("assistant", [_function_call("call_1", "lookup", fc_id="fc_live")]),
    ]
    annotate_message_groups(complete_group, force_reannotate=True)

    prepared = _prepare([complete_group[0]])

    assert prepared["input"] == [_USER_ITEM]


def test_annotated_reasoning_message_without_tool_calls_still_replays() -> None:
    reasoning_message = Message(
        "assistant",
        [_reasoning(), Content.from_text("Answer")],
    )
    annotate_message_groups([reasoning_message], force_reannotate=True)

    prepared = _prepare([reasoning_message])

    assert prepared["input"][1] == {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "Plan"}],
        "encrypted_content": "encrypted",
    }
    assert prepared["input"][2] == {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Answer", "annotations": []}],
    }


@pytest.mark.parametrize(
    "result",
    [
        Content.from_function_result("orphan", result="discarded"),
        _mcp_result("orphan", "discarded"),
    ],
    ids=["function", "mcp"],
)
def test_result_only_reasoning_groups_drop_orphan_outputs(result: Content) -> None:
    prepared = _prepare(
        [
            Message("assistant", [_reasoning()]),
            Message("tool", [result]),
        ]
    )

    assert prepared["input"] == [_USER_ITEM]


@pytest.mark.parametrize("second_call_id", ["mcp_2", "mcp_1"], ids=["distinct-id", "reused-id"])
def test_degraded_later_mcp_sibling_preserves_earlier_owned_result(second_call_id: str) -> None:
    messages = [
        Message("assistant", [_reasoning("rs_valid"), _mcp_call("mcp_1", "first")]),
        Message("assistant", [_mcp_call(second_call_id, "second")]),
        Message(
            "tool",
            [
                _mcp_result("mcp_1", "first output"),
                _mcp_result(second_call_id, "second output"),
            ],
        ),
    ]
    annotate_message_groups(messages)

    prepared = _prepare(messages)

    assert prepared["input"] == [
        _USER_ITEM,
        {
            "type": "reasoning",
            "id": "rs_valid",
            "summary": [{"type": "summary_text", "text": "Plan"}],
            "encrypted_content": "encrypted",
        },
        {
            "type": "mcp_call",
            "id": "mcp_1",
            "server_label": "remote",
            "name": "first",
            "arguments": '{"q":"cats"}',
            "output": "first output",
        },
    ]


def test_same_call_id_replayable_mcp_twins_degrade_the_later_group() -> None:
    prepared = _prepare(
        [
            Message("assistant", [_reasoning("rs_first", text="First"), _mcp_call("shared", "first")]),
            Message("tool", [_mcp_result("shared", "one")]),
            Message("assistant", [_reasoning("rs_second", text="Second"), _mcp_call("shared", "second")]),
            Message("tool", [_mcp_result("shared", "two")]),
        ]
    )

    assert prepared["input"] == [
        _USER_ITEM,
        {
            "type": "reasoning",
            "id": "rs_first",
            "summary": [{"type": "summary_text", "text": "First"}],
            "encrypted_content": "encrypted",
        },
        {
            "type": "mcp_call",
            "id": "shared",
            "server_label": "remote",
            "name": "first",
            "arguments": '{"q":"cats"}',
            "output": "one",
        },
    ]


def test_duplicate_reasoning_id_twins_do_not_merge_across_groups() -> None:
    prepared = _prepare(
        [
            Message(
                "assistant",
                [_reasoning("rs_shared", text="First"), _function_call("call_1", "first", fc_id="fc_first")],
            ),
            Message("tool", [Content.from_function_result("call_1", result="one")]),
            Message(
                "assistant",
                [_reasoning("rs_shared", text="Second"), _function_call("call_2", "second", fc_id="fc_second")],
            ),
            Message("tool", [Content.from_function_result("call_2", result="two")]),
        ]
    )

    assert prepared["input"] == [
        _USER_ITEM,
        {
            "type": "reasoning",
            "id": "rs_shared",
            "summary": [{"type": "summary_text", "text": "First"}],
            "encrypted_content": "encrypted",
        },
        {
            "call_id": "call_1",
            "id": "fc_first",
            "type": "function_call",
            "name": "first",
            "arguments": "{}",
        },
        {
            "call_id": "call_1",
            "type": "function_call_output",
            "output": "one",
        },
        {
            "call_id": "call_2",
            "type": "function_call",
            "name": "second",
            "arguments": "{}",
        },
        {
            "call_id": "call_2",
            "type": "function_call_output",
            "output": "two",
        },
    ]


def test_multi_sibling_reasoning_accepts_payload_on_one_sibling() -> None:
    private = _reasoning(text="Private", reasoning_text=True)
    summary = _reasoning(text="Visible", encrypted_content=None)
    prepared = _prepare([Message("assistant", [private, summary, _function_call("call_1", "lookup", fc_id="fc_live")])])

    assert prepared["input"] == [
        _USER_ITEM,
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "Visible"}],
            "encrypted_content": "encrypted",
            "content": [{"type": "reasoning_text", "text": "Private"}],
        },
        {
            "call_id": "call_1",
            "id": "fc_live",
            "type": "function_call",
            "name": "lookup",
            "arguments": "{}",
        },
    ]


def test_legacy_encrypted_content_payload_replays() -> None:
    prepared = _prepare(
        [
            Message(
                "assistant",
                [
                    _reasoning(legacy_payload=True),
                    _function_call("call_1", "lookup", fc_id="fc_live"),
                ],
            )
        ]
    )

    assert prepared["input"] == [
        _USER_ITEM,
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "Plan"}],
            "encrypted_content": "encrypted",
        },
        {
            "call_id": "call_1",
            "id": "fc_live",
            "type": "function_call",
            "name": "lookup",
            "arguments": "{}",
        },
    ]


def test_mapping_conversation_uses_service_storage_without_inline_reconstruction() -> None:
    prepared = _prepare(
        [
            Message(
                "assistant",
                [_reasoning(), _function_call("call_1", "lookup", fc_id="fc_live")],
            ),
            Message("tool", [Content.from_function_result("call_1", result="done")]),
        ],
        {"conversation": {"id": "conv_1"}},
    )

    assert "include" not in prepared
    assert prepared["conversation"] == {"id": "conv_1"}
    assert prepared["input"] == [
        _USER_ITEM,
        {
            "call_id": "call_1",
            "type": "function_call_output",
            "output": "done",
        },
    ]


@pytest.mark.parametrize(
    ("caller_include", "expected_include"),
    [
        (
            ["message.output_text.logprobs"],
            ["message.output_text.logprobs", "reasoning.encrypted_content"],
        ),
        (
            ["reasoning.encrypted_content", "message.output_text.logprobs"],
            ["reasoning.encrypted_content", "message.output_text.logprobs"],
        ),
    ],
)
def test_caller_include_values_are_preserved_when_encrypted_reasoning_is_appended(
    caller_include: list[str],
    expected_include: list[str],
) -> None:
    prepared = _prepare(
        [],
        {"include": caller_include},
    )

    assert prepared["include"] == expected_include


def test_stateless_request_without_caller_include_injects_encrypted_reasoning() -> None:
    prepared = _prepare([])

    assert prepared["include"] == ["reasoning.encrypted_content"]


def test_explicit_empty_caller_include_is_omitted() -> None:
    prepared = _prepare([], {"include": []})

    assert "include" not in prepared


def test_service_storage_string_marker_keeps_existing_strip_behavior() -> None:
    prepared = _prepare(
        [
            Message(
                "assistant",
                [_reasoning(), _function_call("call_1", "lookup", fc_id="fc_live")],
            ),
            Message("tool", [Content.from_function_result("call_1", result="done")]),
        ],
        {"previous_response_id": "resp_1"},
    )

    assert "include" not in prepared
    assert prepared["input"] == [
        _USER_ITEM,
        {
            "call_id": "call_1",
            "type": "function_call_output",
            "output": "done",
        },
    ]


def test_degraded_groups_emit_one_aggregate_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="chrys.service.llm.openai_responses"):
        _prepare(
            [
                Message(
                    "assistant",
                    [_reasoning("rs_1", encrypted_content=None), _function_call("call_1", "first")],
                ),
                Message(
                    "assistant",
                    [_reasoning("rs_2", encrypted_content=None), _function_call("call_2", "second")],
                ),
            ]
        )

    warnings = [record for record in caplog.records if record.message.startswith("Degraded stateless reasoning replay")]
    assert len(warnings) == 1
    assert "rs_1" in warnings[0].message
    assert "rs_2" in warnings[0].message
    assert "call_1" in warnings[0].message
    assert "call_2" in warnings[0].message


async def test_encrypted_reasoning_capability_rejection_is_terminal() -> None:
    async_client = AsyncOpenAI(api_key="sk-fake")
    client = _client(async_client)
    create = AsyncMock()
    service_error = BadRequestError(
        message="Encrypted reasoning is not supported",
        response=MagicMock(),
        body={
            "error": {
                "code": "invalid_request",
                "message": "Encrypted reasoning is not supported",
            }
        },
    )
    service_error.code = "invalid_request"
    create.side_effect = service_error

    with (
        patch.object(async_client.responses.with_raw_response, "create", new=create),
        pytest.raises(ChatClientException, match="Encrypted reasoning is not supported"),
    ):
        await client.get_response(
            [Message("user", ["Start"])],
            options={"store": False},
        )

    create.assert_awaited_once()
    assert create.await_args is not None
    assert create.await_args.kwargs["include"] == ["reasoning.encrypted_content"]


@pytest.mark.parametrize(
    "result",
    [
        Content.from_function_result("orphan", result="discarded"),
        _mcp_result("orphan", "discarded"),
    ],
    ids=["function", "mcp"],
)
def test_assistant_role_orphan_results_degrade_with_their_reasoning(result: Content) -> None:
    prepared = _prepare(
        [
            Message("assistant", [_reasoning()]),
            Message("assistant", [result]),
        ]
    )

    assert prepared["input"] == [_USER_ITEM]


def test_assistant_role_mcp_result_coalesces_into_its_reasoning_call_group() -> None:
    prepared = _prepare(
        [
            Message("assistant", [_reasoning(), _mcp_call("independent")]),
            Message("assistant", [_mcp_result("independent", "independent output")]),
        ]
    )

    assert prepared["input"] == [
        _USER_ITEM,
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "Plan"}],
            "encrypted_content": "encrypted",
        },
        {
            "type": "mcp_call",
            "id": "independent",
            "server_label": "remote",
            "name": "search",
            "arguments": '{"q":"cats"}',
            "output": "independent output",
        },
    ]


def test_assistant_role_mcp_result_coalesces_without_reasoning() -> None:
    prepared = _prepare(
        [
            Message("assistant", [_mcp_call("independent")]),
            Message("assistant", [_mcp_result("independent", "independent output")]),
        ]
    )

    assert prepared["input"] == [
        _USER_ITEM,
        {
            "type": "mcp_call",
            "id": "independent",
            "server_label": "remote",
            "name": "search",
            "arguments": '{"q":"cats"}',
            "output": "independent output",
        },
    ]


def test_plain_mcp_group_yields_to_earlier_reasoning_twin() -> None:
    prepared = _prepare(
        [
            Message("assistant", [_reasoning("rs_first", text="First"), _mcp_call("shared", "first")]),
            Message("tool", [_mcp_result("shared", "one")]),
            Message("assistant", [_mcp_call("shared", "second")]),
            Message("tool", [_mcp_result("shared", "two")]),
        ]
    )

    assert prepared["input"] == [
        _USER_ITEM,
        {
            "type": "reasoning",
            "id": "rs_first",
            "summary": [{"type": "summary_text", "text": "First"}],
            "encrypted_content": "encrypted",
        },
        {
            "type": "mcp_call",
            "id": "shared",
            "server_label": "remote",
            "name": "first",
            "arguments": '{"q":"cats"}',
            "output": "one",
        },
    ]


def test_reasoning_mcp_group_yields_to_later_function_only_wire_id() -> None:
    prepared = _prepare(
        [
            Message("assistant", [_reasoning(), _mcp_call("fc_call_9")]),
            Message("tool", [_mcp_result("fc_call_9", "one")]),
            Message("assistant", [_function_call("call_9", "plain")]),
            Message("tool", [Content.from_function_result("call_9", result="two")]),
        ]
    )

    assert prepared["input"] == [
        _USER_ITEM,
        {
            "call_id": "call_9",
            "id": "fc_call_9",
            "type": "function_call",
            "name": "plain",
            "arguments": "{}",
        },
        {
            "call_id": "call_9",
            "type": "function_call_output",
            "output": "two",
        },
    ]


def test_same_fc_wire_id_replayable_function_twins_degrade_the_later_group() -> None:
    prepared = _prepare(
        [
            Message(
                "assistant",
                [_reasoning("rs_first", text="First"), _function_call("call_1", "first", fc_id="fc_shared")],
            ),
            Message("tool", [Content.from_function_result("call_1", result="one")]),
            Message(
                "assistant",
                [_reasoning("rs_second", text="Second"), _function_call("call_2", "second", fc_id="fc_shared")],
            ),
            Message("tool", [Content.from_function_result("call_2", result="two")]),
        ]
    )

    assert prepared["input"] == [
        _USER_ITEM,
        {
            "type": "reasoning",
            "id": "rs_first",
            "summary": [{"type": "summary_text", "text": "First"}],
            "encrypted_content": "encrypted",
        },
        {
            "call_id": "call_1",
            "id": "fc_shared",
            "type": "function_call",
            "name": "first",
            "arguments": "{}",
        },
        {
            "call_id": "call_1",
            "type": "function_call_output",
            "output": "one",
        },
        {
            "call_id": "call_2",
            "type": "function_call",
            "name": "second",
            "arguments": "{}",
        },
        {
            "call_id": "call_2",
            "type": "function_call_output",
            "output": "two",
        },
    ]


def test_reused_call_id_reasoning_twins_degrade_the_later_group() -> None:
    prepared = _prepare(
        [
            Message("assistant", [_reasoning("rs_first", text="First"), _function_call("call_1", "first")]),
            Message("tool", [Content.from_function_result("call_1", result="one")]),
            Message("assistant", [_reasoning("rs_second", text="Second"), _function_call("call_1", "second")]),
            Message("tool", [Content.from_function_result("call_1", result="two")]),
        ]
    )

    assert prepared["input"] == [
        _USER_ITEM,
        {
            "type": "reasoning",
            "id": "rs_first",
            "summary": [{"type": "summary_text", "text": "First"}],
            "encrypted_content": "encrypted",
        },
        {
            "call_id": "call_1",
            "id": "fc_call_1",
            "type": "function_call",
            "name": "first",
            "arguments": "{}",
        },
        {
            "call_id": "call_1",
            "type": "function_call_output",
            "output": "one",
        },
        {
            "call_id": "call_1",
            "type": "function_call",
            "name": "second",
            "arguments": "{}",
        },
        {
            "call_id": "call_1",
            "type": "function_call_output",
            "output": "two",
        },
    ]


def test_plain_function_group_wire_id_is_reserved_before_reasoning_twin() -> None:
    prepared = _prepare(
        [
            Message("assistant", [_function_call("call_1", "plain")]),
            Message("tool", [Content.from_function_result("call_1", result="one")]),
            Message("assistant", [_reasoning(), _function_call("call_1", "reasoned")]),
            Message("tool", [Content.from_function_result("call_1", result="two")]),
        ]
    )

    assert prepared["input"] == [
        _USER_ITEM,
        {
            "call_id": "call_1",
            "id": "fc_call_1",
            "type": "function_call",
            "name": "plain",
            "arguments": "{}",
        },
        {
            "call_id": "call_1",
            "type": "function_call_output",
            "output": "one",
        },
        {
            "call_id": "call_1",
            "type": "function_call",
            "name": "reasoned",
            "arguments": "{}",
        },
        {
            "call_id": "call_1",
            "type": "function_call_output",
            "output": "two",
        },
    ]


def test_annotated_assistant_role_mcp_result_still_joins_its_call_group() -> None:
    messages = [
        Message("user", ["Start"]),
        Message("assistant", [_reasoning(), _mcp_call("independent")]),
        Message("assistant", [_mcp_result("independent", "independent output")]),
    ]
    annotate_message_groups(messages)

    prepared = _build(messages, {"store": False})

    assert prepared["input"] == [
        _USER_ITEM,
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "Plan"}],
            "encrypted_content": "encrypted",
        },
        {
            "type": "mcp_call",
            "id": "independent",
            "server_label": "remote",
            "name": "search",
            "arguments": '{"q":"cats"}',
            "output": "independent output",
        },
    ]


def test_shared_reasoning_content_instance_emits_its_item_once() -> None:
    shared = _reasoning()
    prepared = _prepare(
        [
            Message("assistant", [shared]),
            Message("assistant", [shared, _function_call("call_1", "lookup", fc_id="fc_live")]),
            Message("tool", [Content.from_function_result("call_1", result="done")]),
        ]
    )

    assert prepared["input"] == [
        _USER_ITEM,
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "Plan"}],
            "encrypted_content": "encrypted",
        },
        {
            "call_id": "call_1",
            "id": "fc_live",
            "type": "function_call",
            "name": "lookup",
            "arguments": "{}",
        },
        {
            "call_id": "call_1",
            "type": "function_call_output",
            "output": "done",
        },
    ]


def test_partition_splits_followers_with_conflicting_annotation_signatures() -> None:
    def _annotated(message: Message, group_id: str) -> Message:
        message.additional_properties[GROUP_ANNOTATION_KEY] = {
            GROUP_ID_KEY: group_id,
            GROUP_KIND_KEY: "tool_call",
            GROUP_INDEX_KEY: 0,
            GROUP_HAS_REASONING_KEY: True,
        }
        return message

    call_message = _annotated(Message("assistant", [_function_call("call_1", "lookup")]), "group_call")
    foreign_result = _annotated(Message("tool", [Content.from_function_result("call_1", result="x")]), "group_other")
    owned_result = _annotated(Message("tool", [Content.from_function_result("call_1", result="x")]), "group_call")

    split = partition_groups([call_message, foreign_result])
    joined = partition_groups([call_message, owned_result])

    assert [group.messages for group in split] == [(call_message,), (foreign_result,)]
    assert [group.messages for group in joined] == [(call_message, owned_result)]


def test_partition_assistant_result_with_conflicting_annotation_splits_unless_it_answers_the_group() -> None:
    """The assistant-role structural join is scoped to results answering this
    group's calls: an owned result keeps joining across a conflicting
    annotation (the annotator assigns results a span of their own), while a
    stray result for a call the group never made splits off instead of
    attaching to — and later degrading — the valid group."""

    def _annotated(message: Message, group_id: str) -> Message:
        message.additional_properties[GROUP_ANNOTATION_KEY] = {
            GROUP_ID_KEY: group_id,
            GROUP_KIND_KEY: "tool_call",
            GROUP_INDEX_KEY: 0,
            GROUP_HAS_REASONING_KEY: True,
        }
        return message

    call_message = _annotated(Message("assistant", [_reasoning(), _mcp_call("call_a")]), "group_a")
    owned_result = _annotated(Message("assistant", [_mcp_result("call_a", "owned output")]), "group_b")
    orphan_result = _annotated(Message("assistant", [_mcp_result("call_b", "stray output")]), "group_b")

    joined = partition_groups([call_message, owned_result])
    split = partition_groups([call_message, orphan_result])

    assert [group.messages for group in joined] == [(call_message, owned_result)]
    assert [group.messages for group in split] == [(call_message,), (orphan_result,)]


def test_partition_marker_message_owned_result_splits_from_call_group() -> None:
    """A history marker is exchange chrome, not the exchange's output run: a
    marker-stamped assistant message carrying the call's result starts its own
    group instead of joining — and later degrading with — the call's group."""
    call_message = Message("assistant", [_function_call("call_1", "lookup")])
    marker_result = Message("assistant", [Content.from_function_result("call_1", result="late")])
    marker_result.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN

    groups = partition_groups([call_message, marker_result])

    assert [group.messages for group in groups] == [(call_message,), (marker_result,)]


def test_partition_image_result_with_conflicting_annotation_joins_owned_call() -> None:
    """Image-generation ownership uses its namespaced image_id pairing key,
    so a result-only assistant message stays with its call even when the
    compaction annotator assigned the result a different span."""

    def _annotated(message: Message, group_id: str) -> Message:
        message.additional_properties[GROUP_ANNOTATION_KEY] = {
            GROUP_ID_KEY: group_id,
            GROUP_KIND_KEY: "tool_call",
            GROUP_INDEX_KEY: 0,
            GROUP_HAS_REASONING_KEY: True,
        }
        return message

    image_call = _annotated(
        Message("assistant", [Content.from_image_generation_tool_call(image_id="img_1")]),
        "group_call",
    )
    image_result = _annotated(
        Message("assistant", [Content.from_image_generation_tool_result(image_id="img_1")]),
        "group_other",
    )

    groups = partition_groups([image_call, image_result])

    assert [group.messages for group in groups] == [(image_call, image_result)]


def test_partition_sibling_call_group_carries_shared_reasoning_annotation() -> None:
    """A run of call-carrying assistant siblings answered by one shared result
    block rides ONE reasoning turn: the second sibling's replay group must
    read as reasoning-bearing instead of degrading separately from the
    reasoning prefix it belongs to."""
    first_call = Message("assistant", [_reasoning(), _function_call("call_a", "lookup")])
    second_call = Message("assistant", [_function_call("call_b", "search")])
    results = Message(
        "tool",
        [
            Content.from_function_result("call_a", result="ra"),
            Content.from_function_result("call_b", result="rb"),
        ],
    )
    messages = [first_call, second_call, results]
    annotate_message_groups(messages)

    groups = partition_groups(messages)

    second_call_group = next(group for group in groups if second_call in group.messages)
    assert second_call_group.annotated_reasoning_tool_call is True


def test_annotate_keeps_user_role_result_message_out_of_reasoning_group() -> None:
    """A user-role message carrying result contents is transcript input, not
    the reasoning turn's output run: annotation must not pull it into the
    tool-call group, and its replay group must not read as reasoning-bearing."""
    call_message = Message("assistant", [_reasoning(), _function_call("call_a", "lookup")])
    user_result = Message("user", [Content.from_function_result("call_a", result="ra")])
    messages = [call_message, user_result]
    annotate_message_groups(messages)

    annotation = user_result.additional_properties.get(GROUP_ANNOTATION_KEY)
    assert annotation is None or annotation[GROUP_KIND_KEY] != "tool_call"

    groups = partition_groups(messages)
    user_group = next(group for group in groups if user_result in group.messages)
    assert user_group.annotated_reasoning_tool_call is False


def test_orphan_assistant_result_degrades_alone_and_valid_group_replays_intact() -> None:
    """The stray result's group degrades by itself; the reasoning tool-call
    group it used to attach to keeps its encrypted reasoning on the wire."""

    def _annotated(message: Message, group_id: str) -> Message:
        message.additional_properties[GROUP_ANNOTATION_KEY] = {
            GROUP_ID_KEY: group_id,
            GROUP_KIND_KEY: "tool_call",
            GROUP_INDEX_KEY: 0,
            GROUP_HAS_REASONING_KEY: True,
        }
        return message

    messages = [
        Message("user", ["Start"]),
        _annotated(Message("assistant", [_reasoning(), _mcp_call("call_a")]), "group_a"),
        _annotated(Message("assistant", [_mcp_result("call_b", "stray output")]), "group_b"),
    ]

    prepared = _build(messages, {"store": False})

    assert prepared["input"] == [
        _USER_ITEM,
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "Plan"}],
            "encrypted_content": "encrypted",
        },
        {
            "type": "mcp_call",
            "id": "call_a",
            "server_label": "remote",
            "name": "search",
            "arguments": '{"q":"cats"}',
        },
    ]


def test_same_id_snapshot_then_final_payload_replays_final() -> None:
    """`added` stamps a snapshot payload; `done` delivers the terminal one.

    The payload-only shape merges in the kernel (same-id later-wins), and the
    replay selection must never resurrect the stale snapshot."""
    snapshot = SimpleNamespace(type="reasoning", id="rs_1", content=None, summary=[], encrypted_content="snapshot-A")
    final = SimpleNamespace(type="reasoning", id="rs_1", content=None, summary=[], encrypted_content="final-B")
    updates = [
        _fresh_update(SimpleNamespace(type="response.output_item.added", item=snapshot, output_index=0)),
        _fresh_update(SimpleNamespace(type="response.output_item.done", item=final, output_index=0)),
    ]

    prepared = _prepare(list(ChatResponse.from_updates(updates).messages))

    reasoning_items = [item for item in prepared["input"] if item.get("type") == "reasoning"]
    assert len(reasoning_items) == 1
    assert reasoning_items[0]["encrypted_content"] == "final-B"


def test_multi_sibling_snapshot_then_final_payload_replays_final() -> None:
    """Private-text + summary siblings cannot merge; `done`'s terminal payload
    lands on the later sibling and must win over the snapshot on the first."""
    snapshot = SimpleNamespace(
        type="reasoning",
        id="rs_1",
        content=[SimpleNamespace(text="Private")],
        summary=[SimpleNamespace(text="Summary")],
        encrypted_content="snapshot-A",
    )
    final = SimpleNamespace(type="reasoning", id="rs_1", content=None, summary=[], encrypted_content="final-B")
    updates = [
        _fresh_update(SimpleNamespace(type="response.output_item.added", item=snapshot, output_index=0)),
        _fresh_update(SimpleNamespace(type="response.output_item.done", item=final, output_index=0)),
    ]

    prepared = _prepare(list(ChatResponse.from_updates(updates).messages))

    reasoning_items = [item for item in prepared["input"] if item.get("type") == "reasoning"]
    assert len(reasoning_items) == 1
    assert reasoning_items[0]["encrypted_content"] == "final-B"
    assert reasoning_items[0]["summary"] == [{"type": "summary_text", "text": "Summary"}]
    assert reasoning_items[0]["content"] == [{"type": "reasoning_text", "text": "Private"}]


def test_format_stamped_idless_reasoning_degrades_function_group() -> None:
    """GLM/DeepSeek dialect reasoning is id-less foreign state on the Responses
    path; the group degrades exactly like other id-less reasoning."""
    glm_reasoning = Content.from_text_reasoning(
        text="GLM chain of thought",
        additional_properties={"openai_reasoning_format": "reasoning_content"},
    )
    prepared = _prepare([Message("assistant", [glm_reasoning, _function_call("call_1", "lookup", fc_id="fc_live")])])

    assert prepared["input"] == [
        _USER_ITEM,
        {
            "call_id": "call_1",
            "type": "function_call",
            "name": "lookup",
            "arguments": "{}",
        },
    ]


def test_streamed_snapshot_reasoning_replays_content_and_summary() -> None:
    snapshot = SimpleNamespace(
        type="reasoning",
        id="rs_snapshot",
        content=[SimpleNamespace(text="Private reasoning")],
        summary=[SimpleNamespace(text="Visible summary")],
        encrypted_content="encrypted-snapshot",
    )
    update = _fresh_update(SimpleNamespace(type="response.output_item.added", item=snapshot, output_index=0))

    prepared = _prepare(list(ChatResponse.from_updates([update]).messages))

    assert prepared["input"] == [
        _USER_ITEM,
        {
            "type": "reasoning",
            "id": "rs_snapshot",
            "summary": [{"type": "summary_text", "text": "Visible summary"}],
            "encrypted_content": "encrypted-snapshot",
            "content": [{"type": "reasoning_text", "text": "Private reasoning"}],
        },
    ]

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Provider-shape coverage for physically scoped Phase-4 side-call slices."""

from __future__ import annotations

from typing import Any

from chrys.kernel import Content, Message, annotate_message_groups
from chrys.service.context.compaction.scoped import (
    DEGRADED_SCOPED_PREAMBLE,
    SLICE_TRUNCATION_MARKER,
    build_scoped_group_timeline,
    prepare_scoped_slice,
)
from chrys.service.llm.anthropic_messages.history import encode_messages
from chrys.service.llm.chat_completions import history as chat_history
from chrys.service.llm.chat_completions.client import DEEPSEEK, OPENAI
from chrys.service.llm.openai_responses.client import OPENAI_RESPONSES
from chrys.service.llm.openai_responses.history import (
    OPENAI_SHELL_OUTPUT_TYPE_KEY,
    OPENAI_SHELL_OUTPUT_TYPE_SHELL_CALL,
)
from chrys.service.llm.openai_responses.replay import encode_input
from tests.support.images import image_bytes


class _LenTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(text)


def _call(call_id: str, name: str, fc_id: str) -> Content:
    return Content.from_function_call(
        call_id,
        name,
        arguments={"path": f"{name}.txt"},
        additional_properties={"fc_id": fc_id, "status": "completed"},
    )


def _representative_slice() -> tuple[Message, ...]:
    runtime_reminder = "<system-reminder>\n[Runtime Environment] cwd=/tmp\n</system-reminder>"
    followup_reminder = "<system-reminder>\n[Runtime Environment] shell=zsh\n</system-reminder>"
    image_result = Content.from_function_result(
        "reused",
        result=[
            Content.from_text("start-" + "x" * 12_000 + "-finish"),
            Content.from_data(image_bytes("PNG"), "image/png"),
        ],
    )
    shell_result = Content.from_function_result(
        "shell",
        result={"stdout": "shell-start-" + "s" * 10_000 + "-shell-finish", "exit_code": 0},
        additional_properties={OPENAI_SHELL_OUTPUT_TYPE_KEY: OPENAI_SHELL_OUTPUT_TYPE_SHELL_CALL},
    )
    live_messages = [
        Message("user", [runtime_reminder]),
        Message(
            "assistant",
            [
                _call("reused", "view_image", "fc_provider_first"),
                _call("parallel", "read_file", "fc_provider_parallel"),
            ],
        ),
        Message(
            "tool",
            [image_result, Content.from_function_result("parallel", result="parallel result")],
        ),
        Message("user", ["keep the provider fields", followup_reminder]),
        Message(
            "assistant",
            [
                Content.from_text_reasoning(text="signed thought", protected_data='{"signature":"signature-1"}'),
                _call("shell", "shell", "fc_provider_shell"),
            ],
        ),
        Message("tool", [shell_result]),
        Message("assistant", [_call("reused", "read_file", "fc_provider_second")]),
        Message("tool", [Content.from_function_result("reused", result="newest result")]),
    ]
    annotate_message_groups(live_messages, force_reannotate=True)
    timeline = build_scoped_group_timeline(
        live_messages,
        span_start=0,
        span_end=len(live_messages),
        degraded=False,
    )
    assert timeline.degraded_opener is True
    full = prepare_scoped_slice(timeline.groups, tokenizer=_LenTokenizer(), slice_budget=100_000)
    assert full is not None
    clipped = prepare_scoped_slice(
        timeline.groups,
        tokenizer=_LenTokenizer(),
        slice_budget=full.estimated_tokens - 15_000,
    )
    assert clipped is not None
    assert clipped.clipped_results >= 1
    assert clipped.messages[0].text == DEGRADED_SCOPED_PREAMBLE
    assert all(followup_reminder not in message.text for message in clipped.messages)
    assert any(
        SLICE_TRUNCATION_MARKER in (item.text or "")
        for message in clipped.messages
        for content in message.contents
        for item in (content.items or [])
        if content.type == "function_result" and item.type == "text"
    )
    assert "-finish" in (image_result.items or [])[0].text
    assert "-shell-finish" in str(shell_result.result)
    return clipped.messages


def _assert_nonempty_content_arrays(prepared: list[dict[str, Any]]) -> None:
    for item in prepared:
        content = item.get("content")
        if isinstance(content, list):
            assert content


def test_anthropic_scoped_slice_has_strict_tool_use_result_groups() -> None:
    prepared = encode_messages(_representative_slice())

    assert prepared[0] == {"role": "user", "content": [{"type": "text", "text": DEGRADED_SCOPED_PREAMBLE}]}
    _assert_nonempty_content_arrays(prepared)
    call_groups = [index for index, item in enumerate(prepared) if item["role"] == "assistant"]
    assert len(call_groups) == 3
    assert [block["id"] for block in prepared[call_groups[0]]["content"] if block["type"] == "tool_use"] == [
        "reused",
        "parallel",
    ]
    assert [
        block["tool_use_id"] for block in prepared[call_groups[0] + 1]["content"] if block["type"] == "tool_result"
    ] == ["reused", "parallel"]
    thinking = prepared[call_groups[1]]["content"][0]
    assert thinking == {
        "type": "thinking",
        "thinking": "signed thought",
        "signature": '{"signature":"signature-1"}',
    }
    assert prepared[call_groups[1] + 1]["content"][0]["tool_use_id"] == "shell"
    assert prepared[call_groups[2] + 1]["content"][0]["tool_use_id"] == "reused"


def test_openai_chat_scoped_slice_has_parallel_calls_images_and_reused_ids() -> None:
    prepared = chat_history.encode_messages(_representative_slice(), variant=OPENAI)

    assert prepared[0] == {"role": "user", "content": DEGRADED_SCOPED_PREAMBLE}
    _assert_nonempty_content_arrays(prepared)
    assistant_indices = [index for index, item in enumerate(prepared) if item["role"] == "assistant"]
    assert len(prepared[assistant_indices[0]]["tool_calls"]) == 2
    first_results = prepared[assistant_indices[0] + 1 : assistant_indices[0] + 3]
    assert [item["tool_call_id"] for item in first_results] == ["reused", "parallel"]
    assert any(
        item.get("role") == "user"
        and isinstance(item.get("content"), list)
        and any(part.get("type") == "image_url" for part in item["content"])
        for item in prepared
    )
    assert prepared[assistant_indices[1] + 1]["tool_call_id"] == "shell"
    assert prepared[assistant_indices[2] + 1]["tool_call_id"] == "reused"


def test_deepseek_scoped_slice_has_chat_completion_shape() -> None:
    prepared = chat_history.encode_messages(_representative_slice(), variant=DEEPSEEK)

    assert prepared[0]["role"] == "user"
    _assert_nonempty_content_arrays(prepared)
    assistant_messages = [item for item in prepared if item["role"] == "assistant"]
    assert [call["id"] for call in assistant_messages[0]["tool_calls"]] == ["reused", "parallel"]
    assert assistant_messages[1]["tool_calls"][0]["id"] == "shell"
    assert assistant_messages[2]["tool_calls"][0]["id"] == "reused"
    assert sum(item.get("role") == "tool" for item in prepared) == 4


def test_responses_store_false_preserves_replayable_fc_ids_and_pairing() -> None:
    prepared = encode_input(_representative_slice(), service_side=False, variant=OPENAI_RESPONSES)

    assert prepared[0]["type"] == "message"
    assert prepared[0]["role"] == "user"
    _assert_nonempty_content_arrays(prepared)
    calls = [item for item in prepared if item["type"] == "function_call"]
    results = [item for item in prepared if item["type"] == "function_call_output"]
    assert not any(item.get("type") == "reasoning" for item in prepared)
    assert [item["call_id"] for item in calls] == ["reused", "parallel", "shell", "reused"]
    assert [item.get("id") for item in calls] == [
        "fc_provider_first",
        "fc_provider_parallel",
        None,
        "fc_provider_second",
    ]
    assert [item["call_id"] for item in results] == ["reused", "parallel", "reused"]
    assert isinstance(results[0]["output"], list)
    assert any(part["type"] == "input_image" for part in results[0]["output"])
    assert not any(item["type"] == OPENAI_SHELL_OUTPUT_TYPE_SHELL_CALL for item in prepared)


def _foreign_responses_reasoning_history() -> list[Message]:
    reasoning_siblings = [
        Content.from_text_reasoning(
            id="rs_1",
            text="Private reasoning",
            protected_data="gAAAAA-opaque-encrypted",
            additional_properties={"reasoning_text": True},
        ),
        Content.from_text_reasoning(id="rs_1", text="Visible summary", protected_data="gAAAAA-opaque-encrypted"),
    ]
    return [
        Message("user", ["Start"]),
        Message("assistant", [*reasoning_siblings, Content.from_function_call("call_1", "lookup", arguments="{}")]),
        Message("tool", [Content.from_function_result("call_1", result="done")]),
        Message("user", ["Continue"]),
    ]


def test_openai_chat_drops_foreign_encrypted_reasoning_instead_of_crashing() -> None:
    prepared = chat_history.encode_messages(_foreign_responses_reasoning_history(), variant=OPENAI)

    assert all("reasoning_details" not in message for message in prepared)


def test_deepseek_drops_foreign_encrypted_reasoning_instead_of_crashing() -> None:
    prepared = chat_history.encode_messages(_foreign_responses_reasoning_history(), variant=DEEPSEEK)

    assert all("reasoning_details" not in message for message in prepared)
    # The foreign reasoning is not replayed; thinking mode still gets the empty field it requires.
    assert [message.get("reasoning_content") for message in prepared] == [None, "", None, None]


def test_anthropic_drops_foreign_responses_reasoning_and_keeps_own_thinking() -> None:
    history = _foreign_responses_reasoning_history()
    history[1].contents.append(Content.from_text_reasoning(text="own thinking", protected_data="anthropic-signature"))

    prepared = encode_messages(history)

    thinking_blocks = [
        block
        for message in prepared
        for block in (message["content"] if isinstance(message["content"], list) else [])
        if isinstance(block, dict) and block.get("type") == "thinking"
    ]
    assert thinking_blocks == [{"type": "thinking", "thinking": "own thinking", "signature": "anthropic-signature"}]

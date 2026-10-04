# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Invariant tests for chrys.kernel.compaction.

Chrys' owned ``BaseChatClient`` now owns the live compaction seam:
``_prepare_messages_for_model_call`` -> ``apply_compaction`` -> chrys
annotation -> chrys strategy -> chrys projection.
"""

from __future__ import annotations

import gc
import json
import weakref
from io import BytesIO
from typing import TYPE_CHECKING, get_args

import pytest

from chrys.foundation.models.history_markers import ANTHROPIC_THINKING_STRIPPED_KEY, HistoryMarkerKind
from chrys.foundation.reasoning_origin import REASONING_ORIGIN_KEY, ReasoningOrigin
from chrys.foundation.text.tokenizer import MixedLanguageTokenizer
from chrys.foundation.tool_execution_stamp import EXECUTION_STAMP_KEY, build_execution_stamp
from chrys.kernel import Content, Message
from chrys.kernel import compaction as chrys_compaction

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _LenTokenizer:
    """Count characters so serialization differences produce different token estimates."""

    def count_tokens(self, text: str) -> int:
        return len(text)


def _user(text: str) -> Message:
    return Message(role="user", contents=[Content.from_text(text)])


def _system(text: str) -> Message:
    return Message(role="system", contents=[Content.from_text(text)])


def _assistant_text(text: str) -> Message:
    return Message(role="assistant", contents=[Content.from_text(text)])


def _assistant_reasoning(text: str) -> Message:
    return Message(role="assistant", contents=[Content.from_text_reasoning(text=text)])


def _assistant_tool_call(call_id: str, name: str) -> Message:
    return Message(role="assistant", contents=[Content.from_function_call(call_id, name, arguments={})])


def _assistant_reasoning_and_call(call_id: str, name: str) -> Message:
    return Message(
        role="assistant",
        contents=[
            Content.from_text_reasoning(text="thinking"),
            Content.from_function_call(call_id, name, arguments={}),
        ],
    )


def _reasoning_tool_call_group(call_id: str, name: str) -> list[Message]:
    return [
        _assistant_reasoning(f"reasoning for {name}"),
        _assistant_tool_call(call_id, name),
        _tool_result(call_id, f"result for {name}"),
    ]


def _tool_result(call_id: str, result: str) -> Message:
    return Message(role="tool", contents=[Content.from_function_result(call_id, result=result)])


def _tool_image_result(call_id: str, data: bytes = b"image") -> Message:
    return Message(
        role="tool",
        contents=[
            Content.from_function_result(
                call_id,
                result=[
                    Content.from_text("caption"),
                    Content.from_data(
                        data,
                        "image/png",
                        additional_properties={"width": 1024, "height": 512, "media_type": "image/png"},
                    ),
                ],
            )
        ],
    )


def _generated_png_bytes(size: tuple[int, int] = (512, 512)) -> bytes:
    from PIL import Image

    image = Image.effect_noise(size, 100).convert("RGB")
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _basic_conversation() -> list[Message]:
    return [
        _system("rules"),
        _user("question"),
        _assistant_tool_call("c1", "search"),
        _tool_result("c1", "found"),
        _tool_result("c1", "more"),
        _assistant_text("answer"),
        _user("follow-up"),
    ]


def _reasoning_prefix_group() -> list[Message]:
    return [
        _user("question"),
        _assistant_reasoning("step 1"),
        _assistant_reasoning("step 2"),
        _assistant_tool_call("c1", "search"),
        _tool_result("c1", "found"),
        _assistant_text("answer"),
    ]


def _reasoning_colocated_with_call() -> list[Message]:
    return [
        _user("question"),
        _assistant_reasoning_and_call("c1", "search"),
        _tool_result("c1", "found"),
    ]


def _reasoning_interleaved_in_group() -> list[Message]:
    return [
        _user("question"),
        _assistant_tool_call("c1", "search"),
        _assistant_reasoning("mid-group thought"),
        _tool_result("c1", "found"),
    ]


def _reasoning_prefix_and_trailing_reasoning() -> list[Message]:
    # Enters group_messages' reasoning-PREFIX branch, then exercises that
    # branch's INNER loop (reasoning between the call and its tool results)
    # — distinct from the non-prefix interleaved shape.
    return [
        _user("question"),
        _assistant_reasoning("plan"),
        _assistant_tool_call("c1", "search"),
        _assistant_reasoning("reflect"),
        _tool_result("c1", "found"),
    ]


def _orphan_tool_run() -> list[Message]:
    return [
        _user("question"),
        _tool_result("c0", "orphan one"),
        _tool_result("c0", "orphan two"),
        _assistant_text("answer"),
    ]


def _dangling_reasoning() -> list[Message]:
    return [
        _user("question"),
        _assistant_reasoning("never followed by a call"),
        _assistant_text("answer"),
    ]


def _empty_contents_assistant() -> list[Message]:
    return [
        _user("question"),
        Message(role="assistant", contents=[]),
        _user("again"),
    ]


def _preset_id_collision() -> list[Message]:
    first = _user("already has the id auto-assignment would pick")
    first.message_id = "msg_1"
    return [first, _user("needs an id"), _assistant_text("needs an id too")]


def _unicode_payloads() -> list[Message]:
    return [
        _user("中文与 emoji 🙂 mixed"),
        _assistant_tool_call("c1", "翻译"),
        _tool_result("c1", "résumé — naïve"),
        _assistant_text("done ✓"),
    ]


_SHAPE_FACTORIES: list[Callable[[], list[Message]]] = [
    _basic_conversation,
    _reasoning_prefix_group,
    _reasoning_colocated_with_call,
    _reasoning_interleaved_in_group,
    _reasoning_prefix_and_trailing_reasoning,
    _orphan_tool_run,
    _dangling_reasoning,
    _empty_contents_assistant,
    _preset_id_collision,
    _unicode_payloads,
]


def _assert_lists_equivalent(first: list[Message], second: list[Message]) -> None:
    assert [m.message_id for m in first] == [m.message_id for m in second]
    assert [m.additional_properties for m in first] == [m.additional_properties for m in second]


# ---------------------------------------------------------------------------
# A. Key/type pins
# ---------------------------------------------------------------------------


class TestKeyDriftPins:
    @pytest.mark.parametrize(
        "name",
        [
            "GROUP_ANNOTATION_KEY",
            "GROUP_ID_KEY",
            "GROUP_KIND_KEY",
            "GROUP_INDEX_KEY",
            "GROUP_HAS_REASONING_KEY",
            "GROUP_TOKEN_COUNT_KEY",
            "GROUP_TOKEN_ESTIMATOR_VERSION_KEY",
            "EXCLUDED_KEY",
            "EXCLUDE_REASON_KEY",
            "SUMMARY_OF_MESSAGE_IDS_KEY",
            "SUMMARY_OF_GROUP_IDS_KEY",
            "SUMMARIZED_BY_SUMMARY_ID_KEY",
        ],
    )
    def test_annotation_key_values_stay_stable(self, name: str) -> None:
        expected = {
            "GROUP_ANNOTATION_KEY": "_group",
            "GROUP_ID_KEY": "id",
            "GROUP_KIND_KEY": "kind",
            "GROUP_INDEX_KEY": "index",
            "GROUP_HAS_REASONING_KEY": "has_reasoning",
            "GROUP_TOKEN_COUNT_KEY": "token_count",
            "GROUP_TOKEN_ESTIMATOR_VERSION_KEY": "token_estimator_v",
            "EXCLUDED_KEY": "_excluded",
            "EXCLUDE_REASON_KEY": "_exclude_reason",
            "SUMMARY_OF_MESSAGE_IDS_KEY": "_summary_of_message_ids",
            "SUMMARY_OF_GROUP_IDS_KEY": "_summary_of_group_ids",
            "SUMMARIZED_BY_SUMMARY_ID_KEY": "_summarized_by_summary_id",
        }
        assert getattr(chrys_compaction, name) == expected[name]

    def test_token_estimator_version_is_current(self) -> None:
        assert chrys_compaction.TOKEN_ESTIMATOR_VERSION == 4

    def test_group_kind_literal_stays_stable(self) -> None:
        # chrys uses a py3.14 ``type`` alias; the Literal lives on __value__.
        assert get_args(chrys_compaction.GroupKind.__value__) == (
            "system",
            "user",
            "assistant_text",
            "tool_call",
        )

    def test_tokenizer_protocol_accepts_the_production_tokenizer(self) -> None:
        tokenizer = MixedLanguageTokenizer()
        assert isinstance(tokenizer, chrys_compaction.TokenizerProtocol)

    def test_tokenizer_protocol_rejects_non_conforming_objects(self) -> None:
        assert not isinstance(object(), chrys_compaction.TokenizerProtocol)


# ---------------------------------------------------------------------------
# B. Repeatability and serialization invariants
# ---------------------------------------------------------------------------


class TestAnnotationDeterminismAndIncrementalStability:
    @pytest.mark.parametrize("make", _SHAPE_FACTORIES, ids=lambda f: f.__name__.lstrip("_"))
    def test_group_annotation_is_repeatable(self, make: Callable[[], list[Message]]) -> None:
        first, second = make(), make()
        first_ids = chrys_compaction.annotate_message_groups(first)
        second_ids = chrys_compaction.annotate_message_groups(second)
        assert first_ids == second_ids
        _assert_lists_equivalent(first, second)

    @pytest.mark.parametrize("make", _SHAPE_FACTORIES, ids=lambda f: f.__name__.lstrip("_"))
    def test_tokenized_annotation_is_repeatable(self, make: Callable[[], list[Message]]) -> None:
        first, second = make(), make()
        chrys_compaction.annotate_message_groups(first, tokenizer=_LenTokenizer())
        chrys_compaction.annotate_message_groups(second, tokenizer=_LenTokenizer())
        _assert_lists_equivalent(first, second)
        assert chrys_compaction.included_token_count(first) == chrys_compaction.included_token_count(second)

    def test_serialize_message_keeps_non_ascii_text_literal(self) -> None:
        messages = _unicode_payloads()
        for message in messages:
            message.message_id = "msg_pinned"

        assert [chrys_compaction._serialize_message(message) for message in messages[:3]] == [
            (
                '{"contents": [{"additional_properties": {}, "text": '
                '"中文与 emoji 🙂 mixed", "type": "text"}], '
                '"message_id": "msg_pinned", "role": "user"}'
            ),
            (
                '{"contents": [{"additional_properties": {}, "arguments": {}, "call_id": "c1", '
                '"name": "翻译", "type": "function_call"}], '
                '"message_id": "msg_pinned", "role": "assistant"}'
            ),
            (
                '{"contents": [{"additional_properties": {}, "call_id": "c1", '
                '"result": "résumé — naïve", "type": "function_result"}], '
                '"message_id": "msg_pinned", "role": "tool"}'
            ),
        ]

    @pytest.mark.parametrize(
        "text",
        ["中文" * 200, "こんにちは" * 80],
        ids=["cjk", "kana"],
    )
    def test_non_ascii_message_estimate_uses_literal_serialization(self, text: str) -> None:
        message = _user(text)
        tokenizer = MixedLanguageTokenizer()
        payload = {
            "role": message.role,
            "message_id": message.message_id,
            "contents": [chrys_compaction._serialize_content(content) for content in message.contents],
        }
        literal = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        escaped = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)

        estimate = chrys_compaction.estimate_message_tokens(message, tokenizer)

        assert chrys_compaction._serialize_message(message) == literal
        assert estimate == tokenizer.count_tokens(literal)
        assert estimate < tokenizer.count_tokens(escaped) * 0.6

    def test_execution_stamp_is_excluded_from_token_serialization(self) -> None:
        message = _tool_result("c1", "done")
        result = message.contents[0]
        result.additional_properties["persisted"] = "kept"
        baseline = chrys_compaction._serialize_message(message)
        result.additional_properties[EXECUTION_STAMP_KEY] = build_execution_stamp(
            {"payload": "x" * 4_096},
            outcome="ok",
        )

        assert chrys_compaction._serialize_message(message) == baseline
        assert EXECUTION_STAMP_KEY in result.additional_properties

    def test_reasoning_origin_stamp_is_excluded_from_token_serialization(self) -> None:
        reasoning = Content.from_text_reasoning(text="thinking", protected_data="sig")
        message = Message(role="assistant", contents=[reasoning])
        baseline = chrys_compaction._serialize_message(message)
        ReasoningOrigin("anthropic_messages", "https://api.anthropic.com:443").stamp(reasoning.additional_properties)

        assert chrys_compaction._serialize_message(message) == baseline
        assert REASONING_ORIGIN_KEY in reasoning.additional_properties

    def test_refused_thinking_marker_is_excluded_from_token_serialization(self) -> None:
        reasoning = Content.from_text_reasoning(text="thinking", protected_data="sig")
        message = Message(role="assistant", contents=[reasoning])
        baseline = chrys_compaction._serialize_message(message)
        reasoning.additional_properties[ANTHROPIC_THINKING_STRIPPED_KEY] = True

        assert chrys_compaction._serialize_message(message) == baseline
        assert reasoning.additional_properties[ANTHROPIC_THINKING_STRIPPED_KEY] is True

    def test_hosted_wire_replay_mirror_is_excluded_from_token_serialization(self) -> None:
        payload = {"encrypted_content": "x" * 4_096}
        baseline = Content.from_search_tool_result(
            "search_1",
            tool_name="web_search",
            result=payload,
            hosted_provider="anthropic",
            additional_properties={"persisted": "kept"},
        )
        mirrored = Content.from_search_tool_result(
            "search_1",
            tool_name="web_search",
            result=payload,
            hosted_provider="anthropic",
            additional_properties={
                "persisted": "kept",
                "anthropic.hosted_block": payload,
            },
        )

        assert chrys_compaction._serialize_content(mirrored) == chrys_compaction._serialize_content(baseline)
        assert "anthropic.hosted_block" in mirrored.additional_properties

    def test_responses_encrypted_reasoning_is_excluded_from_token_serialization(self) -> None:
        baseline = Content.from_text_reasoning(id="rs_1", text="Visible summary")
        encrypted = Content.from_text_reasoning(
            id="rs_1",
            text="Visible summary",
            protected_data="opaque-base64-state" * 1_024,
        )

        assert chrys_compaction._serialize_content(encrypted) == chrys_compaction._serialize_content(baseline)
        assert encrypted.protected_data is not None

    def test_idless_protected_reasoning_remains_in_token_serialization(self) -> None:
        content = Content.from_text_reasoning(
            protected_data='{"reasoning_content":"replayed chat reasoning"}',
        )

        assert chrys_compaction._serialize_content(content)["protected_data"] == content.protected_data

    def test_incremental_annotation_is_repeatable(self) -> None:
        first, second = _basic_conversation(), _basic_conversation()
        chrys_compaction.annotate_message_groups(first)
        chrys_compaction.annotate_message_groups(second)
        for target in (first, second):
            target.extend([_user("new turn"), _assistant_tool_call("c9", "again"), _tool_result("c9", "ok")])
        first_ids = chrys_compaction.annotate_message_groups(first)
        second_ids = chrys_compaction.annotate_message_groups(second)
        assert first_ids == second_ids
        _assert_lists_equivalent(first, second)

    def test_from_index_inside_a_group_backs_up_identically(self) -> None:
        first, second = _basic_conversation(), _basic_conversation()
        chrys_compaction.annotate_message_groups(first)
        chrys_compaction.annotate_message_groups(second)
        # Index 4 is the middle of the c1 tool-call group; both passes
        # must back up to the group start before re-annotating.
        first_ids = chrys_compaction.annotate_message_groups(first, from_index=4)
        second_ids = chrys_compaction.annotate_message_groups(second, from_index=4)
        assert first_ids == second_ids
        _assert_lists_equivalent(first, second)

    def test_force_reannotate_is_repeatable(self) -> None:
        first, second = _reasoning_prefix_group(), _reasoning_prefix_group()
        chrys_compaction.annotate_message_groups(first, tokenizer=_LenTokenizer())
        chrys_compaction.annotate_message_groups(second, tokenizer=_LenTokenizer())
        first_ids = chrys_compaction.annotate_message_groups(first, force_reannotate=True)
        second_ids = chrys_compaction.annotate_message_groups(second, force_reannotate=True)
        assert first_ids == second_ids
        _assert_lists_equivalent(first, second)

    def test_token_gap_after_plain_annotation_is_repeatable(self) -> None:
        # Fully group-annotated but untokenized lists: the gap scan must key
        # off first_untokenized (include_tokens branch of
        # _first_annotation_gaps) and backfill counts identically. Unreachable
        # from chrys production today (the strategy never passes tokenizer to
        # annotate_message_groups) — pinned so the dormant branch cannot drift.
        first, second = _basic_conversation(), _basic_conversation()
        chrys_compaction.annotate_message_groups(first)
        chrys_compaction.annotate_message_groups(second)
        first_ids = chrys_compaction.annotate_message_groups(first, tokenizer=_LenTokenizer())
        second_ids = chrys_compaction.annotate_message_groups(second, tokenizer=_LenTokenizer())
        assert first_ids == second_ids
        _assert_lists_equivalent(first, second)
        assert chrys_compaction.included_token_count(second) == chrys_compaction.included_token_count(first) > 0

    def test_annotate_token_counts_is_repeatable(self) -> None:
        first, second = _unicode_payloads(), _unicode_payloads()
        chrys_compaction.annotate_token_counts(first, tokenizer=_LenTokenizer())
        chrys_compaction.annotate_token_counts(second, tokenizer=_LenTokenizer())
        _assert_lists_equivalent(first, second)
        # Appended tail is tokenized incrementally without retokenizing the prefix.
        for target in (first, second):
            target.append(_user("tail"))
        chrys_compaction.annotate_token_counts(first, tokenizer=_LenTokenizer())
        chrys_compaction.annotate_token_counts(second, tokenizer=_LenTokenizer())
        _assert_lists_equivalent(first, second)

    def test_tool_result_images_get_additive_token_estimate(self) -> None:
        message = _tool_image_result("call_1")
        estimate = chrys_compaction.estimate_toolresult_image_tokens(message)

        assert estimate == 425

        text_only = _tool_result("call_1", "caption")
        chrys_compaction.annotate_token_counts([text_only], tokenizer=_LenTokenizer())
        chrys_compaction.annotate_token_counts([message], tokenizer=_LenTokenizer())

        assert (chrys_compaction._token_count(message) or 0) == (
            (chrys_compaction._token_count(text_only) or 0) + estimate
        )

    def test_top_level_user_images_do_not_get_additive_tool_result_estimate(self) -> None:
        message = Message(
            role="user",
            contents=[
                Content.from_text("describe"),
                Content.from_data(
                    b"image",
                    "image/png",
                    additional_properties={"width": 1024, "height": 512, "media_type": "image/png"},
                ),
            ],
        )

        assert chrys_compaction.estimate_toolresult_image_tokens(message) == 0

    def test_top_level_user_image_tokens_use_visual_estimate_not_base64_text(self) -> None:
        data = _generated_png_bytes()
        assert len(data) > 300_000
        image = Content.from_data(data, "image/png")
        message = Message(role="user", contents=[Content.from_text("describe"), image])

        chrys_compaction.annotate_token_counts([message], tokenizer=_LenTokenizer())

        assert chrys_compaction.estimate_message_image_tokens(message) == 255
        assert (chrys_compaction._token_count(message) or 0) < 2_000

    def test_dimension_cache_is_warmed_before_serialized_token_count(self) -> None:
        data = _generated_png_bytes()
        message = Message(
            role="user",
            contents=[
                Content.from_text("describe"),
                Content.from_data(data, "image/png"),
            ],
        )

        chrys_compaction.annotate_token_counts([message], tokenizer=_LenTokenizer())
        first_count = chrys_compaction._token_count(message)
        chrys_compaction.annotate_token_counts([message], tokenizer=_LenTokenizer(), force_retokenize=True)

        assert chrys_compaction._token_count(message) == first_count

    def test_multiple_direct_images_sum_visual_estimates(self) -> None:
        data = _generated_png_bytes()
        message = Message(
            role="user",
            contents=[
                Content.from_text("compare"),
                Content.from_data(data, "image/png"),
                Content.from_data(data, "image/png"),
            ],
        )

        assert chrys_compaction.estimate_message_image_tokens(message) == 510

    def test_repeated_same_image_object_counts_each_serialized_occurrence(self) -> None:
        data = _generated_png_bytes()
        image = Content.from_data(data, "image/png")
        message = Message(role="user", contents=[Content.from_text("compare"), image, image])

        serialized = chrys_compaction._serialize_message(message)

        assert serialized.count("[image data omitted for token estimate]") == 2
        assert chrys_compaction.estimate_message_image_tokens(message) == 510

    def test_nested_image_generation_result_uses_visual_estimate_not_base64_text(self) -> None:
        data = _generated_png_bytes()
        image = Content.from_data(data, "image/png")
        message = Message(
            role="assistant",
            contents=[Content.from_image_generation_tool_result(image_id="img_1", outputs=image)],
        )

        chrys_compaction.annotate_token_counts([message], tokenizer=_LenTokenizer())

        assert "data:image/png;base64," not in chrys_compaction._serialize_message(message)
        assert chrys_compaction.estimate_message_image_tokens(message) == 255
        assert (chrys_compaction._token_count(message) or 0) < 2_000

    def test_nested_code_interpreter_input_image_uses_recursive_estimate(self) -> None:
        data = _generated_png_bytes()
        image = Content.from_data(data, "image/png")
        message = Message(
            role="assistant",
            contents=[Content.from_code_interpreter_tool_call(call_id="call_1", inputs=[image])],
        )

        chrys_compaction.annotate_token_counts([message], tokenizer=_LenTokenizer())

        assert "data:image/png;base64," not in chrys_compaction._serialize_message(message)
        assert chrys_compaction.estimate_message_image_tokens(message) == 255
        assert (chrys_compaction._token_count(message) or 0) < 2_000

    def test_restored_mcp_output_image_dict_uses_visual_estimate_not_base64_text(self) -> None:
        data = _generated_png_bytes()
        image = Content.from_data(data, "image/png")
        live = Message(
            role="assistant",
            contents=[
                Content.from_mcp_server_tool_result(
                    "call_1",
                    output=[Content.from_text("caption"), image],
                )
            ],
        )
        restored = Message.from_dict(live.to_dict())
        restored_output = restored.contents[0].output
        assert isinstance(restored_output, list)
        assert isinstance(restored_output[1], dict)

        chrys_compaction.annotate_token_counts([restored], tokenizer=_LenTokenizer())

        assert "data:image/png;base64," not in chrys_compaction._serialize_message(restored)
        assert chrys_compaction.estimate_message_image_tokens(restored) == 255
        assert (chrys_compaction._token_count(restored) or 0) < 2_000
        assert restored_output[1]["additional_properties"]["width"] == 512
        assert restored_output[1]["additional_properties"]["height"] == 512

    def test_data_uri_image_without_media_type_uses_visual_estimate_not_base64_text(self) -> None:
        data = _generated_png_bytes()
        image = Content.from_dict(
            {
                "type": "data",
                "uri": Content.from_data(data, "image/png").uri,
            }
        )
        assert image.media_type is None
        message = Message(role="user", contents=[image])

        chrys_compaction.annotate_token_counts([message], tokenizer=_LenTokenizer())

        assert "data:image/png;base64," not in chrys_compaction._serialize_message(message)
        assert chrys_compaction.estimate_message_image_tokens(message) == 255
        assert (chrys_compaction._token_count(message) or 0) < 2_000

    def test_uppercase_data_uri_base64_marker_uses_visual_estimate(self) -> None:
        data = _generated_png_bytes()
        image = Content.from_dict(
            {
                "type": "data",
                "uri": Content.from_data(data, "image/png").uri.replace(
                    "data:image/png;base64,",
                    "DATA:IMAGE/PNG;BASE64,",
                    1,
                ),
            }
        )
        message = Message(role="user", contents=[image])

        assert chrys_compaction.estimate_message_image_tokens(message) == 255

    def test_image_shaped_function_call_arguments_remain_text_budgeted(self) -> None:
        data = _generated_png_bytes()
        image_uri = Content.from_data(data, "image/png").uri
        message = Message(
            role="assistant",
            contents=[
                Content.from_function_call(
                    "call_1",
                    "tool_with_json_args",
                    arguments={"payload": {"type": "data", "uri": image_uri}},
                )
            ],
        )

        chrys_compaction.annotate_token_counts([message], tokenizer=_LenTokenizer())

        assert "data:image/png;base64," in chrys_compaction._serialize_message(message)
        assert chrys_compaction.estimate_message_image_tokens(message) == 0
        assert (chrys_compaction._token_count(message) or 0) > 300_000

    def test_non_tool_call_arguments_key_in_mcp_output_still_redacts_image_payload(self) -> None:
        data = _generated_png_bytes()
        image_uri = Content.from_data(data, "image/png").uri
        message = Message(
            role="assistant",
            contents=[
                Content.from_mcp_server_tool_result(
                    "call_1",
                    output={"arguments": {"type": "data", "uri": image_uri}},
                )
            ],
        )

        chrys_compaction.annotate_token_counts([message], tokenizer=_LenTokenizer())

        assert "data:image/png;base64," not in chrys_compaction._serialize_message(message)
        assert chrys_compaction.estimate_message_image_tokens(message) == 255
        assert (chrys_compaction._token_count(message) or 0) < 2_000

    def test_image_dimension_inference_failure_falls_back_without_raising(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        image = Content.from_uri("data:image/png;base64,AAAA")
        message = Message(role="user", contents=[image])

        def raise_unexpected(_data: bytes) -> tuple[int, int]:
            raise NotImplementedError("unsupported decoder")

        monkeypatch.setattr(chrys_compaction, "inspect_image_dimensions", raise_unexpected)

        assert chrys_compaction.estimate_message_image_tokens(message) == 765

    def test_unknown_tool_result_uri_does_not_raise_or_count_as_image(self) -> None:
        message = Message(
            role="tool",
            contents=[
                Content.from_function_result(
                    "call_1",
                    result=[Content.from_text("unknown"), Content.from_uri("https://example.com/blob")],
                )
            ],
        )

        assert chrys_compaction.estimate_toolresult_image_tokens(message) == 0


# ---------------------------------------------------------------------------
# C. Mixed-writer seam — the production sequence
# ---------------------------------------------------------------------------


class TestMixedWriterSeam:
    def test_production_sequence_chrys_annotation_strategy_projection(self) -> None:
        messages = _basic_conversation()
        # Step 1: the owned wire client annotates with the Chrys implementation
        # before invoking the strategy.
        chrys_compaction.annotate_message_groups(messages)
        # Step 2: the chrys strategy may re-annotate and tokenize the same list.
        chrys_compaction.annotate_message_groups(messages, force_reannotate=True)
        chrys_compaction.annotate_token_counts(messages, tokenizer=_LenTokenizer())
        excluded = [messages[2], messages[3], messages[4]]
        for message in excluded:
            assert chrys_compaction.set_excluded(message, excluded=True, reason="seam-test")
        # Step 3: the owned wire client projects with the Chrys implementation.
        projected = chrys_compaction.project_included_messages(messages)
        assert projected == chrys_compaction.included_messages(messages)
        assert all(message not in projected for message in excluded)
        assert [message.role for message in projected] == ["system", "user", "assistant", "user"]
        assert chrys_compaction.included_token_count(messages) == sum(
            chrys_compaction._token_count(message) or 0 for message in projected
        )

    def test_chrys_reads_annotated_prefix_without_rewriting_it(self) -> None:
        messages = _basic_conversation()
        chrys_compaction.annotate_message_groups(messages)
        prefix_annotations = [m.additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY] for m in messages]
        messages.extend([_user("new turn"), _assistant_text("reply")])
        ids = chrys_compaction.annotate_message_groups(messages)
        # Prefix annotation dicts stay the same objects — untouched, not
        # rewritten — except the final prefix group: incremental annotation
        # deliberately backs up into it (the group could continue into the new
        # tail), rewriting it with identical values.
        for annotation, message in zip(prefix_annotations[:-1], messages, strict=False):
            assert message.additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY] is annotation
        assert (
            messages[len(prefix_annotations) - 1].additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY]
            == prefix_annotations[-1]
        )
        # The suffix continues the existing group index sequence.
        assert chrys_compaction._group_index(messages[-1]) > chrys_compaction._group_index(messages[-3])
        assert ids == chrys_compaction.annotate_message_groups(messages)

    def test_repeated_annotation_keeps_chrys_prefix_without_rewriting_it(self) -> None:
        messages = _basic_conversation()
        chrys_compaction.annotate_message_groups(messages)
        prefix_annotations = [m.additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY] for m in messages]
        messages.extend([_user("new turn"), _assistant_text("reply")])
        chrys_compaction.annotate_message_groups(messages)
        # Same final-group rewrite tolerance as the prefix test above.
        for annotation, message in zip(prefix_annotations[:-1], messages, strict=False):
            assert message.additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY] is annotation
        assert (
            messages[len(prefix_annotations) - 1].additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY]
            == prefix_annotations[-1]
        )

    async def test_apply_compaction_seam_runs_a_chrys_primitive_strategy(self) -> None:
        messages = _basic_conversation()

        class _ChrysPrimitiveStrategy:
            async def __call__(
                self, msgs: list[Message], context: chrys_compaction.CompactionCallContext | None = None
            ) -> bool:
                chrys_compaction.annotate_message_groups(msgs, force_reannotate=True)
                chrys_compaction.annotate_token_counts(msgs, tokenizer=_LenTokenizer())
                changed = False
                for msg in msgs:
                    if chrys_compaction._group_kind(msg) == "tool_call":
                        changed = chrys_compaction.set_excluded(msg, excluded=True, reason="seam") or changed
                return changed

        projected = await chrys_compaction.apply_compaction(
            messages, strategy=_ChrysPrimitiveStrategy(), tokenizer=None
        )
        assert projected == chrys_compaction.included_messages(messages)
        assert all(message.role != "tool" for message in projected)

    @pytest.mark.parametrize(
        "excluded_group_indexes",
        [
            {0},
            {1, 2},
        ],
        ids=["reasoning-dropped", "calls-dropped"],
    )
    async def test_apply_compaction_degrades_partial_reasoning_tool_groups(
        self,
        excluded_group_indexes: set[int],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        group = _reasoning_tool_call_group("call-1", "lookup")
        messages = [_user("question"), *group, _assistant_text("answer")]

        class _NonAtomicStrategy:
            async def __call__(
                self,
                msgs: list[Message],
                context: chrys_compaction.CompactionCallContext | None = None,
            ) -> bool:
                del context
                for group_index in excluded_group_indexes:
                    chrys_compaction.set_excluded(
                        msgs[group_index + 1],
                        excluded=True,
                        reason="custom strategy",
                    )
                return True

        with caplog.at_level("WARNING", logger="chrys.kernel.compaction"):
            projected = await chrys_compaction.apply_compaction(
                messages,
                strategy=_NonAtomicStrategy(),
            )

        assert projected == [messages[0], messages[-1]]
        assert all(message.additional_properties[chrys_compaction.EXCLUDED_KEY] for message in group)
        warnings = [
            record
            for record in caplog.records
            if record.message.startswith("Degraded partially projected tool-call groups")
        ]
        assert len(warnings) == 1

    async def test_restored_partial_group_degrades_even_in_strict_mode(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        live = [_user("question"), *_reasoning_tool_call_group("call-1", "lookup")]
        chrys_compaction.annotate_message_groups(live)
        chrys_compaction.set_excluded(live[2], excluded=True, reason="persisted strategy")
        restored = [Message.from_dict(message.to_dict()) for message in live]

        class _NoOpStrategy:
            async def __call__(
                self,
                msgs: list[Message],
                context: chrys_compaction.CompactionCallContext | None = None,
            ) -> bool:
                del msgs, context
                return False

        with caplog.at_level("WARNING", logger="chrys.kernel.compaction"):
            projected = await chrys_compaction.apply_compaction(
                restored,
                strategy=_NoOpStrategy(),
                strict_projection_atomicity=True,
            )

        assert projected == [restored[0]]
        assert all(message.additional_properties[chrys_compaction.EXCLUDED_KEY] for message in restored[1:])
        assert any(
            record.message.startswith("Degraded partially projected tool-call groups") for record in caplog.records
        )

    async def test_duplicate_group_id_twins_degrade_only_the_partial_occurrence(self) -> None:
        first_group = _reasoning_tool_call_group("call-1", "first")
        second_group = _reasoning_tool_call_group("call-2", "second")
        first_group[0].message_id = "duplicate"
        second_group[0].message_id = "duplicate"
        messages = [*first_group, *second_group]
        chrys_compaction.annotate_message_groups(messages, force_reannotate=True)
        assert chrys_compaction._group_id(first_group[0]) == chrys_compaction._group_id(second_group[0])
        assert chrys_compaction._group_index(first_group[0]) != chrys_compaction._group_index(second_group[0])

        class _SecondGroupOnlyStrategy:
            async def __call__(
                self,
                msgs: list[Message],
                context: chrys_compaction.CompactionCallContext | None = None,
            ) -> bool:
                del context
                return chrys_compaction.set_excluded(
                    msgs[4],
                    excluded=True,
                    reason="custom strategy",
                )

        projected = await chrys_compaction.apply_compaction(
            messages,
            strategy=_SecondGroupOnlyStrategy(),
        )

        assert projected == first_group
        assert all(not message.additional_properties[chrys_compaction.EXCLUDED_KEY] for message in first_group)
        assert all(message.additional_properties[chrys_compaction.EXCLUDED_KEY] for message in second_group)

    async def test_identical_persisted_signatures_split_at_structural_boundaries(self) -> None:
        first_group = _reasoning_tool_call_group("call-1", "first")
        second_group = _reasoning_tool_call_group("call-2", "second")
        messages = [*first_group, *second_group]
        for message in messages:
            message.additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY] = {
                chrys_compaction.GROUP_ID_KEY: "group_duplicate",
                chrys_compaction.GROUP_KIND_KEY: "tool_call",
                chrys_compaction.GROUP_INDEX_KEY: 0,
                chrys_compaction.GROUP_HAS_REASONING_KEY: True,
            }
        chrys_compaction.set_excluded(messages[4], excluded=True, reason="persisted partial")

        projected = chrys_compaction.project_included_messages(messages)

        assert projected == first_group
        assert all(
            not message.additional_properties.get(chrys_compaction.EXCLUDED_KEY, False) for message in first_group
        )
        assert all(message.additional_properties[chrys_compaction.EXCLUDED_KEY] for message in second_group)

    async def test_identical_persisted_signatures_split_after_assistant_role_results(self) -> None:
        def _mcp_group(call_id: str, name: str) -> list[Message]:
            return [
                Message(
                    role="assistant",
                    contents=[
                        Content.from_text_reasoning(text=f"reasoning for {name}"),
                        Content.from_mcp_server_tool_call(call_id, name, server_name="s", arguments="{}"),
                    ],
                ),
                Message(
                    role="assistant",
                    contents=[Content.from_mcp_server_tool_result(call_id, output=[Content.from_text("done")])],
                ),
            ]

        first_group = _mcp_group("call-1", "first")
        second_group = _mcp_group("call-2", "second")
        messages = [*first_group, *second_group]
        for message in messages:
            message.additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY] = {
                chrys_compaction.GROUP_ID_KEY: "group_duplicate",
                chrys_compaction.GROUP_KIND_KEY: "tool_call",
                chrys_compaction.GROUP_INDEX_KEY: 0,
                chrys_compaction.GROUP_HAS_REASONING_KEY: True,
            }
        chrys_compaction.set_excluded(messages[3], excluded=True, reason="persisted partial")

        projected = chrys_compaction.project_included_messages(messages)

        assert projected == first_group
        assert all(
            not message.additional_properties.get(chrys_compaction.EXCLUDED_KEY, False) for message in first_group
        )
        assert all(message.additional_properties[chrys_compaction.EXCLUDED_KEY] for message in second_group)

    async def test_assistant_role_mcp_result_is_annotated_into_its_call_group(self) -> None:
        messages = [
            Message(role="user", contents=[Content.from_text("Start")]),
            Message(
                role="assistant",
                contents=[
                    Content.from_text_reasoning(id="rs_1", text="Plan", protected_data="encrypted"),
                    Content.from_mcp_server_tool_call("done_call", "search", server_name="s", arguments="{}"),
                ],
            ),
            Message(
                role="assistant",
                contents=[Content.from_mcp_server_tool_result("done_call", output=[Content.from_text("done")])],
            ),
        ]
        chrys_compaction.annotate_message_groups(messages)
        assert chrys_compaction._group_id(messages[1]) == chrys_compaction._group_id(messages[2])

        chrys_compaction.set_excluded(messages[2], excluded=True, reason="strategy excluded result only")
        projected = chrys_compaction.project_included_messages(messages)

        assert [message.role for message in projected] == ["user"]
        assert all(message.additional_properties[chrys_compaction.EXCLUDED_KEY] for message in messages[1:])

    async def test_reasoning_riding_in_absorbed_result_counts_toward_the_group(self) -> None:
        messages = [
            Message(role="user", contents=[Content.from_text("Start")]),
            Message(
                role="assistant",
                contents=[Content.from_mcp_server_tool_call("done_call", "search", server_name="s", arguments="{}")],
            ),
            Message(
                role="assistant",
                contents=[
                    Content.from_text_reasoning(id="rs_1", text="Plan", protected_data="encrypted"),
                    Content.from_mcp_server_tool_result("done_call", output=[Content.from_text("out")]),
                ],
            ),
        ]
        chrys_compaction.annotate_message_groups(messages)
        chrys_compaction.set_excluded(messages[1], excluded=True, reason="strategy excluded call only")

        projected = chrys_compaction.project_included_messages(messages)

        assert [message.role for message in projected] == ["user"]
        assert all(message.additional_properties[chrys_compaction.EXCLUDED_KEY] for message in messages[1:])

    async def test_result_only_assistant_member_stays_inside_its_occurrence(self) -> None:
        messages = [
            _assistant_reasoning("reasoning"),
            _assistant_tool_call("call-1", "lookup"),
            _tool_result("call-1", "tool half"),
            Message(
                role="assistant",
                contents=[Content.from_mcp_server_tool_result("call-1", output=[Content.from_text("mcp half")])],
            ),
        ]
        for message in messages:
            message.additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY] = {
                chrys_compaction.GROUP_ID_KEY: "group_persisted",
                chrys_compaction.GROUP_KIND_KEY: "tool_call",
                chrys_compaction.GROUP_INDEX_KEY: 0,
                chrys_compaction.GROUP_HAS_REASONING_KEY: True,
            }
        chrys_compaction.set_excluded(messages[2], excluded=True, reason="persisted partial")

        projected = chrys_compaction.project_included_messages(messages)

        assert projected == []
        assert all(message.additional_properties[chrys_compaction.EXCLUDED_KEY] for message in messages)

    async def test_strict_atomicity_rejects_before_mutating_the_live_list(self) -> None:
        messages = _reasoning_tool_call_group("call-1", "lookup")
        original_messages = tuple(messages)
        original_state = [message.to_dict(exclude_none=False) for message in messages]

        class _NonAtomicStrategy:
            async def __call__(
                self,
                msgs: list[Message],
                context: chrys_compaction.CompactionCallContext | None = None,
            ) -> bool:
                del context
                return chrys_compaction.set_excluded(
                    msgs[0],
                    excluded=True,
                    reason="custom strategy",
                )

        with pytest.raises(chrys_compaction.CompactionProjectionAtomicityError):
            await chrys_compaction.apply_compaction(
                messages,
                strategy=_NonAtomicStrategy(),
                strict_projection_atomicity=True,
            )

        assert all(message is original for message, original in zip(messages, original_messages, strict=True))
        assert [message.to_dict(exclude_none=False) for message in messages] == original_state

    def test_partial_group_member_comparison_is_order_insensitive_and_requires_liveness(self) -> None:
        """The strict-mode "did the strategy introduce this partial group?"
        comparison is "same set of LIVE referents by identity": reordering
        the same members compares equal (reorder-only strategies stay
        accepted), a different member set does not, and DEAD members never
        compare equal — a freed group can never vouch for a same-shaped
        newcomer across the strategy await."""
        first = _assistant_reasoning("r")
        second = _assistant_tool_call("call-1", "lookup")
        snapshot_a = tuple(weakref.ref(m) for m in (first, second))
        snapshot_b = tuple(weakref.ref(m) for m in (second, first))

        assert chrys_compaction._same_live_members(snapshot_a, snapshot_b)

        other = _tool_result("call-1", "x")
        snapshot_c = tuple(weakref.ref(m) for m in (first, other))
        assert not chrys_compaction._same_live_members(snapshot_a, snapshot_c)

        del first
        gc.collect()
        assert not chrys_compaction._same_live_members(snapshot_a, snapshot_b)

    async def test_strict_mode_accepts_reorder_only_strategy_on_preexisting_partial_group(self) -> None:
        """Reordering a PREEXISTING partial group's members is not a newly
        introduced partial group: the strict check compares member sets, not
        member order, so the run degrades (as without the strategy) instead
        of raising."""
        messages = [_user("question"), *_reasoning_tool_call_group("call-1", "lookup")]
        chrys_compaction.annotate_message_groups(messages)
        chrys_compaction.set_excluded(messages[3], excluded=True, reason="persisted strategy")

        class _ReorderStrategy:
            async def __call__(
                self,
                msgs: list[Message],
                context: chrys_compaction.CompactionCallContext | None = None,
            ) -> bool:
                del context
                msgs[1], msgs[2] = msgs[2], msgs[1]
                return True

        projected = await chrys_compaction.apply_compaction(
            messages,
            strategy=_ReorderStrategy(),
            strict_projection_atomicity=True,
        )

        assert [message.role for message in projected] == ["user"]
        assert all(message.additional_properties[chrys_compaction.EXCLUDED_KEY] for message in messages[1:])

    async def test_projection_degrades_partial_pairing_in_a_plain_tool_call_group(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A fused non-reasoning group whose call is excluded while its result
        stays included (sessions compacted under the earlier split grouping
        persist exactly this shape) must not project the orphan result.
        Narration-only members are not pairing carriers and stay included."""
        messages = [
            _user("question"),
            _assistant_tool_call("call-1", "lookup"),
            _assistant_text("checking the config next"),
            _tool_result("call-1", "ok"),
        ]
        chrys_compaction.annotate_message_groups(messages)
        chrys_compaction.set_excluded(messages[1], excluded=True, reason="persisted compaction")

        with caplog.at_level("WARNING", logger="chrys.kernel.compaction"):
            projected = chrys_compaction.project_included_messages(messages)

        assert projected == [messages[0], messages[2]]
        assert messages[3].additional_properties[chrys_compaction.EXCLUDED_KEY]
        assert not messages[2].additional_properties.get(chrys_compaction.EXCLUDED_KEY, False)
        assert any(
            record.message.startswith("Degraded partially projected tool-call groups") for record in caplog.records
        )

    async def test_projection_degrades_a_dangling_call_when_its_results_are_excluded(self) -> None:
        """The reverse partial pairing — call included, results excluded —
        projects a call that never resolves; both directions must degrade to
        a uniformly excluded carrier set."""
        messages = [
            _user("question"),
            _assistant_tool_call("call-1", "lookup"),
            _assistant_text("checking the config next"),
            _tool_result("call-1", "ok"),
        ]
        chrys_compaction.annotate_message_groups(messages)
        chrys_compaction.set_excluded(messages[3], excluded=True, reason="persisted partial")

        projected = chrys_compaction.project_included_messages(messages)

        assert projected == [messages[0], messages[2]]
        assert messages[1].additional_properties[chrys_compaction.EXCLUDED_KEY]
        assert not messages[2].additional_properties.get(chrys_compaction.EXCLUDED_KEY, False)

    async def test_uniformly_excluded_carriers_keep_narration_without_degrade(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Sessions that compacted BOTH halves of a split exchange kept the
        mid-exchange narration on the wire; carrier-scoped pairing atomicity
        must not reclassify that shape as partial and drop the text."""
        messages = [
            _user("question"),
            _assistant_tool_call("call-1", "lookup"),
            _assistant_text("checking the config next"),
            _tool_result("call-1", "ok"),
        ]
        chrys_compaction.annotate_message_groups(messages)
        chrys_compaction.set_excluded(messages[1], excluded=True, reason="persisted compaction")
        chrys_compaction.set_excluded(messages[3], excluded=True, reason="persisted compaction")

        with caplog.at_level("WARNING", logger="chrys.kernel.compaction"):
            projected = chrys_compaction.project_included_messages(messages)

        assert projected == [messages[0], messages[2]]
        assert not messages[2].additional_properties.get(chrys_compaction.EXCLUDED_KEY, False)
        assert not any(record.message.startswith("Degraded") for record in caplog.records)

    async def test_excluded_narration_alone_is_not_a_partial_pairing(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Excluding only a narration member (text compression) leaves the
        call/result pairing intact — the wire stays legal and no degrade
        may fire."""
        messages = [
            _user("question"),
            _assistant_tool_call("call-1", "lookup"),
            _assistant_text("checking the config next"),
            _tool_result("call-1", "ok"),
        ]
        chrys_compaction.annotate_message_groups(messages)
        chrys_compaction.set_excluded(messages[2], excluded=True, reason="compression")

        with caplog.at_level("WARNING", logger="chrys.kernel.compaction"):
            projected = chrys_compaction.project_included_messages(messages)

        assert projected == [messages[0], messages[1], messages[3]]
        assert not any(record.message.startswith("Degraded") for record in caplog.records)

    async def test_legacy_split_grouping_exclusions_heal_after_forced_reannotation(self) -> None:
        """Persisted pre-fusion annotations: the call-only group was excluded
        by compaction, the detached result group stayed included. Forced
        re-annotation on restore fuses the exchange into one group; projection
        must repair the mixed pairing instead of sending a result without its
        call."""
        call = _assistant_tool_call("call-1", "lookup")
        narration = _assistant_text("checking the config next")
        result = _tool_result("call-1", "ok")
        legacy_annotations = (
            (call, "group_legacy_call", "tool_call", 1),
            (narration, "group_legacy_text", "assistant_text", 2),
            (result, "group_legacy_result", "tool_call", 3),
        )
        for message, group_id, kind, group_index in legacy_annotations:
            message.additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY] = {
                chrys_compaction.GROUP_ID_KEY: group_id,
                chrys_compaction.GROUP_KIND_KEY: kind,
                chrys_compaction.GROUP_INDEX_KEY: group_index,
                chrys_compaction.GROUP_HAS_REASONING_KEY: False,
            }
        chrys_compaction.set_excluded(call, excluded=True, reason="compaction")
        messages = [_user("question"), call, narration, result]

        chrys_compaction.annotate_message_groups(messages, force_reannotate=True)
        projected = chrys_compaction.project_included_messages(messages)

        projected_calls = {
            content.call_id
            for message in projected
            for content in message.contents
            if content.type in chrys_compaction.TOOL_CALL_CONTENT_TYPES
        }
        projected_results = {
            content.call_id
            for message in projected
            for content in message.contents
            if content.type in chrys_compaction.TOOL_RESULT_CONTENT_TYPES
        }
        assert projected_results <= projected_calls
        assert projected == [messages[0], narration]

    async def test_strict_atomicity_rejects_an_introduced_plain_partial_pairing(self) -> None:
        """Strict mode vetoes strategies that split a non-reasoning group's
        pairing, exactly as it vetoes partial reasoning groups."""
        messages = [_assistant_tool_call("call-1", "lookup"), _tool_result("call-1", "ok")]

        class _CallOnlyExcluder:
            async def __call__(
                self,
                msgs: list[Message],
                context: chrys_compaction.CompactionCallContext | None = None,
            ) -> bool:
                del context
                return chrys_compaction.set_excluded(
                    msgs[0],
                    excluded=True,
                    reason="custom strategy",
                )

        with pytest.raises(chrys_compaction.CompactionProjectionAtomicityError):
            await chrys_compaction.apply_compaction(
                messages,
                strategy=_CallOnlyExcluder(),
                strict_projection_atomicity=True,
            )

    async def test_reasoning_group_leaves_plain_narration_out_of_the_atomic_set(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The atomic set is carriers plus reasoning-bearing members in EVERY
        group: plain narration fused into a reasoning exchange stays free, so
        uniformly compacted carriers leave it on the wire without a degrade."""
        messages = [
            _user("question"),
            _assistant_reasoning_and_call("call-1", "lookup"),
            _assistant_text("checking the config next"),
            _tool_result("call-1", "ok"),
        ]
        chrys_compaction.annotate_message_groups(messages)
        chrys_compaction.set_excluded(messages[1], excluded=True, reason="persisted compaction")
        chrys_compaction.set_excluded(messages[3], excluded=True, reason="persisted compaction")

        with caplog.at_level("WARNING", logger="chrys.kernel.compaction"):
            projected = chrys_compaction.project_included_messages(messages)

        assert projected == [messages[0], messages[2]]
        assert not messages[2].additional_properties.get(chrys_compaction.EXCLUDED_KEY, False)
        assert not any(record.message.startswith("Degraded") for record in caplog.records)

    async def test_reasoning_group_mixed_carriers_degrade_without_dragging_narration(self) -> None:
        """A partial pairing inside a reasoning exchange degrades the carriers
        and reasoning riders, while fused plain narration stays on the wire."""
        messages = [
            _user("question"),
            _assistant_reasoning_and_call("call-1", "lookup"),
            _assistant_text("checking the config next"),
            _tool_result("call-1", "ok"),
        ]
        chrys_compaction.annotate_message_groups(messages)
        chrys_compaction.set_excluded(messages[1], excluded=True, reason="persisted compaction")

        projected = chrys_compaction.project_included_messages(messages)

        assert projected == [messages[0], messages[2]]
        assert messages[3].additional_properties[chrys_compaction.EXCLUDED_KEY]
        assert not messages[2].additional_properties.get(chrys_compaction.EXCLUDED_KEY, False)


# ---------------------------------------------------------------------------
# D. Mirror unit behaviour
# ---------------------------------------------------------------------------


class TestMirrorBehaviour:
    @pytest.mark.parametrize(
        "tool_call",
        [
            Content.from_function_call("call-1", "local"),
            Content.from_code_interpreter_tool_call(call_id="code-1"),
            Content.from_image_generation_tool_call(image_id="image-1"),
            Content.from_mcp_server_tool_call("mcp-1", "remote"),
            Content.from_search_tool_call("search-1", tool_name="web_search"),
            Content.from_shell_tool_call(call_id="shell-1", commands=["pwd"]),
        ],
        ids=["function", "code", "image", "mcp", "search", "shell"],
    )
    def test_reasoning_prefix_is_atomic_with_every_hosted_call_family(self, tool_call: Content) -> None:
        messages = [_assistant_reasoning("private plan"), Message(role="assistant", contents=[tool_call])]

        spans = chrys_compaction.group_messages(messages)

        assert len(spans) == 1
        assert spans[0]["kind"] == "tool_call"
        assert spans[0]["start_index"] == 0
        assert spans[0]["end_index"] == 1
        assert spans[0]["has_reasoning"] is True

    def test_annotate_empty_list_returns_empty(self) -> None:
        assert chrys_compaction.annotate_message_groups([]) == []

    def test_repeat_annotation_without_gaps_is_a_no_op(self) -> None:
        messages = _basic_conversation()
        chrys_compaction.annotate_message_groups(messages)
        annotations = [m.additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY] for m in messages]
        chrys_compaction.annotate_message_groups(messages)
        for annotation, message in zip(annotations, messages, strict=True):
            assert message.additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY] is annotation

    def test_set_excluded_change_detection_and_sticky_reason(self) -> None:
        message = _user("x")
        assert chrys_compaction.set_excluded(message, excluded=True, reason="first") is True
        assert chrys_compaction.set_excluded(message, excluded=True, reason="second") is False
        assert message.additional_properties[chrys_compaction.EXCLUDE_REASON_KEY] == "second"
        # Re-including with no reason leaves the stale reason in place.
        assert chrys_compaction.set_excluded(message, excluded=False) is True
        assert message.additional_properties[chrys_compaction.EXCLUDED_KEY] is False
        assert message.additional_properties[chrys_compaction.EXCLUDE_REASON_KEY] == "second"

    def test_included_token_count_skips_excluded_and_untokenized(self) -> None:
        messages = _basic_conversation()
        chrys_compaction.annotate_message_groups(messages, tokenizer=_LenTokenizer())
        full = chrys_compaction.included_token_count(messages)
        assert full > 0
        chrys_compaction.set_excluded(messages[0], excluded=True, reason="test")
        assert chrys_compaction.included_token_count(messages) < full
        # A message without annotations contributes nothing.
        messages.append(_user("untokenized tail"))
        assert chrys_compaction.included_token_count(messages) == chrys_compaction.included_token_count(messages[:-1])

    def test_unversioned_cached_token_count_is_recomputed_and_stamped(self) -> None:
        message = _user("中文")
        chrys_compaction.annotate_message_groups([message])
        annotation = message.additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY]
        annotation[chrys_compaction.GROUP_TOKEN_COUNT_KEY] = 99_999

        assert chrys_compaction.GROUP_TOKEN_ESTIMATOR_VERSION_KEY not in annotation
        assert chrys_compaction._token_count(message) is None

        chrys_compaction.annotate_token_counts([message], tokenizer=_LenTokenizer())

        assert chrys_compaction._token_count(message) == len(chrys_compaction._serialize_message(message))
        assert annotation[chrys_compaction.GROUP_TOKEN_ESTIMATOR_VERSION_KEY] == (
            chrys_compaction.TOKEN_ESTIMATOR_VERSION
        )

    def test_current_cached_token_count_remains_incremental(self) -> None:
        message = _user("cached")
        chrys_compaction.annotate_message_groups([message])
        annotation = message.additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY]
        annotation[chrys_compaction.GROUP_TOKEN_COUNT_KEY] = 123
        annotation[chrys_compaction.GROUP_TOKEN_ESTIMATOR_VERSION_KEY] = chrys_compaction.TOKEN_ESTIMATOR_VERSION

        chrys_compaction.annotate_token_counts([message], tokenizer=_LenTokenizer())

        assert chrys_compaction._token_count(message) == 123
        assert chrys_compaction.included_token_count([message]) == 123

    def test_strict_accessors_reject_malformed_annotations(self) -> None:
        message = _user("x")
        message.additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY] = {
            chrys_compaction.GROUP_ID_KEY: "group_1",
            chrys_compaction.GROUP_KIND_KEY: "user",
            # GROUP_INDEX_KEY / GROUP_HAS_REASONING_KEY missing -> invalid.
        }
        assert chrys_compaction._group_id(message) is None
        assert chrys_compaction._group_kind(message) is None
        message.additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY] = {
            chrys_compaction.GROUP_ID_KEY: "group_1",
            chrys_compaction.GROUP_KIND_KEY: "user",
            chrys_compaction.GROUP_INDEX_KEY: 0,
            chrys_compaction.GROUP_HAS_REASONING_KEY: False,
            chrys_compaction.GROUP_TOKEN_COUNT_KEY: "not-an-int",
        }
        assert chrys_compaction._group_id(message) is None
        assert chrys_compaction._group_kind(message) is None
        assert chrys_compaction._group_index(message) is None
        assert chrys_compaction._token_count(message) is None

    def test_write_group_annotation_preserves_unknown_fields_token_count_and_estimator_version(self) -> None:
        messages = [_user("x")]
        chrys_compaction.annotate_message_groups(messages, tokenizer=_LenTokenizer())
        annotation = messages[0].additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY]
        token_count = annotation[chrys_compaction.GROUP_TOKEN_COUNT_KEY]
        estimator_version = annotation[chrys_compaction.GROUP_TOKEN_ESTIMATOR_VERSION_KEY]
        annotation[chrys_compaction.SUMMARIZED_BY_SUMMARY_ID_KEY] = "summary_1"
        chrys_compaction.annotate_message_groups(messages, force_reannotate=True)
        rewritten = messages[0].additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY]
        assert rewritten[chrys_compaction.SUMMARIZED_BY_SUMMARY_ID_KEY] == "summary_1"
        assert rewritten[chrys_compaction.GROUP_TOKEN_COUNT_KEY] == token_count
        assert rewritten[chrys_compaction.GROUP_TOKEN_ESTIMATOR_VERSION_KEY] == estimator_version

    def test_write_token_count_requires_an_existing_annotation(self) -> None:
        message = _user("x")
        chrys_compaction._write_token_count(message, 42)
        assert chrys_compaction.GROUP_ANNOTATION_KEY not in message.additional_properties


class TestExchangeGroupSpans:
    """Span shapes over multi-message exchanges: once an exchange has results,
    every call span and the result block fuse into ONE group — compaction
    excludes group-wise, so a call landing in a different group than its
    result lets a partial exclusion put a result without its call (or a call
    without its result) on the wire, which providers reject outright. Text
    siblings between the first call and the results are absorbed into the
    fused group; text before the first call keeps its own span, and without
    results text still splits call runs. User-role or marker messages never
    extend a tool-call span."""

    @staticmethod
    def _span_tuples(messages: list[Message]) -> list[tuple[str, int, int]]:
        return [
            (span["kind"], span["start_index"], span["end_index"]) for span in chrys_compaction.group_messages(messages)
        ]

    @staticmethod
    def _shared_result_block(*call_ids: str) -> Message:
        return Message(
            role="tool",
            contents=[Content.from_function_result(call_id, result=f"result for {call_id}") for call_id in call_ids],
        )

    def test_sibling_call_run_sharing_result_block_is_one_span(self) -> None:
        messages = [
            _assistant_tool_call("call_a", "first"),
            _assistant_tool_call("call_b", "second"),
            self._shared_result_block("call_a", "call_b"),
        ]

        assert self._span_tuples(messages) == [("tool_call", 0, 2)]

    def test_text_only_assistant_between_call_and_results_joins_the_call_span(self) -> None:
        messages = [
            _assistant_tool_call("call_a", "first"),
            _assistant_text("narration"),
            self._shared_result_block("call_a"),
        ]

        assert self._span_tuples(messages) == [("tool_call", 0, 2)]

    def test_empty_assistant_shell_between_call_and_results_joins_the_call_span(self) -> None:
        messages = [
            _assistant_tool_call("call_a", "first"),
            Message(role="assistant", contents=[]),
            self._shared_result_block("call_a"),
        ]

        assert self._span_tuples(messages) == [("tool_call", 0, 2)]

    def test_result_only_assistant_after_text_sibling_joins_the_call_span(self) -> None:
        messages = [
            _assistant_tool_call("call_a", "first"),
            _assistant_text("narration"),
            Message(role="assistant", contents=[Content.from_function_result("call_a", result="ra")]),
        ]

        assert self._span_tuples(messages) == [("tool_call", 0, 2)]

    def test_mixed_text_and_result_message_is_output_and_joins_the_call_span(self) -> None:
        """An assistant message carrying a result and no call is result-only
        regardless of text riding along, so it lands in the output block and
        fuses with the call span."""
        messages = [
            _assistant_tool_call("call_a", "first"),
            Message(
                role="assistant",
                contents=[
                    Content.from_text("narration"),
                    Content.from_function_result("call_a", result="ra"),
                ],
            ),
        ]

        assert self._span_tuples(messages) == [("tool_call", 0, 1)]

    def test_embedded_results_fuse_split_call_spans_without_an_output_block(self) -> None:
        """Results embedded in a later call-carrying sibling answer earlier
        calls of the exchange, so the split call spans must fuse even though
        no output block exists; trailing text after the last call keeps its
        own span."""
        messages = [
            _assistant_tool_call("call_a", "first"),
            _assistant_text("narration"),
            Message(
                role="assistant",
                contents=[
                    Content.from_function_call("call_b", "second", arguments={}),
                    Content.from_function_result("call_a", result="ra"),
                    Content.from_function_result("call_b", result="rb"),
                ],
            ),
            _assistant_text("trailing"),
        ]

        assert self._span_tuples(messages) == [
            ("tool_call", 0, 2),
            ("assistant_text", 3, 3),
        ]

    def test_incremental_reannotation_fuses_a_result_landing_after_an_annotated_prefix(self) -> None:
        """An annotated ``[call, text]`` prefix must not freeze the split
        grouping: when the result lands and annotation runs again, the
        rewind has to reach back past the exchange's earlier spans so the
        fused group forms — a rewind stopping at the previous group's start
        regroups the suffix without its call and leaves the call excludable
        apart from its result."""
        messages = [
            _user("question"),
            _assistant_tool_call("call_a", "first"),
            _assistant_text("narration"),
        ]
        chrys_compaction.annotate_message_groups(messages)
        messages.append(self._shared_result_block("call_a"))
        chrys_compaction.annotate_message_groups(messages)

        annotations = [message.additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY] for message in messages]
        kinds = [annotation[chrys_compaction.GROUP_KIND_KEY] for annotation in annotations]
        assert kinds == ["user", "tool_call", "tool_call", "tool_call"]
        assert len({annotation[chrys_compaction.GROUP_ID_KEY] for annotation in annotations[1:]}) == 1

    def test_incremental_reannotation_stops_at_a_completed_exchange_boundary(self) -> None:
        """Appending a later exchange's result must regroup only that
        exchange: the completed previous exchange keeps its annotation dicts
        untouched. A rewind crossing finished exchanges rewrites their
        annotations on every pass and turns a long tool loop's incremental
        annotation quadratic."""
        messages = [
            _user("question"),
            _assistant_tool_call("call_a", "first"),
            _tool_result("call_a", "ra"),
            _assistant_tool_call("call_b", "second"),
        ]
        chrys_compaction.annotate_message_groups(messages)
        prior = [messages[index].additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY] for index in range(3)]
        messages.append(_tool_result("call_b", "rb"))
        chrys_compaction.annotate_message_groups(messages)

        for index, annotation in enumerate(prior):
            assert messages[index].additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY] is annotation
        annotations = [message.additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY] for message in messages]
        kinds = [annotation[chrys_compaction.GROUP_KIND_KEY] for annotation in annotations]
        assert kinds == ["user", "tool_call", "tool_call", "tool_call", "tool_call"]
        group_ids = [annotation[chrys_compaction.GROUP_ID_KEY] for annotation in annotations]
        assert group_ids[1] == group_ids[2]
        assert group_ids[3] == group_ids[4]
        assert group_ids[1] != group_ids[3]

    def test_incremental_reannotation_keys_the_rewind_on_the_group_occurrence(self) -> None:
        """Adjacent exchanges can carry the SAME group id when preset
        message_ids collide (group ids are minted from the start message's
        id). The rewind must key on the group occurrence — id AND index —
        or duplicate ids walk it back through the finished exchange before
        the shape-aware boundary check ever runs."""
        call_a = _assistant_tool_call("call_a", "first")
        call_a.message_id = "duplicate"
        call_b = _assistant_tool_call("call_b", "second")
        call_b.message_id = "duplicate"
        messages = [
            _user("question"),
            call_a,
            _tool_result("call_a", "ra"),
            call_b,
        ]
        chrys_compaction.annotate_message_groups(messages)
        assert (
            messages[1].additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY][chrys_compaction.GROUP_ID_KEY]
            == messages[3].additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY][chrys_compaction.GROUP_ID_KEY]
        )
        prior = [messages[index].additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY] for index in range(3)]
        messages.append(_tool_result("call_b", "rb"))
        chrys_compaction.annotate_message_groups(messages)

        for index, annotation in enumerate(prior):
            assert messages[index].additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY] is annotation
        annotations = [message.additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY] for message in messages]
        kinds = [annotation[chrys_compaction.GROUP_KIND_KEY] for annotation in annotations]
        assert kinds == ["user", "tool_call", "tool_call", "tool_call", "tool_call"]
        indices = [annotation[chrys_compaction.GROUP_INDEX_KEY] for annotation in annotations]
        assert indices[1] == indices[2]
        assert indices[3] == indices[4]
        assert indices[1] != indices[3]

    def test_leading_text_only_assistant_stays_out_of_call_span(self) -> None:
        messages = [
            _assistant_text("narration"),
            _assistant_tool_call("call_a", "first"),
            self._shared_result_block("call_a"),
        ]

        assert self._span_tuples(messages) == [
            ("assistant_text", 0, 0),
            ("tool_call", 1, 2),
        ]

    def test_split_call_runs_sharing_a_result_block_fuse_into_one_span(self) -> None:
        """The shared result block answers BOTH calls, so the first call may
        not sit in a separate group: excluding it alone would leave its
        result on the wire without the call."""
        messages = [
            _assistant_tool_call("call_a", "first"),
            _assistant_text("narration"),
            _assistant_tool_call("call_b", "second"),
            self._shared_result_block("call_a", "call_b"),
        ]

        assert self._span_tuples(messages) == [("tool_call", 0, 3)]

    def test_call_runs_without_results_still_split_across_text_only_gap(self) -> None:
        messages = [
            _assistant_tool_call("call_a", "first"),
            _assistant_text("narration"),
            _assistant_tool_call("call_b", "second"),
        ]

        assert self._span_tuples(messages) == [
            ("tool_call", 0, 0),
            ("assistant_text", 1, 1),
            ("tool_call", 2, 2),
        ]

    def test_leading_text_stays_out_while_mid_exchange_text_is_absorbed(self) -> None:
        messages = [
            _assistant_text("preamble"),
            _assistant_tool_call("call_a", "first"),
            _assistant_text("narration"),
            self._shared_result_block("call_a"),
        ]

        assert self._span_tuples(messages) == [
            ("assistant_text", 0, 0),
            ("tool_call", 1, 3),
        ]

    def test_user_role_result_message_does_not_extend_tool_call_span(self) -> None:
        messages = [
            _assistant_tool_call("call_a", "first"),
            Message(role="user", contents=[Content.from_function_result("call_a", result="ra")]),
        ]

        tool_spans = [span for span in self._span_tuples(messages) if span[0] == "tool_call"]
        assert tool_spans == [("tool_call", 0, 0)]

    def test_marker_message_result_does_not_extend_tool_call_span(self) -> None:
        marker = Message(role="assistant", contents=[Content.from_function_result("call_a", result="ra")])
        marker.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
        messages = [_assistant_tool_call("call_a", "first"), marker]

        tool_spans = [span for span in self._span_tuples(messages) if span[0] == "tool_call"]
        assert tool_spans == [("tool_call", 0, 0)]

    def test_partial_group_spanning_user_role_result_degrades_atomic_members(self) -> None:
        """A persisted reasoning tool-call annotation can span a user-role
        message carrying result contents; that message is transcript input
        for occurrence-boundary purposes, so it must not close the occurrence
        — the excluded reasoning follower stays in the same occurrence and
        partial projection degrades the occurrence's atomic members. The
        rider's result content still makes it a pairing carrier (serializers
        emit result blocks by content, regardless of the kernel role that
        carries them), so it degrades with the occurrence instead of
        projecting an orphaned result."""
        call_member = _assistant_reasoning_and_call("call_a", "lookup")
        result_rider = Message(role="user", contents=[Content.from_function_result("call_a", result="ra")])
        follower = _assistant_reasoning("continuation reasoning")
        for message in (call_member, result_rider, follower):
            message.additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY] = {
                chrys_compaction.GROUP_ID_KEY: "group_restored",
                chrys_compaction.GROUP_KIND_KEY: "tool_call",
                chrys_compaction.GROUP_INDEX_KEY: 0,
                chrys_compaction.GROUP_HAS_REASONING_KEY: True,
            }
        chrys_compaction.set_excluded(follower, excluded=True, reason="persisted strategy")

        projected = chrys_compaction.project_included_messages([call_member, result_rider, follower])

        assert projected == []
        assert call_member.additional_properties[chrys_compaction.EXCLUDED_KEY]
        assert result_rider.additional_properties[chrys_compaction.EXCLUDED_KEY]

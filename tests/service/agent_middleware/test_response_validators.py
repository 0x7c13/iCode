# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Rule-level tests for DefaultResponseValidator and the interactions between its rules."""

from __future__ import annotations

import pytest

from chrys.foundation.hosted_tools import HostedToolFamily, HostedToolPhase
from chrys.foundation.trajectory.event_types import ValidationReason
from chrys.kernel import (
    ChatResponse,
    Content,
    Message,
)
from chrys.service.agent_middleware.validators import (
    CONTENT_FILTERED_REASON,
    HOSTED_EVIDENCE_MISSING_FINAL_TEXT_REASON,
    NO_VISIBLE_OUTPUT_REASON,
    OUTPUT_TRUNCATED_REASON,
    REASONING_EXHAUSTED_OUTPUT_REASON,
    UNNAMED_TOOL_CALL_REASON,
    DefaultResponseValidator,
    ValidationResult,
)
from tests.service.agent_middleware._response_validation_fakes import (
    _assistant,
    _assistant_truncated,
    _search_without_final_text,
)

# ---------------------------------------------------------------------------
# DefaultResponseValidator — rule-level tests
# ---------------------------------------------------------------------------


class TestDefaultValidatorRules:
    def test_empty_contents_is_invalid(self) -> None:
        v = DefaultResponseValidator()
        result = v.validate(_assistant([]))
        assert result == ValidationResult.invalid("empty contents")

    def test_text_only_empty_string_is_invalid(self) -> None:
        v = DefaultResponseValidator()
        result = v.validate(_assistant([Content.from_text("")]))
        assert not result.ok
        assert "empty or whitespace" in result.reason

    def test_text_only_whitespace_newlines_is_invalid(self) -> None:
        """contents=[{type=text, text='\\n\\n\\n'}] must be flagged."""
        v = DefaultResponseValidator()
        result = v.validate(_assistant([Content.from_text("\n\n\n")]))
        assert not result.ok
        assert "whitespace" in result.reason

    def test_text_only_spaces_and_tabs_is_invalid(self) -> None:
        v = DefaultResponseValidator()
        result = v.validate(_assistant([Content.from_text("   \t  \n ")]))
        assert not result.ok

    def test_real_text_is_valid(self) -> None:
        v = DefaultResponseValidator()
        result = v.validate(_assistant([Content.from_text("Here is the answer.")]))
        assert result.ok

    def test_tool_call_only_is_valid(self) -> None:
        """A pure tool-calling turn with NO text is perfectly legal."""
        v = DefaultResponseValidator()
        tool_call = Content.from_function_call("call_1", "read_file", arguments={"path": "x"})
        result = v.validate(_assistant([tool_call]))
        assert result.ok

    def test_tool_call_with_empty_text_is_valid(self) -> None:
        """Don't flag empty text when the message has a tool call."""
        v = DefaultResponseValidator()
        tool_call = Content.from_function_call("call_1", "read_file", arguments={"path": "x"})
        result = v.validate(_assistant([Content.from_text(""), tool_call]))
        assert result.ok

    def test_informational_custom_tool_call_only_is_valid(self) -> None:
        """A preserved non-executable Responses custom call is still output."""
        call = Content.from_function_call(
            "call_custom_1",
            "python",
            arguments="print('hi')",
            informational_only=True,
            additional_properties={"item_type": "custom_tool_call"},
        )

        assert DefaultResponseValidator().validate(_assistant([call])).ok

    def test_terminal_hosted_only_response_is_valid(self) -> None:
        """A provider-hosted terminal item is usable output without text."""
        v = DefaultResponseValidator()
        hosted = Content.from_hosted_tool_result(
            "hosted_1",
            tool_name="server_task",
            status="completed",
            provider_phase=HostedToolPhase.TERMINAL,
            provider_status="completed",
            result="done",
        )

        assert v.validate(_assistant([hosted])).ok

    def test_running_hosted_only_response_is_not_terminal_output(self) -> None:
        v = DefaultResponseValidator()
        hosted = Content.from_hosted_tool_call(
            "hosted_1",
            tool_name="server_task",
            status="running",
            provider_phase=HostedToolPhase.START,
            provider_status="running",
        )

        result = v.validate(_assistant([hosted]))

        assert not result.ok

    def test_terminal_search_after_intermediate_text_requires_final_answer(self) -> None:
        result = DefaultResponseValidator().validate(_search_without_final_text())

        assert result == ValidationResult.invalid(
            HOSTED_EVIDENCE_MISSING_FINAL_TEXT_REASON,
        )

    @pytest.mark.parametrize(
        "family",
        [HostedToolFamily.SEARCH, HostedToolFamily.FETCH, HostedToolFamily.TOOL_DISCOVERY],
    )
    @pytest.mark.parametrize("content_type", ["call", "result"])
    def test_terminal_evidence_only_hosted_content_requires_final_answer(
        self,
        family: HostedToolFamily,
        content_type: str,
    ) -> None:
        if content_type == "call":
            content = Content.from_hosted_tool_call(
                "evidence_1",
                tool_name="provider_tool",
                hosted_family=family,
                status="completed",
                provider_phase=HostedToolPhase.TERMINAL,
                provider_status="completed",
            )
        else:
            content = Content.from_hosted_tool_result(
                "evidence_1",
                tool_name="provider_tool",
                hosted_family=family,
                status="completed",
                provider_phase=HostedToolPhase.TERMINAL,
                provider_status="completed",
                result={"evidence": "found"},
            )

        assert DefaultResponseValidator().validate(_assistant([content])) == ValidationResult.invalid(
            HOSTED_EVIDENCE_MISSING_FINAL_TEXT_REASON,
        )

    def test_terminal_tool_discovery_followed_by_final_text_is_valid(self) -> None:
        discovery = Content.from_hosted_tool_result(
            "discovery_1",
            tool_name="tool_search",
            hosted_family=HostedToolFamily.TOOL_DISCOVERY,
            status="completed",
            provider_phase=HostedToolPhase.TERMINAL,
            provider_status="completed",
            result={"tools": [{"name": "get_weather"}]},
        )

        assert (
            DefaultResponseValidator()
            .validate(_assistant([discovery, Content.from_text("I found the appropriate tool.")]))
            .ok
        )

    def test_terminal_search_followed_by_final_text_is_valid(self) -> None:
        response = _search_without_final_text()
        response.messages[0].contents.append(Content.from_text("Here is the answer."))

        assert DefaultResponseValidator().validate(response).ok

    def test_terminal_search_followed_by_image_result_is_valid(self) -> None:
        response = _search_without_final_text()
        response.messages[0].contents.append(
            Content.from_image_generation_tool_result(
                image_id="image_1",
                outputs=["data:image/png;base64,AA=="],
                provider_phase=HostedToolPhase.TERMINAL,
                provider_status="completed",
            )
        )

        assert DefaultResponseValidator().validate(response).ok

    def test_local_function_call_before_terminal_search_is_valid(self) -> None:
        response = _search_without_final_text()
        response.messages[0].contents[0] = Content.from_function_call(
            "call_1", "read_file", arguments={"path": "README.md"}
        )

        assert DefaultResponseValidator().validate(response).ok

    # -- Text is never judged by what it says ------------------------------

    @pytest.mark.parametrize(
        "text",
        [
            'minimax:tool_call {"name":"read_file"} </minimax:tool_call>',
            "Here is my plan.\nminimax:tool_call {} </minimax:tool_call>\nDone.",
            "<tool_use name='read_file'>...</tool_use>",
            "<function_call>...</function_call>",
            "The MiniMax model was released in 2024.",
        ],
        ids=["minimax_marker", "marker_in_prose", "tool_use_tag", "function_call_tag", "prose"],
    )
    def test_tool_call_markers_in_text_are_answer_text(self, text: str) -> None:
        assert DefaultResponseValidator().validate(_assistant([Content.from_text(text)])).ok

    def test_empty_response_no_messages(self) -> None:
        """ChatResponse with no messages at all."""
        v = DefaultResponseValidator()
        result = v.validate(ChatResponse(messages=[]))
        assert not result.ok

    def test_length_finish_reason_empty_contents_is_terminal(self) -> None:
        # Empty output + finish_reason="length" → no room to reply; terminal.
        v = DefaultResponseValidator()
        result = v.validate(_assistant_truncated([]))
        assert result == ValidationResult.invalid(OUTPUT_TRUNCATED_REASON)

    def test_length_finish_reason_whitespace_text_is_terminal(self) -> None:
        v = DefaultResponseValidator()
        result = v.validate(_assistant_truncated([Content.from_text("\n\n  \t")]))
        assert not result.ok
        assert result.reason == OUTPUT_TRUNCATED_REASON

    def test_length_finish_reason_with_real_text_is_valid(self) -> None:
        # A response truncated mid-sentence that still has content is valid —
        # "length" alone must not fabricate a failure (no false positive).
        v = DefaultResponseValidator()
        result = v.validate(_assistant_truncated([Content.from_text("partial answer")]))
        assert result.ok

    def test_length_finish_reason_with_reasoning_text_is_reasoning_exhausted(self) -> None:
        # Reasoning-only output cut off by the output budget means the model
        # over-thought — the generated reasoning proves the input did NOT fill
        # the context window, so the reason points at the output budget.
        v = DefaultResponseValidator()
        result = v.validate(_assistant_truncated([Content.from_text_reasoning(text="partial thought")]))
        assert result == ValidationResult.invalid(REASONING_EXHAUSTED_OUTPUT_REASON)

    def test_length_finish_reason_with_protected_reasoning_is_reasoning_exhausted(self) -> None:
        # Signed/private reasoning is generated content too: the length cutoff
        # came from over-thinking, not a full context window.
        v = DefaultResponseValidator()
        result = v.validate(_assistant_truncated([Content.from_text_reasoning(protected_data="sig-123")]))
        assert result == ValidationResult.invalid(REASONING_EXHAUSTED_OUTPUT_REASON)

    def test_stop_finish_reason_with_reasoning_text_has_no_visible_output(self) -> None:
        # Reasoning-only stop has no user-visible answer: the turn fails with
        # the reason rather than a blank final AgentMessage.
        v = DefaultResponseValidator()
        result = v.validate(_assistant([Content.from_text_reasoning(text="complete thought")]))
        assert result == ValidationResult.invalid(NO_VISIBLE_OUTPUT_REASON)

    def test_stop_finish_reason_with_protected_reasoning_has_no_visible_output(self) -> None:
        v = DefaultResponseValidator()
        result = v.validate(_assistant([Content.from_text_reasoning(protected_data="sig-123")]))
        assert result == ValidationResult.invalid(NO_VISIBLE_OUTPUT_REASON)

    def test_reasoning_with_visible_text_is_valid(self) -> None:
        v = DefaultResponseValidator()
        result = v.validate(
            _assistant(
                [
                    Content.from_text_reasoning(text="complete thought"),
                    Content.from_text("visible answer"),
                ]
            )
        )
        assert result.ok


# ---------------------------------------------------------------------------
# Content filter without an answer
# ---------------------------------------------------------------------------


def _filtered(contents: list[Content]) -> ChatResponse:
    return ChatResponse(messages=[Message(role="assistant", contents=contents)], finish_reason="content_filter")


def _terminal_search_result() -> Content:
    return Content.from_search_tool_result(
        "ws_1",
        tool_name="web_search",
        status="completed",
        provider_phase=HostedToolPhase.TERMINAL,
        provider_status="completed",
        result={"query": "Chrys"},
    )


def _image_result() -> Content:
    return Content.from_image_generation_tool_result(
        image_id="image_1",
        outputs=["data:image/png;base64,AA=="],
        provider_phase=HostedToolPhase.TERMINAL,
        provider_status="completed",
    )


_FILTERED = ValidationResult.invalid(CONTENT_FILTERED_REASON)


class TestContentFilteredWithoutAnswer:
    @pytest.mark.parametrize(
        "response",
        [
            pytest.param(ChatResponse(messages=[], finish_reason="content_filter"), id="no_message"),
            pytest.param(_filtered([]), id="empty_contents"),
            pytest.param(_filtered([Content.from_text(" \n ")]), id="whitespace_text"),
            pytest.param(_filtered([Content.from_text_reasoning(text="thinking")]), id="reasoning_only"),
            pytest.param(
                _filtered(
                    [
                        Content.from_hosted_tool_call(
                            "hosted_1",
                            tool_name="server_task",
                            status="running",
                            provider_phase=HostedToolPhase.START,
                            provider_status="running",
                        )
                    ]
                ),
                id="running_hosted_work",
            ),
            pytest.param(_filtered([_terminal_search_result()]), id="evidence_only"),
            pytest.param(
                _filtered([Content.from_text("Checking sources."), _terminal_search_result()]),
                id="text_before_evidence",
            ),
        ],
    )
    def test_is_reported_as_filtered(self, response: ChatResponse) -> None:
        result = DefaultResponseValidator().validate(response)

        assert result == _FILTERED
        assert result.code == ValidationReason.CONTENT_FILTERED

    @pytest.mark.parametrize(
        "contents",
        [
            pytest.param([Content.from_text("Part of the answer")], id="visible_text"),
            pytest.param(
                [Content.from_function_call("call_1", "read_file", arguments={"path": "README.md"})],
                id="local_call",
            ),
            pytest.param(
                [
                    Content.from_function_call("call_1", "read_file", arguments={"path": "README.md"}),
                    _terminal_search_result(),
                ],
                id="local_call_before_evidence",
            ),
            pytest.param([_image_result()], id="answer_bearing_hosted_output"),
            pytest.param(
                [_terminal_search_result(), Content.from_text("Here is the answer.")], id="answer_after_evidence"
            ),
        ],
    )
    def test_an_answer_the_filter_cut_short_is_kept(self, contents: list[Content]) -> None:
        assert DefaultResponseValidator().validate(_filtered(contents)).ok

    def test_disabled_rules_stay_disabled(self) -> None:
        no_op = DefaultResponseValidator(
            disable_empty_contents=True, disable_whitespace_text=True, disable_unnamed_tool_call=True
        )
        whitespace_allowed = DefaultResponseValidator(disable_whitespace_text=True)
        empty_allowed = DefaultResponseValidator(disable_empty_contents=True)

        assert no_op.validate(_filtered([])).ok
        assert whitespace_allowed.validate(_filtered([Content.from_text(" ")])).ok
        assert whitespace_allowed.validate(_filtered([])) == _FILTERED
        # The whitespace rule still reads an empty message as blank.
        assert empty_allowed.validate(_filtered([])) == _FILTERED

    def test_other_finish_reasons_keep_their_rules(self) -> None:
        empty = DefaultResponseValidator().validate(_assistant([]))

        assert empty == ValidationResult.invalid("empty contents")
        assert empty.code == ValidationReason.EMPTY_CONTENTS


# ---------------------------------------------------------------------------
# Tool call without a function name
# ---------------------------------------------------------------------------


def _unnamed_call(name: str = "") -> Content:
    return Content.from_function_call("call_1", name, arguments={"path": "x"})


class TestToolCallWithoutAName:
    @pytest.mark.parametrize(
        "contents",
        [
            pytest.param([_unnamed_call()], id="alone"),
            pytest.param([Content.from_text("Reading it."), _unnamed_call()], id="after_text"),
            pytest.param(
                [Content.from_function_call("call_0", "read_file", arguments={"path": "a"}), _unnamed_call()],
                id="beside_a_named_call",
            ),
        ],
    )
    def test_a_response_with_a_call_that_names_no_function_is_rejected(self, contents: list[Content]) -> None:
        result = DefaultResponseValidator().validate(_assistant(contents))

        assert result == ValidationResult.invalid(UNNAMED_TOOL_CALL_REASON)
        assert result.code == ValidationReason.UNNAMED_TOOL_CALL

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(
                Content.from_function_call("call_custom_1", "", arguments="x", informational_only=True),
                id="informational",
            ),
            pytest.param(
                Content(
                    "function_call", call_id="hosted_1", name="", provider_hosted=True, provider_status="completed"
                ),
                id="provider_hosted",
            ),
        ],
    )
    def test_a_call_no_tool_loop_runs_needs_no_name(self, call: Content) -> None:
        assert call.provider_hosted or call.informational_only
        result = DefaultResponseValidator().validate(_assistant([call, Content.from_text("Done.")]))

        assert result.ok

    def test_the_content_filter_outranks_a_call_without_a_name(self) -> None:
        assert DefaultResponseValidator().validate(_filtered([_unnamed_call()])) == _FILTERED

    def test_the_rule_can_be_disabled(self) -> None:
        assert DefaultResponseValidator(disable_unnamed_tool_call=True).validate(_assistant([_unnamed_call()])).ok


# ---------------------------------------------------------------------------
# Rule-interaction edge cases
# ---------------------------------------------------------------------------


class TestRuleInteractions:
    def test_multiple_text_fragments_concat_for_whitespace_check(self) -> None:
        """contents=[text='', text='\\n', text='   '] — all blank together, so the response has no answer."""
        response = _assistant([Content.from_text(""), Content.from_text("\n"), Content.from_text("   ")])

        result = DefaultResponseValidator().validate(response)

        assert result == ValidationResult.invalid("empty or whitespace-only text response")
        assert result.code == ValidationReason.WHITESPACE_ONLY

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for persisted approval metadata in session history."""

from __future__ import annotations

from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.tool_invocation_order import TOOL_INVOCATION_ORDER_KEY
from chrys.foundation.tool_result_metadata import TOOL_ERROR_KIND_METADATA_KEY
from chrys.kernel import Content, Message
from chrys.service.session.history import SessionHistoryManager


def test_persist_approval_decisions_preserves_reason() -> None:
    """Approval decisions with a reason should persist on the assistant tool-call message."""
    function_call = Content.from_function_call(call_id="call_1", name="zsh", arguments={"command": "rm file"})
    assistant = Message("assistant", [function_call])
    history = SessionHistoryManager()
    history.bind({"messages": [Message("user", ["run command"]), assistant]})

    history.persist_approval_decisions(
        [
            {
                "request_id": "req-1",
                "tool_name": "zsh",
                "status": "user_rejected",
                "reason": "Use a narrower command",
            }
        ]
    )

    assert assistant.additional_properties["_approval"] == {
        "tool_name": "zsh",
        "status": "user_rejected",
        "request_id": "req-1",
        "reason": "Use a narrower command",
    }
    assert function_call.additional_properties["_approval"] == assistant.additional_properties["_approval"]


def test_persist_approval_decisions_updates_modified_arguments() -> None:
    """Approval-edited args should persist on the matching function_call."""
    function_call = Content.from_function_call(
        call_id="call_1",
        name="explore_agent",
        arguments={"prompt": "old prompt"},
    )
    assistant = Message("assistant", [function_call])
    history = SessionHistoryManager()
    history.bind({"messages": [Message("user", ["delegate"]), assistant]})

    history.persist_approval_decisions(
        [
            {
                "request_id": "req-1",
                "tool_name": "explore_agent",
                "status": "user_approved",
                "modified_args": '{"prompt": "new prompt"}',
            }
        ]
    )

    assert function_call.arguments == {"prompt": "new prompt"}
    assert function_call.additional_properties["_approval"] == {
        "tool_name": "explore_agent",
        "status": "user_approved",
        "request_id": "req-1",
    }


def test_persist_approval_decisions_tags_each_function_call() -> None:
    """Multiple same-name tool calls in one message keep distinct decisions."""
    first_call = Content.from_function_call(call_id="call_1", name="zsh", arguments={"command": "wc -l a.py"})
    second_call = Content.from_function_call(call_id="call_2", name="zsh", arguments={"command": "wc -l b.py"})
    assistant = Message("assistant", [first_call, second_call])
    history = SessionHistoryManager()
    history.bind({"messages": [Message("user", ["count lines"]), assistant]})

    history.persist_approval_decisions(
        [
            {"request_id": "req-1", "tool_name": "zsh", "status": "user_approved"},
            {"request_id": "req-2", "tool_name": "zsh", "status": "user_rejected"},
        ]
    )

    assert first_call.additional_properties["_approval"] == {
        "tool_name": "zsh",
        "status": "user_approved",
        "request_id": "req-1",
    }
    assert second_call.additional_properties["_approval"] == {
        "tool_name": "zsh",
        "status": "user_rejected",
        "request_id": "req-2",
    }
    assert assistant.additional_properties["_approval"] == first_call.additional_properties["_approval"]


def test_persist_approval_decisions_keeps_alignment_after_auto_approved_same_name_call() -> None:
    """Neutral auto-approval placeholders prevent same-name decision shifting."""
    safe_call = Content.from_function_call(call_id="safe_call", name="zsh", arguments={"command": "ls"})
    safe_assistant = Message("assistant", [safe_call])
    reject_call = Content.from_function_call(call_id="reject_call", name="zsh", arguments={"command": "rm file"})
    reject_assistant = Message("assistant", [reject_call])
    history = SessionHistoryManager()
    history.bind(
        {
            "messages": [
                Message("user", ["run both"]),
                safe_assistant,
                Message("tool", [Content.from_function_result(call_id="safe_call", result="file.py")]),
                reject_assistant,
                Message(
                    "tool",
                    [Content.from_function_result(call_id="reject_call", result="Error: Tool execution was rejected")],
                ),
            ]
        }
    )

    history.persist_approval_decisions(
        [
            {"request_id": "", "tool_name": "zsh", "status": "auto_approved"},
            {"request_id": "req-1", "tool_name": "zsh", "status": "user_rejected"},
        ]
    )

    assert safe_call.additional_properties["_approval"] == {
        "tool_name": "zsh",
        "status": "auto_approved",
        "request_id": "",
    }
    assert reject_call.additional_properties["_approval"] == {
        "tool_name": "zsh",
        "status": "user_rejected",
        "request_id": "req-1",
    }
    assert safe_assistant.additional_properties["_approval"] == safe_call.additional_properties["_approval"]
    assert reject_assistant.additional_properties["_approval"] == reject_call.additional_properties["_approval"]


def test_persist_approval_decisions_prefers_human_status_for_legacy_message_field() -> None:
    """Message-level legacy metadata should not hide a later human decision."""
    safe_call = Content.from_function_call(call_id="safe_call", name="zsh", arguments={"command": "ls"})
    reject_call = Content.from_function_call(call_id="reject_call", name="zsh", arguments={"command": "rm file"})
    assistant = Message("assistant", [safe_call, reject_call])
    history = SessionHistoryManager()
    history.bind({"messages": [Message("user", ["run both"]), assistant]})

    history.persist_approval_decisions(
        [
            {"request_id": "", "tool_name": "zsh", "status": "auto_approved"},
            {"request_id": "req-1", "tool_name": "zsh", "status": "user_rejected"},
        ]
    )

    assert safe_call.additional_properties["_approval"]["status"] == "auto_approved"
    assert reject_call.additional_properties["_approval"]["status"] == "user_rejected"
    assert assistant.additional_properties["_approval"] == reject_call.additional_properties["_approval"]


def test_persist_approval_decisions_matches_call_id_before_order() -> None:
    """Out-of-order same-name decisions should attach by call_id."""
    unsafe_call = Content.from_function_call(call_id="unsafe-call", name="zsh", arguments={"command": "rm file"})
    safe_call = Content.from_function_call(call_id="safe-call", name="zsh", arguments={"command": "ls"})
    assistant = Message("assistant", [unsafe_call, safe_call])
    history = SessionHistoryManager()
    history.bind({"messages": [Message("user", ["run both"]), assistant]})

    history.persist_approval_decisions(
        [
            {"request_id": "", "tool_name": "zsh", "status": "auto_approved", "call_id": "safe-call"},
            {"request_id": "req-1", "tool_name": "zsh", "status": "user_rejected", "call_id": "unsafe-call"},
        ]
    )

    assert unsafe_call.additional_properties["_approval"] == {
        "tool_name": "zsh",
        "status": "user_rejected",
        "request_id": "req-1",
        "call_id": "unsafe-call",
    }
    assert safe_call.additional_properties["_approval"] == {
        "tool_name": "zsh",
        "status": "auto_approved",
        "request_id": "",
        "call_id": "safe-call",
    }
    assert assistant.additional_properties["_approval"] == unsafe_call.additional_properties["_approval"]


def test_persist_approval_decisions_matches_tool_order_before_decision_order() -> None:
    """Parallel same-name decisions should attach by the kernel-stamped ordinal.

    Decision call_ids deliberately diverge from content call_ids: live
    decisions carry the chrys-minted id while history contents keep the
    provider wire id, so order matching must succeed on the stamp alone —
    decision LIST order (safe first) must not win over the stamped
    invocation order.
    """
    unsafe_call = Content.from_function_call(call_id="framework-unsafe", name="zsh", arguments={"command": "rm file"})
    unsafe_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 0
    safe_call = Content.from_function_call(call_id="framework-safe", name="zsh", arguments={"command": "ls"})
    safe_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 1
    assistant = Message("assistant", [unsafe_call, safe_call])
    history = SessionHistoryManager()
    history.bind({"messages": [Message("user", ["run both"]), assistant]})

    history.persist_approval_decisions(
        [
            {
                "request_id": "",
                "tool_name": "zsh",
                "status": "auto_approved",
                "call_id": "live-safe",
                "tool_order": "1",
            },
            {
                "request_id": "req-1",
                "tool_name": "zsh",
                "status": "user_rejected",
                "call_id": "live-unsafe",
                "tool_order": "0",
            },
        ]
    )

    assert unsafe_call.additional_properties["_approval"] == {
        "tool_name": "zsh",
        "status": "user_rejected",
        "request_id": "req-1",
    }
    assert safe_call.additional_properties["_approval"] == {
        "tool_name": "zsh",
        "status": "auto_approved",
        "request_id": "",
    }
    assert assistant.additional_properties["_approval"] == unsafe_call.additional_properties["_approval"]


def test_persist_approval_decisions_order_match_survives_skipped_ordinal() -> None:
    """Stamped ordinals match verbatim even when an earlier ordinal never reached approval."""
    first_call = Content.from_function_call(call_id="c-first", name="zsh", arguments={"command": "ls"})
    first_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 1
    second_call = Content.from_function_call(call_id="c-second", name="zsh", arguments={"command": "rm file"})
    second_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 2
    assistant = Message("assistant", [first_call, second_call])
    history = SessionHistoryManager()
    history.bind({"messages": [Message("user", ["run both"]), assistant]})

    history.persist_approval_decisions(
        [
            {"request_id": "req-2", "tool_name": "zsh", "status": "user_rejected", "tool_order": "2"},
            {"request_id": "", "tool_name": "zsh", "status": "auto_approved", "tool_order": "1"},
        ]
    )

    assert first_call.additional_properties["_approval"]["status"] == "auto_approved"
    assert second_call.additional_properties["_approval"]["status"] == "user_rejected"


def test_persist_approval_decisions_order_match_prefers_most_recent_pass() -> None:
    """An earlier pass's retained same-ordinal call must not steal the decision.

    Ordinals restart at zero for every kernel run, and a retried/resumed
    turn retains the failed pass's completed messages in the same turn
    region — including calls that never reached the approval middleware
    (pre-validation failures). The decision belongs to the most recent
    pass, so the LAST stamp match wins. Decision call ids cannot help:
    they are chrys-minted while contents keep provider wire ids.
    """
    leftover_call = Content.from_function_call(call_id="call_00", name="zsh", arguments="{invalid json")
    leftover_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 0
    retry_call = Content.from_function_call(call_id="call_00", name="zsh", arguments={"command": "rm file"})
    retry_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 0
    retry_assistant = Message("assistant", [retry_call])
    history = SessionHistoryManager()
    history.bind(
        {
            "messages": [
                Message("user", ["run it"]),
                Message("assistant", [leftover_call]),
                Message("tool", [Content.from_function_result(call_id="call_00", result="Error: invalid arguments")]),
                retry_assistant,
                Message("tool", [Content.from_function_result(call_id="call_00", result="done")]),
            ]
        }
    )

    history.persist_approval_decisions(
        [
            {
                "request_id": "req-1",
                "tool_name": "zsh",
                "status": "user_approved",
                "call_id": "call_00",
                "tool_order": "0",
                "modified_args": '{"command": "rm -i file"}',
            }
        ]
    )

    assert "_approval" not in leftover_call.additional_properties
    assert leftover_call.arguments == "{invalid json"
    assert retry_call.additional_properties["_approval"]["status"] == "user_approved"
    assert retry_call.arguments == {"command": "rm -i file"}
    assert retry_assistant.additional_properties["_approval"] == retry_call.additional_properties["_approval"]


def test_persist_approval_decisions_skips_prevalidation_failed_call_in_any_order() -> None:
    """A pre-pipeline-failed call never receives a decision, even when it sits last.

    The empty-retry repair path can insert the newest pass BEFORE an earlier
    pass's retained messages (the synthetic continuation anchor was removed
    before the merge), so positional recency alone cannot be trusted. A call
    whose immediate result carries a pre-pipeline ``tool_error_kind`` is
    excluded from matching outright.
    """
    retry_call = Content.from_function_call(call_id="call_00", name="zsh", arguments={"command": "rm file"})
    retry_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 0
    invalid_call = Content.from_function_call(call_id="call_00", name="zsh", arguments="{invalid json")
    invalid_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 0
    invalid_result = Content.from_function_result(call_id="call_00", result="Error: invalid arguments")
    invalid_result.additional_properties[TOOL_ERROR_KIND_METADATA_KEY] = "argument_parsing"
    history = SessionHistoryManager()
    # Newest pass FIRST (inverted repair layout), earlier pass's invalid call last.
    history.bind(
        {
            "messages": [
                Message("user", ["run it"]),
                Message("assistant", [retry_call]),
                Message("tool", [Content.from_function_result(call_id="call_00", result="done")]),
                Message("assistant", [invalid_call]),
                Message("tool", [invalid_result]),
            ]
        }
    )

    history.persist_approval_decisions(
        [
            {
                "request_id": "req-1",
                "tool_name": "zsh",
                "status": "user_approved",
                "call_id": "call_00",
                "tool_order": "0",
                "modified_args": '{"command": "rm -i file"}',
            }
        ]
    )

    assert "_approval" not in invalid_call.additional_properties
    assert invalid_call.arguments == "{invalid json"
    assert retry_call.additional_properties["_approval"]["status"] == "user_approved"
    assert retry_call.arguments == {"command": "rm -i file"}


def test_persist_approval_decisions_skips_failure_behind_consecutive_assistant_block() -> None:
    """The failure result can sit behind ANOTHER assistant message of the response.

    The kernel folds all of a response's tool results into ONE tool message
    appended after the response's complete message list, so a response with
    several assistant messages persists as ``[assistant, assistant, tool]``
    and the failure result answering the FIRST assistant's call lives behind
    the second. The failed-call scan must read the block after the whole
    consecutive assistant run — otherwise, in the inverted repair layout,
    the failure goes undiscovered and last-match-wins hands the real
    approval (and its modified arguments) to the prevalidation-failed call.
    """
    retry_call = Content.from_function_call(call_id="call_00", name="zsh", arguments={"command": "rm file"})
    retry_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 0
    invalid_call = Content.from_function_call(call_id="call_00", name="zsh", arguments="{invalid json")
    invalid_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 0
    invalid_result = Content.from_function_result(call_id="call_00", result="Error: invalid arguments")
    invalid_result.additional_properties[TOOL_ERROR_KIND_METADATA_KEY] = "argument_parsing"
    other_call = Content.from_function_call(call_id="call_01", name="grep", arguments={"pattern": "x"})
    other_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 1
    history = SessionHistoryManager()
    # Newest pass FIRST (inverted repair layout); the earlier failed pass is a
    # multi-message response sharing one combined result block.
    history.bind(
        {
            "messages": [
                Message("user", ["run it"]),
                Message("assistant", [retry_call]),
                Message("tool", [Content.from_function_result(call_id="call_00", result="done")]),
                Message("assistant", [invalid_call]),
                Message("assistant", [other_call]),
                Message(
                    "tool",
                    [invalid_result, Content.from_function_result(call_id="call_01", result="matches")],
                ),
            ]
        }
    )

    history.persist_approval_decisions(
        [
            {
                "request_id": "req-1",
                "tool_name": "zsh",
                "status": "user_approved",
                "call_id": "call_00",
                "tool_order": "0",
                "modified_args": '{"command": "rm -i file"}',
            }
        ]
    )

    assert "_approval" not in invalid_call.additional_properties
    assert invalid_call.arguments == "{invalid json"
    assert "_approval" not in other_call.additional_properties
    assert retry_call.additional_properties["_approval"]["status"] == "user_approved"
    assert retry_call.arguments == {"command": "rm -i file"}


def test_persist_approval_decisions_skips_every_failure_of_a_multi_failure_block() -> None:
    """A parallel batch can fail several calls at once; ALL of them are excluded.

    The failed-call scan collects every prevalidation failure of the block,
    across mixed error kinds. With the inverted repair layout the earlier
    pass's failed twin sits last, so if even one failure escaped the
    exclusion it would absorb a decision by positional recency — and a
    decision aimed at a failed call's ordinal must attach nowhere at all.
    """
    retry_call = Content.from_function_call(call_id="call_00", name="zsh", arguments={"command": "rm file"})
    retry_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 0
    bad_first = Content.from_function_call(call_id="call_00", name="zsh", arguments="{invalid json")
    bad_first.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 0
    bad_second = Content.from_function_call(call_id="call_01", name="zsh", arguments={"command": "rm other"})
    bad_second.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 1
    err_first = Content.from_function_result(call_id="call_00", result="Error: invalid arguments")
    err_first.additional_properties[TOOL_ERROR_KIND_METADATA_KEY] = "argument_parsing"
    err_second = Content.from_function_result(call_id="call_01", result="Error: not found")
    err_second.additional_properties[TOOL_ERROR_KIND_METADATA_KEY] = "tool_not_found"
    history = SessionHistoryManager()
    # Newest pass FIRST (inverted repair layout), the multi-failure block last.
    history.bind(
        {
            "messages": [
                Message("user", ["run it"]),
                Message("assistant", [retry_call]),
                Message("tool", [Content.from_function_result(call_id="call_00", result="done")]),
                Message("assistant", [bad_first, bad_second]),
                Message("tool", [err_first, err_second]),
            ]
        }
    )

    history.persist_approval_decisions(
        [
            {
                "request_id": "req-1",
                "tool_name": "zsh",
                "status": "user_approved",
                "call_id": "call_00",
                "tool_order": "0",
            },
            {
                "request_id": "req-2",
                "tool_name": "zsh",
                "status": "user_approved",
                "call_id": "call_01",
                "tool_order": "1",
            },
        ]
    )

    assert "_approval" not in bad_first.additional_properties
    assert "_approval" not in bad_second.additional_properties
    approval = retry_call.additional_properties["_approval"]
    assert approval["status"] == "user_approved"
    assert approval["request_id"] == "req-1"


def test_persist_approval_decisions_excludes_none_call_id_call() -> None:
    """A call whose call id is None never competes for decisions.

    The kernel raises on a None call id before the middleware context is
    created, so such a call can never legitimately hold a decision — yet
    without the exclusion, last-match-wins would hand this same-(ordinal,
    name) leftover the real call's decision and modified arguments.
    """
    valid_call = Content.from_function_call(call_id="call_00", name="zsh", arguments={"command": "rm file"})
    valid_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 0
    none_id_call = Content.from_function_call(call_id="unset", name="zsh", arguments={"command": "ls"})
    none_id_call.call_id = None
    none_id_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 0
    history = SessionHistoryManager()
    # The id-less leftover sits LAST, where last-match-wins would pick it.
    history.bind(
        {
            "messages": [
                Message("user", ["run it"]),
                Message("assistant", [valid_call]),
                Message("tool", [Content.from_function_result(call_id="call_00", result="done")]),
                Message("assistant", [none_id_call]),
            ]
        }
    )

    history.persist_approval_decisions(
        [
            {
                "request_id": "req-1",
                "tool_name": "zsh",
                "status": "user_approved",
                "call_id": "call_00",
                "tool_order": "0",
                "modified_args": '{"command": "rm -i file"}',
            }
        ]
    )

    assert "_approval" not in none_id_call.additional_properties
    assert valid_call.additional_properties["_approval"]["status"] == "user_approved"
    assert valid_call.arguments == {"command": "rm -i file"}


def test_persist_approval_decisions_excludes_empty_id_call_with_unpairable_failure() -> None:
    """An empty-id call sharing its block with an id-less failure fails closed.

    A pre-pipeline failure result without a call id can only answer an
    id-less call in the same block, but cannot be paired to a specific one —
    so the block's empty-id calls are excluded rather than risk handing one
    the real call's decision via a same-(ordinal, name) last-match.
    """
    valid_call = Content.from_function_call(call_id="call_00", name="zsh", arguments={"command": "rm file"})
    valid_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 0
    idless_call = Content.from_function_call(call_id="", name="zsh", arguments="{invalid json")
    idless_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 0
    idless_result = Content.from_function_result(call_id="", result="Error: invalid arguments")
    idless_result.additional_properties[TOOL_ERROR_KIND_METADATA_KEY] = "argument_parsing"
    history = SessionHistoryManager()
    # The id-less leftover sits LAST, where last-match-wins would pick it.
    history.bind(
        {
            "messages": [
                Message("user", ["run it"]),
                Message("assistant", [valid_call]),
                Message("tool", [Content.from_function_result(call_id="call_00", result="done")]),
                Message("assistant", [idless_call]),
                Message("tool", [idless_result]),
            ]
        }
    )

    history.persist_approval_decisions(
        [
            {
                "request_id": "req-1",
                "tool_name": "zsh",
                "status": "user_approved",
                "call_id": "call_00",
                "tool_order": "0",
                "modified_args": '{"command": "rm -i file"}',
            }
        ]
    )

    assert "_approval" not in idless_call.additional_properties
    assert idless_call.arguments == "{invalid json"
    assert valid_call.additional_properties["_approval"]["status"] == "user_approved"
    assert valid_call.arguments == {"command": "rm -i file"}


def test_persist_approval_decisions_attaches_order_decision_to_empty_id_call() -> None:
    """An empty-id call that executed normally keeps its decision.

    Provider adapters mint ``""`` when the wire response carries no tool-call
    id; such calls pass the kernel's missing-id check, reach the approval
    middleware, and execute — excluding them wholesale would silently drop a
    real decision and its modified-arguments audit trail.
    """
    empty_id_call = Content.from_function_call(call_id="", name="zsh", arguments={"text": "original"})
    empty_id_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 0
    assistant = Message("assistant", [empty_id_call])
    history = SessionHistoryManager()
    history.bind(
        {
            "messages": [
                Message("user", ["run it"]),
                assistant,
                Message("tool", [Content.from_function_result(call_id="", result="done")]),
            ]
        }
    )

    history.persist_approval_decisions(
        [
            {
                "request_id": "req-1",
                "tool_name": "zsh",
                "status": "user_approved",
                "tool_order": "0",
                "modified_args": '{"text": "edited"}',
            }
        ]
    )

    assert empty_id_call.additional_properties["_approval"]["status"] == "user_approved"
    assert empty_id_call.arguments == {"text": "edited"}


def test_persist_approval_decisions_unstamped_content_fails_closed_for_order_decisions() -> None:
    """Unstamped legacy content skips the order stage; order-carrying decisions never fall back by name."""
    unstamped_call = Content.from_function_call(call_id="legacy-call", name="zsh", arguments={"command": "ls"})
    assistant = Message("assistant", [unstamped_call])
    history = SessionHistoryManager()
    history.bind({"messages": [Message("user", ["run"]), assistant]})

    history.persist_approval_decisions(
        [
            {"request_id": "req-1", "tool_name": "zsh", "status": "user_rejected", "tool_order": "0"},
            {
                "request_id": "req-2",
                "tool_name": "zsh",
                "status": "user_approved",
                "call_id": "legacy-call",
            },
        ]
    )

    assert unstamped_call.additional_properties["_approval"] == {
        "tool_name": "zsh",
        "status": "user_approved",
        "request_id": "req-2",
        "call_id": "legacy-call",
    }


def test_persisted_approval_survives_failed_run_trim_without_shift() -> None:
    """Trimmed incomplete same-name calls should not shift approvals onto retained calls."""
    trimmed_call = Content.from_function_call(call_id="trimmed_call", name="zsh", arguments={"command": "ls"})
    retained_call = Content.from_function_call(call_id="retained_call", name="zsh", arguments={"command": "rm file"})
    assistant = Message("assistant", [trimmed_call, retained_call])
    result = Message(
        "tool",
        [Content.from_function_result(call_id="retained_call", result="Error: Tool execution was rejected")],
    )
    history = SessionHistoryManager()
    history.bind({"messages": [Message("user", ["run both"]), assistant, result]})

    history.persist_approval_decisions(
        [
            {"request_id": "", "tool_name": "zsh", "status": "auto_approved"},
            {"request_id": "req-1", "tool_name": "zsh", "status": "user_rejected"},
        ]
    )
    history.trim_to_last_complete_tool_results()

    assert [c.call_id for c in assistant.contents if c.type == "function_call"] == ["retained_call"]
    assert retained_call.additional_properties["_approval"] == {
        "tool_name": "zsh",
        "status": "user_rejected",
        "request_id": "req-1",
    }
    assert assistant.additional_properties["_approval"] == retained_call.additional_properties["_approval"]


def test_interrupt_repair_preserves_informational_call_without_local_result() -> None:
    hosted_call = Content.from_function_call(
        call_id="hosted_call",
        name="web_search",
        arguments={"query": "chrys"},
        informational_only=True,
    )
    assistant = Message("assistant", [hosted_call])
    history = SessionHistoryManager()
    history.bind({"messages": [Message("user", ["search"]), assistant]})

    history.trim_to_last_complete_tool_results()

    assert len(history.messages) == 2
    assert history.messages[0].text == "search"
    assert history.messages[1] is assistant
    assert assistant.contents == [hosted_call]


def test_interrupt_repair_removes_unmatched_local_call_but_keeps_mixed_informational_call() -> None:
    local_call = Content.from_function_call("local", "echo", arguments={})
    hosted_call = Content.from_function_call("hosted", "web_search", arguments={}, informational_only=True)
    assistant = Message("assistant", [Content.from_text("working"), local_call, hosted_call])
    history = SessionHistoryManager()
    history.bind({"messages": [Message("user", ["go"]), assistant]})

    history.trim_to_last_complete_tool_results()

    assert history.messages[-1] is assistant
    assert assistant.contents == [Content.from_text("working"), hosted_call]


def test_interrupt_repair_scans_past_text_only_assistant_sibling() -> None:
    dangling_call = Content.from_function_call("local", "echo", arguments={})
    call_message = Message("assistant", [dangling_call], message_id="assistant-1")
    text_message = Message("assistant", [Content.from_text("working")], message_id="assistant-2")
    user_message = Message("user", ["go"])
    history = SessionHistoryManager()
    history.bind({"messages": [user_message, call_message, text_message]})

    history.trim_to_last_complete_tool_results()

    assert history.messages == [user_message, text_message]


def test_persist_approval_decisions_start_index_ignores_prior_same_name_calls() -> None:
    """Current-run approval decisions should not attach to older same-name calls."""
    old_call = Content.from_function_call(call_id="old_call", name="zsh", arguments={"command": "wc -l old.py"})
    old_assistant = Message("assistant", [old_call])
    old_result = Message("tool", [Content.from_function_result(call_id="old_call", result="10 old.py")])
    new_call = Content.from_function_call(call_id="new_call", name="zsh", arguments={"command": "rm file"})
    new_assistant = Message("assistant", [new_call])
    messages = [
        Message("user", ["old turn"]),
        old_assistant,
        old_result,
        Message("user", ["new turn"]),
        new_assistant,
    ]
    history = SessionHistoryManager()
    history.bind({"messages": messages})

    history.persist_approval_decisions(
        [{"request_id": "req-1", "tool_name": "zsh", "status": "user_rejected"}],
        start_index=3,
    )

    assert "_approval" not in old_assistant.additional_properties
    assert "_approval" not in old_call.additional_properties
    assert new_call.additional_properties["_approval"] == {
        "tool_name": "zsh",
        "status": "user_rejected",
        "request_id": "req-1",
    }
    assert new_assistant.additional_properties["_approval"] == new_call.additional_properties["_approval"]


def test_persist_approval_decisions_uses_turn_marker_when_compaction_shifts_index() -> None:
    """A stale post-compaction index should still find current-turn approvals."""
    old_call = Content.from_function_call(call_id="old_call", name="zsh", arguments={"command": "wc -l old.py"})
    old_assistant = Message("assistant", [old_call])
    turn_marker = Message("assistant", [""])
    turn_marker.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
    new_call = Content.from_function_call(call_id="new_call", name="zsh", arguments={"command": "rm file"})
    new_assistant = Message("assistant", [new_call])
    messages = [
        Message("user", ["old turn"]),
        old_assistant,
        Message("tool", [Content.from_function_result(call_id="old_call", result="10 old.py")]),
        turn_marker,
        Message("user", ["new turn"]),
        new_assistant,
    ]
    history = SessionHistoryManager()
    history.bind({"messages": messages})

    history.persist_approval_decisions(
        [{"request_id": "req-1", "tool_name": "zsh", "status": "user_rejected"}],
        start_index=99,
    )

    assert "_approval" not in old_assistant.additional_properties
    assert "_approval" not in old_call.additional_properties
    assert new_call.additional_properties["_approval"] == {
        "tool_name": "zsh",
        "status": "user_rejected",
        "request_id": "req-1",
    }


def test_persisted_tool_call_context_reader() -> None:
    from chrys.service.session.message_metadata import (
        TOOL_CALL_CONTEXT_METADATA_KEY,
        persisted_tool_call_context,
    )

    assert persisted_tool_call_context(None) == {}
    assert persisted_tool_call_context({"other": 1}) == {}
    assert persisted_tool_call_context({TOOL_CALL_CONTEXT_METADATA_KEY: {"skill_name": "pdf"}}) == {"skill_name": "pdf"}
    assert persisted_tool_call_context({TOOL_CALL_CONTEXT_METADATA_KEY: "corrupt"}) == {}


# --- exchange-shape pins ------------------------------------------------


def _stamped_call(call_id: object, order: int | None = None, name: str = "zsh") -> Content:
    content = Content.from_function_call(call_id, name, arguments={})
    if order is not None:
        content.additional_properties[TOOL_INVOCATION_ORDER_KEY] = order
    return content


def _failure_result(call_id: object, error_kind: str = "argument_parsing") -> Content:
    content = Content.from_function_result(call_id, result=f"error_{call_id!r}")
    content.additional_properties[TOOL_ERROR_KIND_METADATA_KEY] = error_kind
    return content


def _order_decision(order: int, request_id: str = "req-1") -> dict[str, str]:
    return {
        "request_id": request_id,
        "tool_name": "zsh",
        "status": "user_approved",
        "tool_order": str(order),
    }


def _persist(messages: list[Message], decisions: list[dict[str, str]]) -> None:
    history = SessionHistoryManager()
    history.bind({"messages": messages})
    history.persist_approval_decisions(decisions)


def test_persist_approval_decisions_shared_block_siblings_each_matched() -> None:
    """Order decisions reach calls in every assistant sibling of a shared block."""
    first = _stamped_call("c1", 0)
    second = _stamped_call("c2", 1)
    _persist(
        [
            Message("user", ["run"]),
            Message("assistant", [first]),
            Message("assistant", [second]),
            Message("tool", [Content.from_function_result(call_id="c2", result="done")]),
        ],
        [_order_decision(0), _order_decision(1, "req-2")],
    )
    assert first.additional_properties["_approval"]["status"] == "user_approved"
    assert second.additional_properties["_approval"]["status"] == "user_approved"


def test_persist_approval_decisions_falsy_nonstring_failure_excludes_empty_id_call() -> None:
    """A falsy non-string failure id feeds the block-wide id-less exclusion."""
    empty_id_call = _stamped_call("", 0)
    _persist(
        [
            Message("user", ["run"]),
            Message("assistant", [empty_id_call]),
            Message("tool", [_failure_result(0)]),
        ],
        [_order_decision(0)],
    )
    assert "_approval" not in empty_id_call.additional_properties


def test_persist_approval_decisions_user_role_failure_does_not_exclude() -> None:
    """A user-role message carrying a validation failure is not part of the block."""
    call = _stamped_call("c1", 0)
    _persist(
        [
            Message("user", ["run"]),
            Message("assistant", [call]),
            Message("user", [_failure_result("c1")]),
        ],
        [_order_decision(0)],
    )
    assert call.additional_properties["_approval"]["status"] == "user_approved"


def test_persist_approval_decisions_marker_failure_does_not_exclude() -> None:
    """A history-marker message never carries the block's validation failures."""
    call = _stamped_call("c1", 0)
    marker = Message("assistant", [_failure_result("c1")])
    marker.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
    _persist(
        [Message("user", ["run"]), Message("assistant", [call]), marker],
        [_order_decision(0)],
    )
    assert call.additional_properties["_approval"]["status"] == "user_approved"


def test_persist_approval_decisions_embedded_failure_excludes_call() -> None:
    """A validation failure embedded in the call's own message proves the call
    never reached approval."""
    call = _stamped_call("c1", 0)
    _persist(
        [
            Message("user", ["run"]),
            Message("assistant", [call, _failure_result("c1")]),
        ],
        [_order_decision(0)],
    )
    assert "_approval" not in call.additional_properties


def test_persist_approval_decisions_digit_string_failure_excludes_numeric_call() -> None:
    """Malformed ids pair by string form: a "7" failure excludes the int-7 call."""
    call = _stamped_call(7, 0)
    _persist(
        [
            Message("user", ["run"]),
            Message("assistant", [call]),
            Message("tool", [_failure_result("7")]),
        ],
        [_order_decision(0)],
    )
    assert "_approval" not in call.additional_properties


def test_persist_approval_decisions_nan_failure_excludes_nan_call() -> None:
    """Two distinct NaN id objects pair through their identical string forms."""
    call = _stamped_call(float("nan"), 0)
    _persist(
        [
            Message("user", ["run"]),
            Message("assistant", [call]),
            Message("tool", [_failure_result(float("nan"))]),
        ],
        [_order_decision(0)],
    )
    assert "_approval" not in call.additional_properties


def test_persist_approval_decisions_bool_failure_does_not_exclude_int_call() -> None:
    """Python-equal ids of different types split by string form: True is not 1."""
    call = _stamped_call(1, 0)
    _persist(
        [
            Message("user", ["run"]),
            Message("assistant", [call]),
            Message("tool", [_failure_result(True)]),
        ],
        [_order_decision(0)],
    )
    assert call.additional_properties["_approval"]["status"] == "user_approved"


def test_persist_approval_decisions_unhashable_id_processed_without_crash() -> None:
    """A truthy unhashable call id is excluded from id matching but neither
    crashes persistence nor blocks other calls' decisions."""
    unhashable = Content.from_function_call(["x"], "zsh", arguments={})
    stamped = _stamped_call("c2", 0)
    _persist(
        [
            Message("user", ["run"]),
            Message("assistant", [unhashable]),
            Message("assistant", [stamped]),
            Message("tool", [Content.from_function_result(call_id="c2", result="done")]),
        ],
        [_order_decision(0)],
    )
    assert stamped.additional_properties["_approval"]["status"] == "user_approved"
    assert "_approval" not in unhashable.additional_properties


def test_persist_approval_decisions_name_fallback_reaches_unhashable_id_call() -> None:
    """An unmatched id-less decision still attaches by tool name to a call whose
    id is unhashable."""
    unhashable = Content.from_function_call(["x"], "zsh", arguments={})
    _persist(
        [Message("user", ["run"]), Message("assistant", [unhashable])],
        [{"request_id": "req-1", "tool_name": "zsh", "status": "user_rejected"}],
    )
    assert unhashable.additional_properties["_approval"]["status"] == "user_rejected"


def test_persist_approval_decisions_falsy_unhashable_call_stays_ordinal_eligible() -> None:
    """A falsy unhashable call id joins the empty-id stream and keeps its
    ordinal-matched decision."""
    call = _stamped_call([], 0)
    _persist(
        [Message("user", ["run"]), Message("assistant", [call])],
        [_order_decision(0)],
    )
    assert call.additional_properties["_approval"]["status"] == "user_approved"

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A child's history after a failed pass: what it keeps, and what its retry sends."""

from __future__ import annotations

from chrys.foundation.models.history_markers import ANSWER_SEGMENT_KEY
from chrys.kernel import AgentSession, Content, LoopRecorder, Message
from chrys.orchestration.invoker.child_history import ChildHistory


def _history(*messages: Message) -> ChildHistory:
    session = AgentSession()
    session.state["chrys_history"] = {"messages": list(messages)}
    return ChildHistory(session, LoopRecorder())


def test_a_failed_pass_keeps_the_answer_it_showed_after_its_prompt() -> None:
    """A first pass's prompt is not in history yet when it fails: the kept answer brings it along."""
    prompt = Message("user", ["review"])
    history = _history()

    history.repair_after_failure([prompt], 0, (Content.from_text("Looking. "),))

    assert [(message.role, message.text) for message in history.messages()] == [
        ("user", "review"),
        ("assistant", "Looking. "),
    ]


def test_a_retry_asks_again_for_the_answer_a_failed_pass_left() -> None:
    history = _history(Message("user", ["review"]), Message("assistant", ["Looking. "]))
    seeded = Message("user", ["review"])

    assert history.retry_input(lambda: [seeded]) == ([seeded], ())
    assert history.messages() == []


def test_a_retry_after_tool_work_goes_on_without_the_answer_left_after_it() -> None:
    call = Message("assistant", [Content.from_function_call(call_id="c1", name="read_file", arguments="{}")])
    result = Message("tool", [Content.from_function_result(call_id="c1", result="ok")])
    history = _history(Message("user", ["review"]), call, result, Message("assistant", ["Found it"]))

    assert history.retry_input(lambda: [Message("user", ["unused"])]) == ([], ())
    assert [message.role for message in history.messages()] == ["user", "assistant", "tool"]


def test_a_retry_names_the_segments_of_the_answer_it_asks_for_again() -> None:
    call = Message("assistant", [Content.from_function_call(call_id="c1", name="read_file", arguments="{}")])
    result = Message("tool", [Content.from_function_result(call_id="c1", result="ok")])
    shown = Content.from_text("Found it", additional_properties={ANSWER_SEGMENT_KEY: "segment-1"})
    history = _history(Message("user", ["review"]), call, result, Message("assistant", [shown]))

    assert history.retry_input(lambda: [Message("user", ["unused"])]) == ([], ("segment-1",))

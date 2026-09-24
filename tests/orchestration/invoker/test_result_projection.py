# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A child's parent result is its final segment, as a main turn's answer is its last message."""

from __future__ import annotations

import pytest

from chrys.kernel import AgentResponse, Content, Message
from chrys.orchestration.engine.run.bindings import TurnBindings
from chrys.orchestration.sub_agents.kernel_policy import KernelSubAgentPolicy
from chrys.service.agent_middleware.events.hosted_tools import FinalSegment


def _narrated_call() -> Message:
    return Message("assistant", ["Let me look. ", Content.from_function_call("c1", "read_file", arguments={})])


def _call_result() -> Message:
    return Message("tool", [Content.from_function_result("c1", result="contents")])


def _hosted_image() -> Content:
    return Content.from_image_generation_tool_result(
        image_id="image-1",
        outputs=[Content.from_uri("data:image/png;base64,QUJD", media_type="image/png")],
        hosted_provider="openai",
        provider_phase="terminal",
        provider_status="completed",
    )


@pytest.mark.parametrize(
    ("messages", "main", "child"),
    [
        ([], "", FinalSegment("", "")),
        (
            [_narrated_call(), _call_result(), Message("assistant", ["Answer."])],
            "Answer.",
            FinalSegment("Answer.", "Answer."),
        ),
        # Nothing follows the last tool result, so the parent gets everything
        # the run said; all of it was intermediate, so the transcript has no
        # final segment.
        ([_narrated_call(), _call_result()], "", FinalSegment("Let me look. ", "")),
        ([Message("assistant", ["first"]), Message("assistant", [])], "", FinalSegment("first", "")),
        (
            [Message("assistant", [Content.from_shell_tool_result(call_id="sh1", outputs=[])])],
            "",
            FinalSegment("", ""),
        ),
    ],
    ids=["empty", "answer-after-tool-loop", "ends-on-tool-result", "empty-last-message", "structured-without-artifact"],
)
def test_child_result_is_its_final_segment(messages: list[Message], main: str, child: FinalSegment) -> None:
    response = AgentResponse(messages=messages)
    assert TurnBindings._extract_final_text(response) == main
    assert KernelSubAgentPolicy._final_segment(response) == child


@pytest.mark.parametrize(
    ("contents", "child"),
    [
        (["Drawing. ", _hosted_image(), "Here it is."], FinalSegment("Here it is.", "Here it is.")),
        (["Drawing. ", _hosted_image()], FinalSegment("Drawing. ", "")),
        # A textless hosted result gets a neutral result the transcript shows too.
        ([_hosted_image()], FinalSegment("Sub-agent returned image output.", "Sub-agent returned image output.")),
    ],
    ids=["trailing-text", "no-trailing-text", "no-text"],
)
def test_hosted_child_final_segment_follows_the_last_hosted_output(
    contents: list[str | Content], child: FinalSegment
) -> None:
    response = AgentResponse(messages=[Message("assistant", contents)])
    assert KernelSubAgentPolicy._final_segment(response) == child

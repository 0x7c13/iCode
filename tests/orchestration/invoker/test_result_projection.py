# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Main and child text projection intentionally have different scopes."""

from __future__ import annotations

import pytest

from chrys.kernel import AgentResponse, Content, Message
from chrys.orchestration.engine.run.bindings import TurnBindings
from chrys.orchestration.sub_agents.kernel_policy import KernelSubAgentPolicy


@pytest.mark.parametrize(
    ("messages", "main", "child"),
    [
        ([], "", ""),
        ([Message("assistant", ["first"]), Message("assistant", ["last"])], "last", "firstlast"),
        ([Message("assistant", ["first"]), Message("assistant", [])], "", "first"),
        (
            [Message("assistant", [Content.from_shell_tool_result(call_id="sh1", outputs=[])])],
            "",
            "",
        ),
    ],
    ids=["empty", "multiple-messages", "empty-last-message", "structured-without-artifact"],
)
def test_main_last_message_and_child_whole_response(messages: list[Message], main: str, child: str) -> None:
    response = AgentResponse(messages=messages)
    assert TurnBindings._extract_final_text(response) == main
    assert KernelSubAgentPolicy._extract_text(response) == child

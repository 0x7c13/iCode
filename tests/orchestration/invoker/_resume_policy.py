# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Drive real resume policies through a signature-checked backend request port.

Shared by input-strategy and persisted-timestamp tests. Only execution is
mocked; policy identity, outcome, evidence, anchoring and metadata are real.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import create_autospec

from chrys.foundation.trajectory.metadata import ANALYTICS_ITEM_ID_KEY
from chrys.kernel import AgentSession, Message
from chrys.orchestration.engine.run.resume import TurnPassState, TurnResumePolicy
from chrys.orchestration.invoker.kernel import KernelConversation


@dataclass
class ResumeHarness:
    backend: KernelConversation
    state: TurnPassState
    inputs: TurnResumePolicy


def make_resume_harness(msgs: list[Message], execute=None):
    """Exercise a real policy against an autospecced backend request boundary."""
    backend = create_autospec(KernelConversation, instance=True)
    backend.session = AgentSession()
    backend.session.state = {"chrys_history": {"messages": msgs}}
    backend.run.side_effect = execute
    backend.validate.return_value = None
    backend.continuation_is_live.return_value = False
    state = TurnPassState()
    return ResumeHarness(backend, state, TurnResumePolicy(backend, "session", state))


def _assert_item_ids(policy: TurnResumePolicy, messages) -> None:
    assert isinstance(policy._opening_item_id, str | None)
    for message in messages:
        assert isinstance(message.additional_properties.get(ANALYTICS_ITEM_ID_KEY), str | None)


async def drive_retry_policy(executor, additional_text="", created_at=None):
    async with executor.inputs.retry_request(additional_text, created_at) as request:
        if request is not None:
            _assert_item_ids(executor.inputs, request.messages)
            await executor.backend.run(request)
    _assert_item_ids(executor.inputs, executor.backend.session.state["chrys_history"]["messages"])


async def drive_fresh_policy(executor, contents, created_at=None):
    executor.inputs.begin_invocation()
    request = executor.inputs.fresh_request(contents, created_at)
    _assert_item_ids(executor.inputs, request.messages)
    await executor.backend.run(request)

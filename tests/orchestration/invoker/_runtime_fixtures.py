# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Explicit resource/operation contracts for existing shell-only test doubles."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

from chrys.foundation.models.invocations import PassHandle
from chrys.kernel import Agent, Content, Message
from chrys.orchestration.invoker.contracts import Ok, RunIntent, RunRequest, StopCause, UsageDelta
from chrys.orchestration.invoker.evidence import ZERO_COUNT, PassEvidence
from chrys.orchestration.invoker.resources import Conversation
from chrys.orchestration.invoker.runtime import KernelRuntime
from chrys.service.agent_middleware.system_reminder import SystemReminderMiddleware
from chrys.service.context.compaction.last_words_state import LastWordsState
from chrys.service.context.manager import ContextManager


class SubAgentPolicyDouble:
    """A real common shell over a scripted backend; each test supplies execute()."""

    failure_reason = None
    final_segment = None

    def __init__(self, *, shell, prompt):
        self.shell = shell
        self.prompt = prompt

    @property
    def backend(self):
        return self

    @property
    def active_handle(self):
        return None

    def request(self, ticket):
        assert ticket is None
        return RunRequest([Message("user", [self.prompt])], RunIntent.FRESH, self.shell.origin)

    async def run(self, request):
        text = await self.execute()
        identity = request.origin.invocation_id
        return Ok(
            handle=PassHandle(identity, identity + ":1"),
            usage=UsageDelta(),
            effects=PassEvidence(identity, identity + ":1", ZERO_COUNT, ZERO_COUNT, ZERO_COUNT, False),
            stop=StopCause.COMPLETED,
            continuation=None,
            segments=(Content.from_text(text),),
        )

    async def execute(self):
        raise AssertionError("The test must supply its scripted execution")

    async def project_result(self, outcome):
        return "".join(item.text or "" for item in outcome.segments)

    def check_terminal_race(self):
        pass

    def observe_continuation_token(self, token):
        pass

    async def finish_run(self):
        pass

    async def run_cancelled(self):
        pass

    def latch_abort(self, cause):
        pass

    async def cancel_active(self):
        pass

    async def finalize_cancellation(self):
        pass


def runtime_for_shell(
    agent: Agent, reminder: SystemReminderMiddleware, context: ContextManager | None = None
) -> Callable[[Conversation, str], KernelRuntime]:
    """Keep a shell test's explicitly supplied runtime rather than a production fallback."""

    def create(owner: Conversation, invocation_id: str) -> KernelRuntime:
        # Shells never read the Phase 4 state; the spec'd stand-in only fills the field.
        return KernelRuntime(owner, agent, context, reminder, MagicMock(spec=LastWordsState))  # type: ignore[arg-type]

    return create

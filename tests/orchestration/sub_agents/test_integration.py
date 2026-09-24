# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""End-to-end integration tests for sub-agent pause / retry / abort.

These tests drive the full pipeline through :class:`MockChatClient` for
both the parent and the sub-agent, exercising the complete chain:

    UserMessage → Engine → TurnBindings → main agent.run()
                → MockChatClient (parent) emits tool_call for sub-agent
                → SubAgentTools._invoke → SubAgentToolShell.run()
                → sub-agent.run() → MockChatClient (sub) raises / succeeds
                → retry loop / pause / resume / abort
                → parent continues with tool_result

The controller-only tests in ``test_sub_agent_controller.py`` use a
hand-rolled stub agent; they cover the controller's state machine in
isolation. These tests exercise AgentEngine and SubAgentToolShell together —
FSM edges, marker insertion under real publish timing,
parent-history repair, event wiring. That's what this file is for.

Retry budgets and backoffs are aggressively shrunk (``max_retries=1``,
zero delays) so tests converge in milliseconds. That's handled by
monkey-patching :class:`KernelSubAgentPolicy`'s constructor defaults for
the duration of the test.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

import chrys.orchestration.engine.build.builder as builder_module
import chrys.orchestration.sub_agents.tools as sub_agent_module
from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    ApprovalRequest,
    ApprovalResponse,
    Event,
    InvocationAborted,
    InvocationAbortRequested,
    InvocationCascadeAborted,
    InvocationMessage,
    InvocationPaused,
    InvocationProgress,
    InvocationResumed,
    InvocationRetryAttempt,
    InvocationRetryRequested,
    InvocationToolCallResult,
    InvocationToolCallStart,
    SessionReady,
    UsageUpdate,
    UserInterrupt,
    UserMessage,
)
from chrys.foundation.hosted_tools import HostedToolFamily
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.models.workspace import Workspace
from chrys.foundation.tool_kinds import KIND_SKILL
from chrys.foundation.util.sub_agent_context import SUB_AGENT_TRANSCRIPT_FINAL_TEXT_METADATA_KEY
from chrys.orchestration.engine.engine import AgentEngine
from chrys.orchestration.engine.state.machine import EngineState
from chrys.orchestration.sub_agents.kernel_policy import KernelSubAgentPolicy
from chrys.service.llm.mock import MockChatClient, MockResponse
from chrys.service.profiles.agents.registry import AgentProfileRegistry
from chrys.service.profiles.agents.schema import (
    AgentProfile,
    ApprovalConfig,
    CompactionConfig,
    ModelConfig,
    SkillConfig,
    SkillsConfig,
    SubAgentRef,
    SubAgentsConfig,
    ToolsConfig,
)
from chrys.service.profiles.models.registry import ModelProfileRegistry
from chrys.service.profiles.models.schema import ModelProfile
from chrys.service.skills.constants import RUN_SKILL_SCRIPT_TOOL_NAME
from chrys.service.state.store import JsonFileStateStore
from tests.support.pipeline_helpers import fail_nested_before_response_on_nth
from tests.support.scripted_clients import ErrorMockChatClient, FrameworkBoom, HostedMockResponse, hosted_image_result
from tests.support.waiting import ENGINE_TEST_WAIT_TIMEOUT, await_run_task_chain

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Error-capable mock client
# ---------------------------------------------------------------------------


# A retryable SDK-style exception.  The name matches
# ``errors.RETRYABLE_TYPE_NAMES`` so it hits the auto-retry path
# without pausing.
class APIConnectionError(Exception):
    pass


# ---------------------------------------------------------------------------
# Fixture: fast controller (shrink retry budget + zero backoff)
# ---------------------------------------------------------------------------


@pytest.fixture
def fast_sub_agent_controller(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make :class:`KernelSubAgentPolicy` use minimal retries + no backoff.

    ``SubAgentTools._make_tool`` constructs the controller without
    passing ``max_retries`` / ``backoff_schedule``, so defaults apply.
    We substitute a subclass whose defaults are test-friendly.
    """

    class _FastController(KernelSubAgentPolicy):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("max_retries", 1)
            kwargs.setdefault("backoff_schedule", (0,))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(sub_agent_module, "KernelSubAgentPolicy", _FastController)


# ---------------------------------------------------------------------------
# Engine factory — parent + sub-agent both backed by ErrorMockChatClient
# ---------------------------------------------------------------------------


class _SubAgentPipelineCtx:
    """Holds everything a sub-agent integration test needs."""

    def __init__(
        self,
        *,
        engine: AgentEngine,
        bus: EventBus,
        events: list[Event],
        main_client: ErrorMockChatClient,
        sub_client: ErrorMockChatClient,
        store: JsonFileStateStore,
        session_id: str,
        create_client_patcher: pytest.MonkeyPatch,
    ) -> None:
        self.engine = engine
        self.bus = bus
        self.events = events
        self.main_client = main_client
        self.sub_client = sub_client
        self.store = store
        self.session_id = session_id
        self._create_client_patcher = create_client_patcher

    async def cleanup(self) -> None:
        self._create_client_patcher.undo()

    # Ceilings sized for contended CI workers (a loaded Windows runner has
    # pushed the mock-client pipeline past 20s) while staying under the
    # global 60s per-test timeout, so the diagnostic TimeoutError below
    # still fires with the observed-event list instead of a thread kill.
    async def wait_for_event(self, event_type: type, *, timeout: float = 45.0, min_count: int = 1) -> list[Event]:
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            matches = [e for e in self.events if isinstance(e, event_type)]
            if len(matches) >= min_count:
                return matches
            await asyncio.sleep(0.02)
        found = [type(e).__name__ for e in self.events]
        raise TimeoutError(f"Timed out waiting for {event_type.__name__} (saw {found})")

    async def wait_for_idle(self, *, timeout: float = ENGINE_TEST_WAIT_TIMEOUT) -> None:
        """Wait for the complete parent run-task chain, including its save."""
        await await_run_task_chain(
            self.engine, turn_state=self.engine.turns.turn_state, timeout=timeout, expect_installed=True
        )


async def _make_ctx(
    tmp_path: Path,
    *,
    main_outcomes: list[MockResponse | BaseException],
    sub_outcomes: list[MockResponse | BaseException],
    sub_stream: bool = False,
    sub_builtins: list[str] | None = None,
    search_respect_gitignore: bool = True,
    sub_skills: SkillsConfig | None = None,
    sub_responses_store: bool = False,
    agent_engine,
) -> _SubAgentPipelineCtx:
    """Build a parent+sub-agent engine with independent mock clients.

    The parent invokes the sub-agent tool ``Explore`` (arg ``prompt``).
    Clients are routed by ``ModelProfile.id``: the parent's active
    profile is ``"main-mock-profile"`` and the sub-agent pins
    ``"sub-mock-profile"`` via its ``ModelConfig.profile_id``. So a
    single ``create_client`` patch can return the right client to each
    caller.
    """
    events: list[Event] = []

    async def _collect(ev: Event) -> None:
        events.append(ev)

    bus = EventBus()
    for cls in (
        SessionReady,
        InvocationMessage,
        ApprovalRequest,
        ApprovalResponse,
        InvocationToolCallStart,
        InvocationToolCallResult,
        InvocationPaused,
        InvocationResumed,
        InvocationAborted,
        InvocationCascadeAborted,
        InvocationRetryAttempt,
        InvocationProgress,
        UsageUpdate,
    ):
        await bus.subscribe(cls, _collect)

    # Two model profiles so create_client routing is trivial.
    model_registry = ModelProfileRegistry()
    main_profile = ModelProfile(
        id="main-mock-profile",
        name="main",
        provider="mock",
        model_id="mock",
        stream=False,
        max_context_tokens=100_000,
    )
    sub_profile = ModelProfile(
        id="sub-mock-profile",
        name="sub",
        provider="openai" if sub_responses_store else "mock",
        api_style="responses" if sub_responses_store else "chat_completions",
        model_id="gpt-sub" if sub_responses_store else "mock",
        stream=sub_stream,
        max_context_tokens=100_000,
        chat_options='{"store": true}' if sub_responses_store else "",
    )
    model_registry.register(main_profile)
    model_registry.register(sub_profile)

    # These scenarios exercise parent/child turns, not advisory disk scans.
    settings = Settings(
        model_profile="main-mock-profile",
        workspace_change_notice=False,
        search_respect_gitignore=search_respect_gitignore,
    )
    store = JsonFileStateStore(tmp_path / "sessions")

    parent_profile = AgentProfile(
        name="Parent",
        instructions="You are the parent. Use the Explore sub-agent.",
        tools=ToolsConfig(builtins=[]),
        approval=ApprovalConfig(default="auto"),
        compaction=CompactionConfig(enabled=False),
        sub_agents=SubAgentsConfig(
            max_total_concurrency=2,
            agents=[SubAgentRef(profile="Explore", tool_name="Explore")],
        ),
    )
    sub_agent_profile = AgentProfile(
        name="Explore",
        instructions="You explore. Return a summary.",
        tools=ToolsConfig(builtins=sub_builtins or []),
        skills=sub_skills or SkillsConfig(auto_load_user_agents_skills=False, auto_load_cwd_agents_skills=False),
        approval=ApprovalConfig(default="auto"),
        compaction=CompactionConfig(enabled=False),
        # Pin sub-agent to its own model profile so create_client routes correctly.
        model=ModelConfig(profile_id="sub-mock-profile"),
    )
    agent_registry = AgentProfileRegistry()
    agent_registry.register(parent_profile)
    agent_registry.register(sub_agent_profile)

    main_client = ErrorMockChatClient(outcomes=main_outcomes)
    sub_client = ErrorMockChatClient(outcomes=sub_outcomes)

    def _patched_create_client(p: Any = None, **kw: Any) -> MockChatClient:
        # Route by resolved ModelProfile id.  Both clients get the
        # intermediate-text callbacks installed so streaming callbacks
        # behave identically to the real path (main is non-streaming in
        # these tests, but this keeps the helper robust if a future test
        # flips streaming on).
        profile_id = getattr(p, "id", "") or ""
        client = sub_client if profile_id == "sub-mock-profile" else main_client
        client._on_intermediate_text_async = kw.get("on_intermediate_text_async")
        client._on_intermediate_text_sync = kw.get("on_intermediate_text_sync")
        return client

    create_client_patcher = pytest.MonkeyPatch()
    create_client_patcher.setattr(builder_module, "create_client", _patched_create_client)

    # Also patch the sub_agent module's create_client reference so
    # SubAgentTools.register (which imports create_client at the module
    # top) picks up the test-scoped routing, not the real one.
    sub_create_client_patcher = pytest.MonkeyPatch()
    sub_create_client_patcher.setattr(sub_agent_module, "create_client", _patched_create_client)

    engine = agent_engine(
        bus,
        settings=settings,
        agent_registry=agent_registry,
        model_registry=model_registry,
        state_store=store,
        initial_workspace=Workspace.from_cwd(str(tmp_path)),
    )
    try:
        await engine.start(parent_profile)
    except Exception:
        create_client_patcher.undo()
        sub_create_client_patcher.undo()
        raise

    # sub_agent_module import was already consumed during start; restore
    # it here so other tests aren't affected.  The parent builder_module
    # patch stays live until cleanup (it's needed if the engine rebuilds).
    sub_create_client_patcher.undo()

    return _SubAgentPipelineCtx(
        engine=engine,
        bus=bus,
        events=events,
        main_client=main_client,
        sub_client=sub_client,
        store=store,
        session_id=engine.session.session_id or "",
        create_client_patcher=create_client_patcher,
    )


def _sub_tool_call(prompt: str, call_id: str = "call-1") -> MockResponse:
    return MockResponse(tool_calls=[("Explore", call_id, {"prompt": prompt})])


# ---------------------------------------------------------------------------
# Scenario tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("respect", [True, False])
async def test_sub_agent_search_inherits_parent_setting(tmp_path: Path, git_repo_factory, agent_engine, respect: bool):
    root = git_repo_factory(tmp_path / "repo")
    (root / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (root / "ignored.py").write_text("NEEDLE\n", encoding="utf-8")
    (root / "visible.py").write_text("NEEDLE\n", encoding="utf-8")
    ctx = await _make_ctx(
        root,
        main_outcomes=[_sub_tool_call("Search files"), MockResponse(text="Done")],
        sub_outcomes=[
            MockResponse(tool_calls=[("grep", "search-call", {"pattern": "NEEDLE", "glob": "*.py"})]),
            MockResponse(text="Searched"),
        ],
        sub_builtins=["search"],
        search_respect_gitignore=respect,
        agent_engine=agent_engine,
    )
    try:
        await ctx.bus.publish(UserMessage(text="Search"))
        await ctx.wait_for_idle()
        results = [
            event for event in ctx.events if isinstance(event, InvocationToolCallResult) and event.tool_name == "grep"
        ]
        assert len(results) == 1
        assert "visible.py" in str(results[0].result)
        assert ("ignored.py" in str(results[0].result)) is not respect
    finally:
        await ctx.cleanup()


@pytest.mark.asyncio
async def test_sub_agent_gates_runtime_skill_script_under_parent_auto_default(
    tmp_path: Path,
    fast_sub_agent_controller: None,
    agent_engine,
) -> None:
    """Sub-agent provider tools retain KIND_SKILL through the real kernel loop."""
    sub_skills = SkillsConfig(
        inline=[SkillConfig(name="review", description="Review files", instructions="Review carefully.")],
        auto_load_user_agents_skills=False,
        auto_load_cwd_agents_skills=False,
    )
    ctx = await _make_ctx(
        tmp_path,
        main_outcomes=[_sub_tool_call("Run the review skill"), MockResponse(text="Parent finished")],
        sub_outcomes=[
            MockResponse(
                tool_calls=[
                    (
                        RUN_SKILL_SCRIPT_TOOL_NAME,
                        "sub-skill-call",
                        {"skill_name": "review", "script_name": "scripts/review.py"},
                    )
                ]
            ),
            MockResponse(text="Sub-agent handled rejection"),
        ],
        sub_skills=sub_skills,
        agent_engine=agent_engine,
    )
    try:
        await ctx.bus.publish(UserMessage(text="Delegate the review"))

        approval_events = await ctx.wait_for_event(ApprovalRequest)
        request = approval_events[0]
        assert isinstance(request, ApprovalRequest)
        assert request.tool_name == RUN_SKILL_SCRIPT_TOOL_NAME
        assert request.tool_kind == KIND_SKILL

        await ctx.bus.publish(
            ApprovalResponse(request_id=request.request_id, approved=False, reason="untrusted sub-agent skill")
        )
        await ctx.wait_for_idle()

        assert ctx.sub_client.call_count == 2
        assert ctx.main_client.call_count == 2
    finally:
        await ctx.cleanup()


@pytest.mark.asyncio
async def test_auto_retry_succeeds_without_pausing(
    tmp_path: Path, fast_sub_agent_controller: None, agent_engine
) -> None:
    """Retryable SDK error → controller auto-retries inside StreamRetryLoop
    and succeeds on the second attempt; no pause, no user input required.
    The parent continues to a final text response."""
    ctx = await _make_ctx(
        tmp_path,
        main_outcomes=[
            _sub_tool_call("explore please"),
            MockResponse(text="Final summary after sub-agent."),
        ],
        sub_outcomes=[
            APIConnectionError("peer closed connection"),
            MockResponse(text="sub-agent result"),
        ],
        agent_engine=agent_engine,
    )
    try:
        await ctx.bus.publish(UserMessage(text="go"))
        await ctx.wait_for_idle()

        # No pause should have fired.
        paused = [e for e in ctx.events if (isinstance(e, InvocationPaused) and e.origin.kind == "sub_agent")]
        assert paused == []

        # At least one InvocationRetryAttempt published for the transient error.
        retries = [e for e in ctx.events if (isinstance(e, InvocationRetryAttempt) and e.origin.kind == "sub_agent")]
        assert len(retries) >= 1
        assert retries[0].origin.invocation_id  # non-empty

        # Parent received the sub-agent's result as a ToolCallResult and
        # produced a final assistant message.
        finals = [
            e for e in ctx.events if (isinstance(e, InvocationMessage) and e.origin.kind == "turn") and e.is_final
        ]
        assert any("Final summary" in f.text for f in finals)

        # Sub-agent was called twice (error + success); parent twice
        # (tool_call dispatch + final summary).
        assert ctx.sub_client.call_count == 2
        assert ctx.main_client.call_count == 2

        # Engine ends IDLE, no lingering paused_sub_agents.
        assert ctx.engine.state is EngineState.IDLE
        assert ctx.engine.turns.turn_state.paused_sub_agents == set()
    finally:
        await ctx.cleanup()


@pytest.mark.asyncio
async def test_sub_agent_uses_model_profile_stream_setting(
    tmp_path: Path, fast_sub_agent_controller: None, agent_engine
) -> None:
    """A sub-agent pinned to a streaming model profile should call its LLM with ``stream=True``."""
    ctx = await _make_ctx(
        tmp_path,
        main_outcomes=[
            _sub_tool_call("explore with streaming"),
            MockResponse(text="Final summary after sub-agent."),
        ],
        sub_outcomes=[
            MockResponse(text="streamed sub-agent result"),
        ],
        sub_stream=True,
        agent_engine=agent_engine,
    )
    try:
        await ctx.bus.publish(UserMessage(text="go"))
        await ctx.wait_for_idle()

        assert ctx.sub_client.stream_flags == [True]
        assert ctx.main_client.stream_flags == [False, False]
        tool_results = [
            e
            for e in ctx.events
            if (isinstance(e, InvocationToolCallResult) and e.origin.kind == "turn") and e.tool_name == "Explore"
        ]
        assert tool_results and "streamed sub-agent result" in tool_results[-1].result
    finally:
        await ctx.cleanup()


@pytest.mark.asyncio
async def test_sub_agent_usage_updates_flow_before_parent_tool_finishes(
    tmp_path: Path,
    fast_sub_agent_controller: None,
    agent_engine,
) -> None:
    """Sub-agent usage publishes live UsageUpdate events and cumulative card progress."""
    ctx = await _make_ctx(
        tmp_path,
        main_outcomes=[
            _sub_tool_call("explore with live usage"),
            MockResponse(text="Parent saw sub-agent usage."),
        ],
        sub_outcomes=[
            MockResponse(
                tool_calls=[("sleep", "sleep-1", {"seconds": 1, "reason": "hold the sub-agent open"})],
                usage_details={"input_token_count": 10, "output_token_count": 2, "total_token_count": 12},
            ),
            MockResponse(
                tool_calls=[("sleep", "sleep-2", {"seconds": 0, "reason": "continue"})],
                usage_details={"input_token_count": 30, "output_token_count": 4, "total_token_count": 34},
            ),
            MockResponse(
                text="sub-agent result",
                usage_details={"input_token_count": 8, "output_token_count": 3, "total_token_count": 11},
            ),
        ],
        sub_builtins=["sleep"],
        agent_engine=agent_engine,
    )
    try:
        await ctx.bus.publish(UserMessage(text="go"))

        first_usage = (await ctx.wait_for_event(UsageUpdate))[0]
        assert isinstance(first_usage, UsageUpdate)
        assert first_usage.agent_profile == "Explore"
        assert first_usage.usage_source_id
        assert first_usage.usage_source_id != ctx.session_id
        assert first_usage.total_tokens == 12
        assert first_usage.total_session_tokens == 12
        assert not [
            e
            for e in ctx.events
            if (isinstance(e, InvocationToolCallResult) and e.origin.kind == "turn") and e.tool_name == "Explore"
        ]
        assert not [
            e for e in ctx.events if (isinstance(e, InvocationMessage) and e.origin.kind == "turn") and e.is_final
        ]

        await ctx.wait_for_idle()

        usage_updates = [e for e in ctx.events if isinstance(e, UsageUpdate)]
        assert any(
            e.agent_profile == "Explore"
            and e.usage_source_id == first_usage.usage_source_id
            and e.total_tokens == 34
            and e.total_session_tokens == 46
            for e in usage_updates
        )
        assert any(
            e.agent_profile == "Explore"
            and e.usage_source_id == first_usage.usage_source_id
            and e.total_tokens == 11
            and e.total_session_tokens == 57
            for e in usage_updates
        )

        progress = [e for e in ctx.events if (isinstance(e, InvocationProgress) and e.origin.kind == "sub_agent")]
        assert any(e.total_tokens == 12 and e.total_usage_tokens == 12 for e in progress)
        assert any(e.total_tokens == 34 and e.total_usage_tokens == 46 for e in progress)
        assert any(e.total_tokens == 11 and e.total_usage_tokens == 57 for e in progress)
        final_progress_index = next(
            i
            for i, e in enumerate(ctx.events)
            if (isinstance(e, InvocationProgress) and e.origin.kind == "sub_agent") and e.total_usage_tokens == 57
        )
        parent_result_index = next(
            i
            for i, e in enumerate(ctx.events)
            if (isinstance(e, InvocationToolCallResult) and e.origin.kind == "turn") and e.tool_name == "Explore"
        )
        assert final_progress_index < parent_result_index
    finally:
        await ctx.cleanup()


async def _narrated_sub_agent_ctx(tmp_path: Path, agent_engine) -> _SubAgentPipelineCtx:
    """A sub-agent that narrates one tool call, then answers."""
    return await _make_ctx(
        tmp_path,
        main_outcomes=[_sub_tool_call("explore"), MockResponse(text="Parent done.")],
        sub_outcomes=[
            MockResponse(
                text="Let me check the tree. ",
                tool_calls=[("sleep", "sleep-1", {"seconds": 0, "reason": "look around"})],
            ),
            MockResponse(text="The answer is 42."),
        ],
        sub_builtins=["sleep"],
        agent_engine=agent_engine,
    )


def _explore_result(ctx: _SubAgentPipelineCtx) -> InvocationToolCallResult:
    (tool_result,) = [
        e
        for e in ctx.events
        if isinstance(e, InvocationToolCallResult) and e.origin.kind == "turn" and e.tool_name == "Explore"
    ]
    return tool_result


@pytest.mark.asyncio
async def test_parent_receives_the_final_segment_not_the_narration(
    tmp_path: Path, fast_sub_agent_controller: None, agent_engine
) -> None:
    """Text written between the sub-agent's tool calls stays in its transcript."""
    ctx = await _narrated_sub_agent_ctx(tmp_path, agent_engine)
    try:
        await ctx.bus.publish(UserMessage(text="go"))
        await ctx.wait_for_idle()

        tool_result = _explore_result(ctx)
        assert tool_result.result == "The answer is 42."
        # The result is the final segment, so the transcript needs no declaration.
        assert SUB_AGENT_TRANSCRIPT_FINAL_TEXT_METADATA_KEY not in tool_result.metadata
        sent = [
            str(content.result)
            for messages, _options in ctx.main_client.call_history
            for message in messages
            for content in message.contents
            if content.type == "function_result" and content.call_id == "call-1"
        ]
        assert sent == ["The answer is 42."]
        assert [
            e.text
            for e in ctx.events
            if isinstance(e, InvocationMessage) and e.origin.kind == "sub_agent" and e.is_intermediate
        ] == ["Let me check the tree. "]
    finally:
        await ctx.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call",
    [("no_such_tool", "c-1", {}), ("sleep", "sleep-1", {})],
    ids=["unknown-tool", "invalid-arguments"],
)
async def test_streamed_narration_of_a_call_outside_the_tool_pipeline_reaches_the_transcript(
    tmp_path: Path, fast_sub_agent_controller: None, agent_engine, call: tuple[str, str, dict[str, Any]]
) -> None:
    """The kernel answers these calls without a tool start, which is what releases streamed text."""
    ctx = await _make_ctx(
        tmp_path,
        main_outcomes=[_sub_tool_call("explore"), MockResponse(text="Parent done.")],
        sub_outcomes=[
            MockResponse(text="Let me check the tree. ", tool_calls=[call]),
            MockResponse(text="The answer is 42."),
        ],
        sub_stream=True,
        sub_builtins=["sleep"],
        agent_engine=agent_engine,
    )
    try:
        await ctx.bus.publish(UserMessage(text="go"))
        await ctx.wait_for_idle()

        assert _explore_result(ctx).result == "The answer is 42."
        narration = [
            i
            for i, e in enumerate(ctx.events)
            if isinstance(e, InvocationMessage)
            and e.origin.kind == "sub_agent"
            and e.is_intermediate
            and e.text == "Let me check the tree. "
        ]
        assert len(narration) == 1
        assert narration[0] < ctx.events.index(_explore_result(ctx))
    finally:
        await ctx.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("sub_stream", [True, False], ids=["streamed", "blocking"])
@pytest.mark.parametrize(
    "call",
    [("sleep", "sleep-1", {"seconds": 0}), ("no_such_tool", "c-1", {})],
    ids=["tool-start", "unknown-tool"],
)
async def test_narration_before_hosted_output_reaches_the_transcript_once(
    tmp_path: Path,
    fast_sub_agent_controller: None,
    agent_engine,
    call: tuple[str, str, dict[str, Any]],
    sub_stream: bool,
) -> None:
    """The narration precedes the next response's hosted output, and the run-end reconcile leaves it alone."""
    ctx = await _make_ctx(
        tmp_path,
        main_outcomes=[_sub_tool_call("explore"), MockResponse(text="Parent done.")],
        sub_outcomes=[
            MockResponse(text="Let me check the tree. ", tool_calls=[call]),
            HostedMockResponse(text="Here it is.", hosted=[hosted_image_result()]),
        ],
        sub_stream=sub_stream,
        sub_builtins=["sleep"],
        agent_engine=agent_engine,
    )
    try:
        await ctx.bus.publish(UserMessage(text="go"))
        await ctx.wait_for_idle()

        assert _explore_result(ctx).result == "Here it is."
        child_results = [
            (e.tool_name, e.provider_hosted)
            for e in ctx.events
            if isinstance(e, InvocationToolCallResult) and e.origin.kind == "sub_agent"
        ]
        assert (HostedToolFamily.IMAGE, True) in child_results
        assert (call[0] == "sleep") == (("sleep", False) in child_results)
        narration = [
            i
            for i, e in enumerate(ctx.events)
            if isinstance(e, InvocationMessage) and e.origin.kind == "sub_agent" and e.is_intermediate
        ]
        assert [ctx.events[i].text for i in narration] == ["Let me check the tree. "]
        hosted_start_at = next(
            i
            for i, e in enumerate(ctx.events)
            if isinstance(e, InvocationToolCallStart) and e.origin.kind == "sub_agent" and e.provider_hosted
        )
        assert narration[0] < hosted_start_at
    finally:
        await ctx.cleanup()


@pytest.mark.asyncio
async def test_narration_of_a_failed_pass_precedes_its_pause(
    tmp_path: Path, fast_sub_agent_controller: None, agent_engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Streamed text of a call that started no tool belongs to the pass that wrote it, not to the retry."""
    ctx = await _make_ctx(
        tmp_path,
        main_outcomes=[_sub_tool_call("explore"), MockResponse(text="Parent done.")],
        sub_outcomes=[
            MockResponse(text="Let me check the tree. ", tool_calls=[("no_such_tool", "c-1", {})]),
            MockResponse(text="The answer is 42."),
        ],
        sub_stream=True,
        agent_engine=agent_engine,
    )
    # No next response starts, so only the end of the failed pass can release the text.
    fail_nested_before_response_on_nth(monkeypatch, 2, main=ctx.engine.current.loaded.bindings._response_validation)

    def narration() -> list[int]:
        return [
            i
            for i, e in enumerate(ctx.events)
            if isinstance(e, InvocationMessage) and e.origin.kind == "sub_agent" and e.is_intermediate
        ]

    try:
        await ctx.bus.publish(UserMessage(text="go"))
        (paused,) = await ctx.wait_for_event(InvocationPaused)
        assert len(narration()) == 1
        assert narration()[0] < ctx.events.index(paused)

        await ctx.bus.publish(InvocationRetryRequested(invocation_id=paused.origin.invocation_id))
        await ctx.wait_for_event(InvocationResumed)
        await ctx.wait_for_idle()

        assert _explore_result(ctx).result == "The answer is 42."
        assert [ctx.events[i].text for i in narration()] == ["Let me check the tree. "]
    finally:
        await ctx.cleanup()


@pytest.mark.asyncio
async def test_declared_final_segment_reaches_the_parent_result_metadata(
    tmp_path: Path, fast_sub_agent_controller: None, agent_engine
) -> None:
    """A run that ends on hosted output returns everything it said and tells the transcript so."""
    ctx = await _make_ctx(
        tmp_path,
        main_outcomes=[_sub_tool_call("explore"), MockResponse(text="Parent done.")],
        sub_outcomes=[
            MockResponse(
                text="Let me check the tree. ",
                tool_calls=[("sleep", "sleep-1", {"seconds": 0, "reason": "look around"})],
            ),
            HostedMockResponse(text="", hosted=[hosted_image_result()]),
        ],
        sub_builtins=["sleep"],
        agent_engine=agent_engine,
    )
    try:
        await ctx.bus.publish(UserMessage(text="go"))
        await ctx.wait_for_idle()

        assert any(
            isinstance(e, InvocationToolCallResult) and e.origin.kind == "sub_agent" and e.provider_hosted
            for e in ctx.events
        )
        tool_result = _explore_result(ctx)
        assert tool_result.result == "Let me check the tree. "
        assert tool_result.metadata[SUB_AGENT_TRANSCRIPT_FINAL_TEXT_METADATA_KEY] == ""
    finally:
        await ctx.cleanup()


@pytest.mark.asyncio
async def test_pause_then_user_retry_recovers(
    tmp_path: Path, fast_sub_agent_controller: None, agent_engine, *, engine_services
) -> None:
    """Non-retryable error exhausts auto-retry budget → InvocationPaused.
    User publishes InvocationRetryRequested → sub-agent is re-run and succeeds;
    the parent receives the successful tool_result and completes normally."""
    ctx = await _make_ctx(
        tmp_path,
        main_outcomes=[
            _sub_tool_call("try hard"),
            MockResponse(text="All good — recovered."),
        ],
        # FrameworkBoom isn't retryable — ``StreamRetryLoop`` re-raises
        # on the first occurrence without consuming the retry budget.
        # So one boom pauses, then the user's retry triggers a fresh
        # ``agent.run()`` that succeeds.
        sub_outcomes=[
            FrameworkBoom("initial boom"),
            MockResponse(text="recovered result"),
        ],
        agent_engine=agent_engine,
    )
    try:
        await ctx.bus.publish(UserMessage(text="go"))
        paused = await ctx.wait_for_event(InvocationPaused)
        assert paused[0].reason == "framework_exc"
        assert "initial boom" in paused[0].last_error
        invocation_id = paused[0].origin.invocation_id

        # FSM must be AWAITING_SUB_AGENTS while paused.
        assert ctx.engine.state is EngineState.AWAITING_SUB_AGENTS
        assert invocation_id in ctx.engine.turns.turn_state.paused_sub_agents

        # User clicks Retry on the paused card.
        await ctx.bus.publish(InvocationRetryRequested(invocation_id=invocation_id))
        await ctx.wait_for_event(InvocationResumed)
        await ctx.wait_for_idle()

        # Sub-agent called 2x (1 boom + 1 success); parent 2x.
        assert ctx.sub_client.call_count == 2
        assert ctx.main_client.call_count == 2

        finals = [
            e for e in ctx.events if (isinstance(e, InvocationMessage) and e.origin.kind == "turn") and e.is_final
        ]
        assert any("recovered" in f.text for f in finals)

        # Tool result delivered to parent carries the success text.
        tool_results = [
            e
            for e in ctx.events
            if (isinstance(e, InvocationToolCallResult) and e.origin.kind == "turn") and e.tool_name == "Explore"
        ]
        assert tool_results and "recovered result" in tool_results[-1].result

        # FSM back to IDLE, paused set empty, no awaiting marker.
        assert ctx.engine.state is EngineState.IDLE
        assert ctx.engine.turns.turn_state.paused_sub_agents == set()
        kinds = [
            m.additional_properties.get(HistoryMarkerKind.KEY) for m in engine_services(ctx.engine).history.messages
        ]
        assert HistoryMarkerKind.AWAITING_SUB_AGENTS not in kinds
    finally:
        await ctx.cleanup()


@pytest.mark.asyncio
async def test_pause_retry_continues_after_completed_sub_agent_tool(
    tmp_path: Path,
    fast_sub_agent_controller: None,
    agent_engine,
) -> None:
    """A paused sub-agent retry should continue after completed inner tool work."""
    ctx = await _make_ctx(
        tmp_path,
        main_outcomes=[
            _sub_tool_call("try with tool"),
            MockResponse(text="Parent saw resumed sub-agent."),
        ],
        sub_outcomes=[
            MockResponse(tool_calls=[("sleep", "sleep-1", {"seconds": 0, "reason": "integration test"})]),
            FrameworkBoom("failed after sleep"),
            MockResponse(text="continued after sleep"),
        ],
        sub_builtins=["sleep"],
        agent_engine=agent_engine,
    )
    try:
        await ctx.bus.publish(UserMessage(text="go"))
        paused = await ctx.wait_for_event(InvocationPaused)
        assert "failed after sleep" in paused[0].last_error

        await ctx.bus.publish(InvocationRetryRequested(invocation_id=paused[0].origin.invocation_id))
        await ctx.wait_for_event(InvocationResumed)
        await ctx.wait_for_idle()

        assert ctx.sub_client.call_count == 3
        retry_messages = ctx.sub_client.call_history[2][0]
        user_texts = [m.text for m in retry_messages if m.role == "user"]
        assert len(user_texts) == 1
        assert user_texts[0].startswith("try with tool")
        assert any(
            content.type == "function_result" and content.call_id == "sleep-1"
            for message in retry_messages
            for content in message.contents
        )

        tool_results = [
            e
            for e in ctx.events
            if (isinstance(e, InvocationToolCallResult) and e.origin.kind == "turn") and e.tool_name == "Explore"
        ]
        assert tool_results and "continued after sleep" in tool_results[-1].result
    finally:
        await ctx.cleanup()


@pytest.mark.asyncio
async def test_repeated_pause_retry_preserves_sub_agent_tool_order(
    tmp_path: Path,
    fast_sub_agent_controller: None,
    agent_engine,
) -> None:
    """Repeated sub-agent retries should append new tool work after earlier completed work."""
    ctx = await _make_ctx(
        tmp_path,
        main_outcomes=[
            _sub_tool_call("try multiple tools"),
            MockResponse(text="Parent saw repeated resumed sub-agent."),
        ],
        sub_outcomes=[
            MockResponse(tool_calls=[("sleep", "sleep-1", {"seconds": 0, "reason": "first"})]),
            FrameworkBoom("failed after first sleep"),
            MockResponse(tool_calls=[("sleep", "sleep-2", {"seconds": 0, "reason": "second"})]),
            FrameworkBoom("failed after second sleep"),
            MockResponse(text="continued after both sleeps"),
        ],
        sub_builtins=["sleep"],
        agent_engine=agent_engine,
    )
    try:
        await ctx.bus.publish(UserMessage(text="go"))
        first_pause = (await ctx.wait_for_event(InvocationPaused))[0]
        assert "failed after first sleep" in first_pause.last_error

        await ctx.bus.publish(InvocationRetryRequested(invocation_id=first_pause.origin.invocation_id))
        second_pause = (await ctx.wait_for_event(InvocationPaused, min_count=2))[1]
        assert "failed after second sleep" in second_pause.last_error

        await ctx.bus.publish(InvocationRetryRequested(invocation_id=second_pause.origin.invocation_id))
        await ctx.wait_for_event(InvocationResumed, min_count=2)
        await ctx.wait_for_idle()

        assert ctx.sub_client.call_count == 5
        final_messages = ctx.sub_client.call_history[4][0]
        user_texts = [m.text for m in final_messages if m.role == "user"]
        assert len(user_texts) == 1
        assert user_texts[0].startswith("try multiple tools")

        result_call_ids = [
            content.call_id
            for message in final_messages
            for content in message.contents
            if content.type == "function_result"
        ]
        assert result_call_ids == ["sleep-1", "sleep-2"]

        tool_results = [
            e
            for e in ctx.events
            if (isinstance(e, InvocationToolCallResult) and e.origin.kind == "turn") and e.tool_name == "Explore"
        ]
        assert tool_results and "continued after both sleeps" in tool_results[-1].result
    finally:
        await ctx.cleanup()


@pytest.mark.asyncio
async def test_store_true_pause_retry_replays_local_history_without_stale_conversation(
    tmp_path: Path,
    fast_sub_agent_controller: None,
    agent_engine,
) -> None:
    """Responses ``store: true`` retries must replay repaired local history, not stale provider state."""
    ctx = await _make_ctx(
        tmp_path,
        main_outcomes=[
            _sub_tool_call("try with stored tool"),
            MockResponse(text="Parent saw stored resumed sub-agent."),
        ],
        sub_outcomes=[
            MockResponse(
                tool_calls=[("sleep", "sleep-store-1", {"seconds": 0, "reason": "stored first"})],
                conversation_id="resp_partial_sub_1",
            ),
            FrameworkBoom("failed after stored sleep"),
            MockResponse(text="continued after stored sleep", conversation_id="resp_recovered_sub_1"),
        ],
        sub_builtins=["sleep"],
        sub_responses_store=True,
        agent_engine=agent_engine,
    )
    try:
        await ctx.bus.publish(UserMessage(text="go"))
        paused = await ctx.wait_for_event(InvocationPaused)
        assert "failed after stored sleep" in paused[0].last_error

        # The failed in-flight tool-loop call had a provider continuation id.
        assert ctx.sub_client.call_history[0][1]["store"] is True
        assert "conversation_id" not in ctx.sub_client.call_history[0][1]
        assert ctx.sub_client.call_history[1][1]["conversation_id"] == "resp_partial_sub_1"

        await ctx.bus.publish(InvocationRetryRequested(invocation_id=paused[0].origin.invocation_id))
        await ctx.wait_for_event(InvocationResumed)
        await ctx.wait_for_idle()

        assert ctx.sub_client.call_count == 3
        retry_messages, retry_options = ctx.sub_client.call_history[2]
        assert retry_options["store"] is True
        assert "conversation_id" not in retry_options

        user_texts = [m.text for m in retry_messages if m.role == "user"]
        assert len(user_texts) == 1
        assert user_texts[0].startswith("try with stored tool")
        assert any(
            content.type == "function_result" and content.call_id == "sleep-store-1"
            for message in retry_messages
            for content in message.contents
        )

        tool_results = [
            e
            for e in ctx.events
            if (isinstance(e, InvocationToolCallResult) and e.origin.kind == "turn") and e.tool_name == "Explore"
        ]
        assert tool_results and "continued after stored sleep" in tool_results[-1].result
    finally:
        await ctx.cleanup()


@pytest.mark.asyncio
async def test_streaming_store_true_pause_retry_replays_completed_tool_work(
    tmp_path: Path,
    fast_sub_agent_controller: None,
    agent_engine,
) -> None:
    """Streamed Responses ``store: true`` retries must preserve completed inner tool work."""
    ctx = await _make_ctx(
        tmp_path,
        main_outcomes=[
            _sub_tool_call("try streamed stored tool"),
            MockResponse(text="Parent saw streamed stored resumed sub-agent."),
        ],
        sub_outcomes=[
            MockResponse(
                tool_calls=[("sleep", "sleep-stream-store-1", {"seconds": 0, "reason": "streamed stored"})],
                conversation_id="resp_stream_partial_sub_1",
                chunk_delay=0,
            ),
            FrameworkBoom("failed after streamed stored sleep"),
            MockResponse(
                text="continued after streamed stored sleep",
                conversation_id="resp_stream_recovered_sub_1",
                chunk_delay=0,
            ),
        ],
        sub_stream=True,
        sub_builtins=["sleep"],
        sub_responses_store=True,
        agent_engine=agent_engine,
    )
    try:
        await ctx.bus.publish(UserMessage(text="go"))
        paused = await ctx.wait_for_event(InvocationPaused)
        assert "failed after streamed stored sleep" in paused[0].last_error

        assert ctx.sub_client.stream_flags == [True, True]
        assert ctx.sub_client.call_history[1][1]["conversation_id"] == "resp_stream_partial_sub_1"

        await ctx.bus.publish(InvocationRetryRequested(invocation_id=paused[0].origin.invocation_id))
        await ctx.wait_for_event(InvocationResumed)
        await ctx.wait_for_idle()

        assert ctx.sub_client.stream_flags == [True, True, True]
        retry_messages, retry_options = ctx.sub_client.call_history[2]
        assert retry_options["store"] is True
        assert "conversation_id" not in retry_options

        user_texts = [m.text for m in retry_messages if m.role == "user"]
        assert len(user_texts) == 1
        assert user_texts[0].startswith("try streamed stored tool")
        assert any(
            content.type == "function_result" and content.call_id == "sleep-stream-store-1"
            for message in retry_messages
            for content in message.contents
        )

        tool_results = [
            e
            for e in ctx.events
            if (isinstance(e, InvocationToolCallResult) and e.origin.kind == "turn") and e.tool_name == "Explore"
        ]
        assert tool_results and "continued after streamed stored sleep" in tool_results[-1].result
    finally:
        await ctx.cleanup()


@pytest.mark.asyncio
async def test_store_true_repeated_pause_retry_clears_each_stale_conversation(
    tmp_path: Path,
    fast_sub_agent_controller: None,
    agent_engine,
) -> None:
    """Each pause after a Responses stored tool-loop failure must clear the newest provider id."""
    ctx = await _make_ctx(
        tmp_path,
        main_outcomes=[
            _sub_tool_call("try multiple stored tools"),
            MockResponse(text="Parent saw repeated stored resumed sub-agent."),
        ],
        sub_outcomes=[
            MockResponse(
                tool_calls=[("sleep", "sleep-store-1", {"seconds": 0, "reason": "first stored"})],
                conversation_id="resp_partial_sub_1",
            ),
            FrameworkBoom("failed after first stored sleep"),
            MockResponse(
                tool_calls=[("sleep", "sleep-store-2", {"seconds": 0, "reason": "second stored"})],
                conversation_id="resp_partial_sub_2",
            ),
            FrameworkBoom("failed after second stored sleep"),
            MockResponse(text="continued after both stored sleeps", conversation_id="resp_recovered_sub_2"),
        ],
        sub_builtins=["sleep"],
        sub_responses_store=True,
        agent_engine=agent_engine,
    )
    try:
        await ctx.bus.publish(UserMessage(text="go"))
        first_pause = (await ctx.wait_for_event(InvocationPaused))[0]
        assert "failed after first stored sleep" in first_pause.last_error
        assert ctx.sub_client.call_history[1][1]["conversation_id"] == "resp_partial_sub_1"

        await ctx.bus.publish(InvocationRetryRequested(invocation_id=first_pause.origin.invocation_id))
        second_pause = (await ctx.wait_for_event(InvocationPaused, min_count=2))[1]
        assert "failed after second stored sleep" in second_pause.last_error
        assert "conversation_id" not in ctx.sub_client.call_history[2][1]
        assert ctx.sub_client.call_history[3][1]["conversation_id"] == "resp_partial_sub_2"

        await ctx.bus.publish(InvocationRetryRequested(invocation_id=second_pause.origin.invocation_id))
        await ctx.wait_for_event(InvocationResumed, min_count=2)
        await ctx.wait_for_idle()

        assert ctx.sub_client.call_count == 5
        final_messages, final_options = ctx.sub_client.call_history[4]
        assert final_options["store"] is True
        assert "conversation_id" not in final_options

        user_texts = [m.text for m in final_messages if m.role == "user"]
        assert len(user_texts) == 1
        assert user_texts[0].startswith("try multiple stored tools")
        result_call_ids = [
            content.call_id
            for message in final_messages
            for content in message.contents
            if content.type == "function_result"
        ]
        assert result_call_ids == ["sleep-store-1", "sleep-store-2"]

        tool_results = [
            e
            for e in ctx.events
            if (isinstance(e, InvocationToolCallResult) and e.origin.kind == "turn") and e.tool_name == "Explore"
        ]
        assert tool_results and "continued after both stored sleeps" in tool_results[-1].result
    finally:
        await ctx.cleanup()


@pytest.mark.asyncio
async def test_pause_then_user_abort_returns_error_to_parent(
    tmp_path: Path, fast_sub_agent_controller: None, agent_engine
) -> None:
    """User abort on a paused sub-agent: the tool call returns an
    ``Error:`` string to the parent, the parent produces a final
    response acknowledging the failure, and the run completes IDLE
    with no dangling paused state."""
    ctx = await _make_ctx(
        tmp_path,
        main_outcomes=[
            _sub_tool_call("explore"),
            MockResponse(text="Acknowledged failure."),
        ],
        sub_outcomes=[
            FrameworkBoom("cannot connect to service"),
            FrameworkBoom("cannot connect to service"),
        ],
        agent_engine=agent_engine,
    )
    try:
        await ctx.bus.publish(UserMessage(text="go"))
        paused = await ctx.wait_for_event(InvocationPaused)
        invocation_id = paused[0].origin.invocation_id

        # User aborts.
        await ctx.bus.publish(InvocationAbortRequested(invocation_id=invocation_id))
        aborted = await ctx.wait_for_event(InvocationAborted)
        assert aborted[0].origin.invocation_id == invocation_id
        await ctx.wait_for_idle()

        # Parent tool result is an Error: string carrying the failure.
        tool_results = [
            e
            for e in ctx.events
            if (isinstance(e, InvocationToolCallResult) and e.origin.kind == "turn") and e.tool_name == "Explore"
        ]
        assert tool_results, "Parent never received a tool_result for the sub-agent"
        result_text = tool_results[-1].result
        assert result_text.startswith("Error: sub-agent 'Explore' aborted by user")
        assert "cannot connect to service" in result_text

        # Parent proceeded to its final response.
        finals = [
            e for e in ctx.events if (isinstance(e, InvocationMessage) and e.origin.kind == "turn") and e.is_final
        ]
        assert any("Acknowledged" in f.text for f in finals)

        # Engine terminal state clean.
        assert ctx.engine.state is EngineState.IDLE
        assert ctx.engine.turns.turn_state.paused_sub_agents == set()
    finally:
        await ctx.cleanup()


@pytest.mark.asyncio
async def test_cascade_abort_via_user_interrupt(
    tmp_path: Path, fast_sub_agent_controller: None, agent_engine, *, engine_services
) -> None:
    """Global interrupt while a sub-agent is paused: the engine cascades,
    the sub-agent's controller emits InvocationCascadeAborted, and the
    parent run transitions to INTERRUPTED with an interrupted marker."""
    ctx = await _make_ctx(
        tmp_path,
        main_outcomes=[
            _sub_tool_call("run forever"),
            # The parent should NOT get a chance to produce this — the
            # interrupt unwinds before the main agent runs again.
            MockResponse(text="Should not appear."),
        ],
        sub_outcomes=[
            FrameworkBoom("bad auth"),
            FrameworkBoom("bad auth"),
        ],
        agent_engine=agent_engine,
    )
    try:
        await ctx.bus.publish(UserMessage(text="go"))
        paused = await ctx.wait_for_event(InvocationPaused)
        invocation_id = paused[0].origin.invocation_id
        assert ctx.engine.state is EngineState.AWAITING_SUB_AGENTS

        # Global UserInterrupt — cascades to every live sub-agent.
        await ctx.bus.publish(UserInterrupt())
        cascades = await ctx.wait_for_event(InvocationCascadeAborted)
        assert any(ev.origin.invocation_id == invocation_id for ev in cascades)
        await ctx.wait_for_idle()

        # Parent never produced the final text; it was interrupted.
        finals = [
            e for e in ctx.events if (isinstance(e, InvocationMessage) and e.origin.kind == "turn") and e.is_final
        ]
        assert not any("Should not appear" in f.text for f in finals)

        # Interrupted marker in history; no awaiting marker left over.
        kinds = [
            m.additional_properties.get(HistoryMarkerKind.KEY) for m in engine_services(ctx.engine).history.messages
        ]
        assert HistoryMarkerKind.INTERRUPTED in kinds
        assert HistoryMarkerKind.AWAITING_SUB_AGENTS not in kinds

        # FSM in INTERRUPTED, paused set drained.
        assert ctx.engine.state is EngineState.INTERRUPTED
        assert ctx.engine.turns.turn_state.paused_sub_agents == set()
    finally:
        await ctx.cleanup()

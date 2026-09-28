# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Nested Chrys ACP frontend smoke test."""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
import yaml
from textual.app import App, ComposeResult

import chrys.orchestration.engine.build.builder as builder_module
from chrys.app.tui.theme import TuiVariableDefaultsMixin
from chrys.app.tui.widgets.chat.renderers.sub_agent import SubAgentToolCall
from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    ApprovalRequest,
    ApprovalResponse,
    AskUserResponse,
    Event,
    InvocationMessage,
    InvocationPaused,
    InvocationProgress,
    InvocationStarted,
    InvocationToolCallResult,
    InvocationToolCallStart,
    QuestionToUser,
    UserMessage,
)
from chrys.foundation.models.ask_user import AskUserAnswer
from chrys.orchestration.engine.engine import AgentEngine
from chrys.service.llm.mock import MockChatClient, MockResponse
from chrys.service.profiles.agents.registry import AgentProfileRegistry
from chrys.service.profiles.agents.schema import (
    AcpAgentConfig,
    AgentProfile,
    ApprovalConfig,
    CompactionConfig,
    SubAgentRef,
    SubAgentsConfig,
    ToolsConfig,
)
from chrys.service.profiles.models.registry import ModelProfileRegistry
from chrys.service.profiles.models.schema import ModelProfile
from chrys.service.session import sub_agent_logs
from chrys.service.state.store import JsonFileStateStore
from tests.support.platform_fakes import platform_with_config_dir
from tests.support.waiting import ENGINE_TEST_WAIT_TIMEOUT, wait_for, with_wait_deadline

_REPO_ROOT = Path(__file__).resolve().parents[3]


class _ScriptedOpenAiServer:
    """Small loopback Chat Completions server used by the nested process."""

    def __init__(self, *, write_path: Path) -> None:
        self._write_path = write_path
        self.requests: list[dict[str, Any]] = []
        self.main_requests: list[dict[str, Any]] = []
        self.errors: list[BaseException] = []
        self._server: asyncio.Server | None = None

    @property
    def base_url(self) -> str:
        server = self._server
        assert server is not None
        socket = server.sockets[0]
        return f"http://127.0.0.1:{socket.getsockname()[1]}/v1"

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)

    async def close(self) -> None:
        server = self._server
        if server is None:
            return
        server.close()
        await server.wait_closed()
        self._server = None

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            lines = header.decode("ascii").split("\r\n")
            request_line = lines[0].split()
            assert request_line[0] == "POST"
            assert request_line[1].endswith("/chat/completions")
            headers = {
                key.strip().casefold(): value.strip()
                for line in lines[1:]
                if ":" in line
                for key, value in [line.split(":", 1)]
            }
            body = await reader.readexactly(int(headers["content-length"]))
            request = json.loads(body)
            self.requests.append(request)
            if request.get("tools"):
                self.main_requests.append(request)
                payload = self._response_payload(request, len(self.main_requests))
            else:
                payload = self._side_response_payload(len(self.requests))
            encoded = json.dumps(payload, separators=(",", ":")).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(encoded)}\r\n".encode()
                + b"Connection: close\r\n\r\n"
                + encoded
            )
            await writer.drain()
        except BaseException as exc:
            self.errors.append(exc)
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()

    def _response_payload(self, request: dict[str, Any], index: int) -> dict[str, Any]:
        content: str | None = None
        tool_calls: list[dict[str, Any]] | None = None
        finish_reason = "tool_calls"
        tools = request.get("tools", [])
        names = [tool.get("function", {}).get("name", "") for tool in tools]
        if index == 1:
            write_name = next(
                (
                    tool.get("function", {}).get("name", "")
                    for tool in tools
                    if {"path", "content"}
                    <= tool.get("function", {}).get("parameters", {}).get("properties", {}).keys()
                ),
                "",
            )
            assert write_name
            tool_calls = [
                {
                    "id": "nested-write",
                    "type": "function",
                    "function": {
                        "name": write_name,
                        "arguments": json.dumps(
                            {
                                "path": str(self._write_path),
                                "content": "nested permission bridge",
                            }
                        ),
                    },
                }
            ]
        elif index == 2:
            assert "ask_user" in names
            tool_calls = [
                {
                    "id": "nested-question",
                    "type": "function",
                    "function": {
                        "name": "ask_user",
                        "arguments": json.dumps(
                            {
                                "questions": [
                                    {
                                        "question": "Which color should nested Chrys use?",
                                        "header": "Color",
                                        "options": [
                                            {"label": "Blue", "description": "Use the blue theme."},
                                            {"label": "Green", "description": "Use the green theme."},
                                        ],
                                    },
                                    {
                                        "question": "Which interfaces should be covered?",
                                        "header": "Interfaces",
                                        "options": [{"label": "TUI"}, {"label": "ACP"}],
                                        "multi_select": True,
                                    },
                                    {
                                        "question": "What rollout note should be recorded?",
                                        "header": "Rollout",
                                    },
                                ],
                            }
                        ),
                    },
                }
            ]
        else:
            content = "child result from nested Chrys"
            finish_reason = "stop"

        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls is not None:
            message["tool_calls"] = tool_calls
        usage_total = (7, 11, 13)[min(index, 3) - 1]
        return {
            "id": f"chatcmpl-nested-{index}",
            "object": "chat.completion",
            "created": 1,
            "model": "nested-mock",
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": usage_total - 2,
                "completion_tokens": 2,
                "total_tokens": usage_total,
            },
        }

    @staticmethod
    def _side_response_payload(index: int) -> dict[str, Any]:
        return {
            "id": f"chatcmpl-side-{index}",
            "object": "chat.completion",
            "created": 1,
            "model": "nested-mock",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Nested Chrys session"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }


def _write_nested_profiles(config_dir: Path, *, base_url: str) -> None:
    models_dir = config_dir / "models"
    agents_dir = config_dir / "agents"
    models_dir.mkdir(parents=True)
    agents_dir.mkdir(parents=True)
    (models_dir / "nested-mock.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "nested-mock",
                "name": "Nested Mock",
                "provider": "openai",
                "api_style": "chat_completions",
                "model_id": "nested-mock",
                "max_context_tokens": 100_000,
                "max_output_tokens": 1_000,
                "base_url": base_url,
                "api_key": "test-key",
                "stream": False,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (agents_dir / "RemoteSmoke.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "remote-smoke",
                "name": "RemoteSmoke",
                "display_name": "Remote Smoke",
                "description": "Nested ACP smoke agent",
                "instructions": "Use the requested tools, then return the final nested result.",
                "model": {"profile_id": "nested-mock"},
                "tools": {"builtins": ["filesystem.write", "ask_user"]},
                "approval": {
                    "default": "require",
                    "overrides": {"ask_user": "skip"},
                    "user_can_override": False,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


@pytest.mark.integration
@pytest.mark.asyncio
@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_chrys_acp_sub_agent_bridges_card_permission_question_result_and_usage_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    agent_engine: Any,
) -> None:
    chrys = shutil.which("chrys", path=str(Path(sys.executable).parent))
    assert chrys is not None
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_path = workspace / "nested-smoke.txt"
    mock_server = _ScriptedOpenAiServer(write_path=write_path)
    await mock_server.start()
    child_home = tmp_path / "child-home"
    child_config = child_home / ".chrys"
    child_appdata = tmp_path / "child-appdata"
    _write_nested_profiles(child_config, base_url=mock_server.base_url)
    _write_nested_profiles(child_appdata / "chrys", base_url=mock_server.base_url)

    bus = EventBus()
    captured: dict[type[Event], list[Event]] = defaultdict(list)

    async def collect(event: Event) -> None:
        captured[type(event)].append(event)

    for event_type in (
        InvocationMessage,
        ApprovalRequest,
        QuestionToUser,
        InvocationStarted,
        InvocationPaused,
        InvocationProgress,
        InvocationToolCallResult,
        InvocationToolCallStart,
    ):
        await bus.subscribe(event_type, collect)

    async def approve_nested_tool(event: ApprovalRequest) -> None:
        await bus.publish(
            ApprovalResponse(
                request_id=event.request_id,
                approved=True,
                reason="integration approval",
                session_id=event.session_id,
            )
        )

    async def answer_nested_question(event: QuestionToUser) -> None:
        assert len(event.questions) == 3
        await bus.publish(
            AskUserResponse(
                request_id=event.request_id,
                answers=(
                    AskUserAnswer(values=("Blue",)),
                    AskUserAnswer(values=("TUI", "ACP")),
                    AskUserAnswer(values=("Canary first",)),
                ),
                session_id=event.session_id,
            )
        )

    await bus.subscribe(ApprovalRequest, approve_nested_tool)
    await bus.subscribe(QuestionToUser, answer_nested_question)

    parent_client = MockChatClient(
        responses=[
            MockResponse(tool_calls=[("Explore", "delegate-1", {"prompt": "run the nested smoke"})]),
            MockResponse(text="parent finished"),
        ]
    )

    async def create_parent_client(_profile: Any = None, **kwargs: Any) -> MockChatClient:
        parent_client._on_intermediate_text_async = kwargs.get("on_intermediate_text_async")
        parent_client._on_intermediate_text_sync = kwargs.get("on_intermediate_text_sync")
        return parent_client

    monkeypatch.setattr(builder_module, "create_client", create_parent_client)
    monkeypatch.setattr(
        sub_agent_logs,
        "get_platform",
        lambda: platform_with_config_dir(tmp_path / "parent-config"),
    )

    model_registry = ModelProfileRegistry()
    model_registry.register(
        ModelProfile(
            id="parent-mock",
            name="Parent Mock",
            provider="mock",
            model_id="parent-mock",
            stream=False,
            max_context_tokens=100_000,
        )
    )
    parent = AgentProfile(
        name="Parent",
        tools=ToolsConfig(),
        approval=ApprovalConfig(default="auto"),
        compaction=CompactionConfig(enabled=False),
        sub_agents=SubAgentsConfig(
            agents=[SubAgentRef(profile="Nested Chrys", tool_name="Explore")],
        ),
    )
    child_env = {
        "HOME": str(child_home),
        "USERPROFILE": str(child_home),
        "APPDATA": str(child_appdata),
        "CHRYS_MODEL_PROFILE": "nested-mock",
        "CHRYS_SESSION_ROOT_DIR": str(tmp_path / "child-sessions"),
        "OPENAI_API_KEY": "test-key",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    }
    external = AgentProfile(
        name="Nested Chrys",
        display_name="Nested Chrys",
        sub_agent_only=True,
        acp=AcpAgentConfig(
            command=chrys,
            args=[
                "acp",
                "--agent",
                "RemoteSmoke",
                "--approval",
                "manual",
                "-C",
                str(workspace),
            ],
            env=child_env,
            cwd=str(_REPO_ROOT),
            allow_external_cwd=True,
            handshake_timeout_seconds=20,
            idle_timeout_seconds=30,
        ),
    )
    agent_registry = AgentProfileRegistry()
    agent_registry.register(parent)
    agent_registry.register(external)
    engine: AgentEngine = agent_engine(
        bus,
        settings=Settings(model_profile="parent-mock", default_approval_mode="manual"),
        agent_registry=agent_registry,
        model_registry=model_registry,
        state_store=JsonFileStateStore(tmp_path / "parent-sessions"),
    )

    class CardApp(TuiVariableDefaultsMixin, App[None]):
        def compose(self) -> ComposeResult:
            yield SubAgentToolCall("delegate-1", "Explore", args={"prompt": "run the nested smoke"})

    try:
        await engine.start(parent)
        async with CardApp().run_test() as pilot:
            card = pilot.app.query_one(SubAgentToolCall)

            async def show_inner_start(event: InvocationToolCallStart) -> None:
                if event.origin.kind != "sub_agent":
                    return
                await card.add_inner_tool_start(
                    event.call_id,
                    event.tool_name,
                    event.args,
                    tool_kind=event.tool_kind,
                )

            async def show_inner_result(event: InvocationToolCallResult) -> None:
                if event.origin.kind != "sub_agent":
                    return
                card.complete_inner_tool(
                    event.call_id,
                    event.result,
                    event.duration_ms,
                    event.image_contents,
                    metadata=event.metadata,
                )

            async def show_progress(event: InvocationProgress) -> None:
                if event.origin.kind != "sub_agent":
                    return
                card.update_progress(
                    event.tool_call_count,
                    event.total_tokens,
                    event.total_usage_tokens,
                    event.usage_unreported_attempts,
                )

            async def show_parent_result(event: InvocationToolCallResult) -> None:
                if event.origin.kind == "turn" and event.tool_name == "Explore":
                    card.set_complete(
                        event.result,
                        event.duration_ms,
                        metadata=event.metadata,
                    )

            await bus.subscribe(InvocationToolCallStart, show_inner_start)
            await bus.subscribe(InvocationToolCallResult, show_inner_result)
            await bus.subscribe(InvocationProgress, show_progress)
            await bus.subscribe(InvocationToolCallResult, show_parent_result)

            await bus.publish(UserMessage(text="Delegate to nested Chrys"))
            try:
                await wait_for(
                    lambda: parent_client.call_count == 2 or bool(captured[InvocationPaused]),
                    timeout=40,
                    description="nested ACP parent completion or pause",
                )
            except AssertionError as wait_error:
                controllers = (
                    list(engine.current.loaded.sub_agent_tools._controllers.values())
                    if engine.current.loaded.sub_agent_tools
                    else []
                )
                stderr = ""
                if controllers and controllers[0].last_spec is not None:
                    stderr_path = controllers[0].last_spec.stderr_log_path
                    if stderr_path.is_file():
                        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
                pytest.fail(
                    "nested ACP turn did not complete: "
                    f"parent_calls={parent_client.call_count}, "
                    f"http_requests={len(mock_server.requests)}, "
                    f"events={{{', '.join(f'{kind.__name__}: {len(events)}' for kind, events in captured.items())}}}, "
                    f"stderr={stderr}; wait={wait_error}"
                )
            pauses = [
                event
                for event in captured[InvocationPaused]
                if (isinstance(event, InvocationPaused) and event.origin.kind == "sub_agent")
            ]
            diagnostic = ""
            if pauses and pauses[-1].diagnostic_path:
                diagnostic = Path(pauses[-1].diagnostic_path).read_text(encoding="utf-8", errors="replace")
            assert not pauses, f"{pauses[-1].last_error}\n{diagnostic}"
            await engine.wait_for_run_task()
            await wait_for(lambda: card.status == "complete", description="completed nested sub-agent card")

            results = [
                event
                for event in captured[InvocationToolCallResult]
                if (isinstance(event, InvocationToolCallResult) and event.origin.kind == "turn")
                and event.tool_name == "Explore"
            ]
            assert len(results) == 1
            assert results[0].result == "child result from nested Chrys"
            assert card.result_text == results[0].result
            assert card._total_inner_calls == 2
            assert card._completed_inner_calls == 2
            assert card._progress_total_usage_tokens == 31

        approvals = [event for event in captured[ApprovalRequest] if isinstance(event, ApprovalRequest)]
        questions = [event for event in captured[QuestionToUser] if isinstance(event, QuestionToUser)]
        assert len(approvals) == 1
        assert approvals[0].tool_kind == ""
        assert approvals[0].tool_name.startswith("acp:")
        assert approvals[0].args["path"] == str(write_path)
        assert len(questions) == 1
        assert tuple(question.question for question in questions[0].questions) == (
            "Which color should nested Chrys use?",
            "Which interfaces should be covered?",
            "What rollout note should be recorded?",
        )
        assert [option.label for option in questions[0].questions[0].options] == ["Blue", "Green"]

        assert len(mock_server.main_requests) == 3
        nested_messages = json.dumps(mock_server.main_requests[2]["messages"])
        assert "responses" in nested_messages
        assert "Canary first" in nested_messages
        assert not mock_server.errors
        assert write_path.read_text(encoding="utf-8") == "nested permission bridge"
        progress = [
            event
            for event in captured[InvocationProgress]
            if (isinstance(event, InvocationProgress) and event.origin.kind == "sub_agent")
        ]
        assert progress[-1].total_usage_tokens == 31
        assert progress[-1].usage_unreported_attempts == 0
        # The child reports a cumulative 7 + 11 + 13 token snapshot. The
        # external-usage callback must add that authoritative total once.
        assert engine.session.runtime_meta.total_session_tokens == 31
    finally:
        await mock_server.close()

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``icode run`` on a real host: progress streams to stderr while the answer waits for stdout."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any
from unittest.mock import create_autospec

import pytest

import chrys.orchestration.engine.build.builder as builder_module
from chrys.app.cli import headless
from chrys.app.cli import run as run_cli
from chrys.app.cli.progress import display_path
from chrys.foundation.config.settings import Settings
from chrys.foundation.config.settings_store import LoadedSettings
from chrys.foundation.i18n import Localizer
from chrys.kernel import FunctionTool
from chrys.orchestration.session_host import ChrysSessionHost
from chrys.service.approval.policy import ApprovalMode
from chrys.service.llm.mock import MockChatClient, MockResponse
from chrys.service.tools.registry import ToolRegistry
from tests.orchestration.workflows._hosting import make_host
from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for

_DURATION = re.compile(r"\b\d+\.\ds\b")
_SESSION = re.compile(r"\bsession [0-9a-z]{12}\b")


def _progress(err: str) -> list[str]:
    return [_SESSION.sub("session <id>", _DURATION.sub("<dur>", line)) for line in err.splitlines()]


def _real_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, client: MockChatClient, tool: FunctionTool) -> None:
    """The command builds a real host (``make_host``) on a mock model, with *tool* as its only builtin.

    Each replacement is autospecced from the production callable, so a call production would reject fails here too.
    """

    async def create_client(
        _profile: Any,
        on_intermediate_text_async: Callable[[str], Awaitable[None]] | None = None,
        on_intermediate_text_sync: Callable[[str], None] | None = None,
        **_routing: object,
    ) -> MockChatClient:
        client._on_intermediate_text_async = on_intermediate_text_async
        client._on_intermediate_text_sync = on_intermediate_text_sync
        return client

    def load_builtins(self: ToolRegistry, _categories: list[str], *_args: object, **_kwargs: object) -> list[Any]:
        self.register(tool)
        return [tool]

    def build(*, profile_name: str, loaded_settings: LoadedSettings, approval_mode: ApprovalMode, **_kwargs: object):
        assert approval_mode is ApprovalMode.BYPASS
        return make_host(tmp_path, project=tmp_path, profile_name=profile_name, loaded_settings=loaded_settings)

    def prepare_runtime(*, restoring_session: bool = False) -> headless.PreparedRuntime:
        assert not restoring_session
        loaded = LoadedSettings(settings=Settings(model_profile="mock-profile"), provenance={})
        return headless.PreparedRuntime(loaded=loaded, localizer=Localizer("en"), pending_warnings=[])

    monkeypatch.setattr(
        builder_module, "create_client", create_autospec(builder_module.create_client, side_effect=create_client)
    )
    monkeypatch.setattr(
        ToolRegistry, "load_builtins", create_autospec(ToolRegistry.load_builtins, side_effect=load_builtins)
    )
    monkeypatch.setattr(run_cli, "ChrysSessionHost", create_autospec(ChrysSessionHost, side_effect=build))
    monkeypatch.setattr(
        headless, "prepare_runtime", create_autospec(headless.prepare_runtime, side_effect=prepare_runtime)
    )


@pytest.mark.parametrize("mode", ["text", "json", "quiet"])
async def test_progress_streams_while_a_tool_runs_and_the_answer_lands_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], mode: str
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def held(message: Annotated[str, "Text to echo"]) -> str:
        entered.set()
        await release.wait()
        return f"held: {message}"

    client = MockChatClient(
        responses=[
            MockResponse(text="Checking first.", tool_calls=[("held_tool", "c1", {"message": "hi"})]),
            MockResponse(text="final answer"),
        ]
    )
    _real_host(monkeypatch, tmp_path, client, FunctionTool(func=held, name="held_tool", description="Waits"))
    flags = {"text": [], "json": ["--json"], "quiet": ["--quiet"]}[mode]
    args = run_cli.build_parser().parse_args(["hello", "--agent", "Headless", *flags])

    task = asyncio.create_task(run_cli.run_command(args, run_cli.PreparedRuntimeHolder()))
    try:
        await wait_for(
            lambda: entered.is_set() or task.done(),
            timeout=ENGINE_TURN_TIMEOUT,
            description="the held tool is running",
        )
        assert not task.done()
        during = capsys.readouterr()
    finally:
        # The run owns a host: it always finishes (and shuts it down) before the test does.
        release.set()
        exit_code = await task
    assert exit_code == 0
    after = capsys.readouterr()

    # Nothing reaches stdout before the final answer, in every mode.
    assert during.out == ""
    if mode == "text":
        assert _progress(during.err) == [
            f"• Headless ready · mock · session <id> · {display_path(str(tmp_path))}",
            "Checking first.",
            "→ held_tool",
        ]
        assert after.out == "final answer\n"
        assert _progress(after.err) == ["  ✓ <dur>", "", "✓ Done · <dur> · 1 tool call · session <id>"]
    elif mode == "json":
        assert during.err == after.err == ""
        payload = json.loads(after.out)
        assert payload["result"] == "final answer"
    else:
        assert during.err == after.err == ""
        assert after.out == "final answer\n"

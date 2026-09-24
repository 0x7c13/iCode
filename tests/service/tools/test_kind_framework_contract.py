# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Characterization of the wire boundary that keeps chrys kinds off ``FunctionTool.kind``.

These tests pin the exact wire behavior that the out-of-band kind channel
in ``chrys.service.tools.kinds`` exists to defend against. The Chrys-owned OpenAI
**Responses** and **Anthropic** clients reserve the bare ``"shell"`` kind:
they silently replace any
``FunctionTool`` whose ``kind == "shell"`` with the provider's *hosted*
shell/bash tool, discarding the real name, JSON schema and description. The
OpenAI **Chat Completions** client does NOT (so the DeepSeek subclass path is
safe).

The wire serializers stay on the request path even with the chrys-owned tool
loop (tools travel to the wire client via options), so chrys stores kinds on
the chrys-owned ``chrys_kind`` attribute and leaves ``.kind`` as ``None`` —
the reservation can never match. These tests pin both halves: the upstream
hijack (bare ``"shell"`` on ``.kind``) and the production shape
(``kind=None`` + ``chrys_kind``) passing through untouched.

They are written against the Chrys-owned ``_prepare_tools_*`` methods so a
local serializer regression fails loudly.
"""

from __future__ import annotations

from types import SimpleNamespace

from chrys.kernel import FunctionTool
from chrys.service.llm.anthropic_chat import RawAnthropicClient
from chrys.service.llm.openai_chat_completion import RawOpenAIChatCompletionClient
from chrys.service.llm.openai_responses import RawOpenAIChatClient
from chrys.service.tools.kinds import KIND_SHELL, get_tool_kind, set_tool_kind


def _tool_with_schema(kind: str | None, *, name: str = "run_shell") -> FunctionTool:
    """A FunctionTool with a real ``command`` parameter, with ``.kind`` set to *kind*."""

    def run(command: str) -> str:
        return command

    return FunctionTool(name=name, description="Run a shell command", kind=kind, func=run)


def _chrys_shaped_shell_tool(*, name: str = "run_shell") -> FunctionTool:
    """The production shape: ``.kind`` is None, the chrys kind rides out of band."""
    t = _tool_with_schema(None, name=name)
    set_tool_kind(t, KIND_SHELL)
    return t


def _as_dict(item: object) -> dict:
    """Normalize a prepared tool to a dict.

    The Responses client emits a Pydantic ``FunctionShellTool`` for the hijack but a
    ``FunctionToolParam`` TypedDict for normal tools — normalize both to a plain dict.
    """
    if hasattr(item, "model_dump"):
        return item.model_dump()  # type: ignore[attr-defined]
    return dict(item)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# The reservation itself
# --------------------------------------------------------------------------- #


def test_upstream_reserves_exactly_shell_and_chrys_stays_off_kind() -> None:
    """The wire serializers reserve ``"shell"``; chrys avoids it by never writing ``.kind``."""
    assert KIND_SHELL == "shell"
    t = _chrys_shaped_shell_tool()
    assert t.kind is None
    assert get_tool_kind(t) == KIND_SHELL


def test_instance_binding_preserves_out_of_band_tool_context() -> None:
    """The provenance context channel rides the same ``copy.copy`` clone as the kind."""
    from chrys.foundation.tool_call_context import (
        get_tool_context,
        resolve_tool_call_context,
        set_tool_context,
        set_tool_context_builder,
    )

    class Tools:
        def execute(self, command: str) -> str:
            return command

    Tools.execute = FunctionTool(name="execute", description="d", func=Tools.execute)
    set_tool_context(Tools.execute, {"server_name": "probe"})
    set_tool_context_builder(Tools.execute, lambda args: {"extra": "built"})
    bound = Tools().execute
    assert bound is not Tools.__dict__["execute"]
    assert get_tool_context(bound) == {"server_name": "probe"}
    assert resolve_tool_call_context(bound, {}) == {"server_name": "probe", "extra": "built"}


# --------------------------------------------------------------------------- #
# OpenAI Responses API (OpenAIChatClient) — hijacks bare "shell"
# --------------------------------------------------------------------------- #


def test_openai_responses_hijacks_bare_shell_kind() -> None:
    """A FunctionTool with bare ``kind="shell"`` becomes the hosted shell tool (schema lost)."""
    client = RawOpenAIChatClient(model="gpt-4o", async_client=SimpleNamespace(base_url=""))
    prepared = client._prepare_tools_for_openai([_tool_with_schema(KIND_SHELL)])
    assert len(prepared) == 1
    item = _as_dict(prepared[0])
    assert item["type"] == "shell"
    # The real declaration is gone — no name, no JSON schema reaches the model.
    assert "name" not in item
    assert "parameters" not in item


def test_openai_responses_preserves_chrys_shaped_tools() -> None:
    """``kind=None`` + out-of-band ``chrys_kind`` keeps the tool as a normal function tool."""
    client = RawOpenAIChatClient(model="gpt-4o", async_client=SimpleNamespace(base_url=""))
    prepared = client._prepare_tools_for_openai([_chrys_shaped_shell_tool()])
    assert len(prepared) == 1
    item = _as_dict(prepared[0])
    assert item["type"] == "function"
    assert item["name"] == "run_shell"
    assert "command" in item["parameters"]["properties"]


def test_openai_responses_does_not_touch_non_shell_kinds() -> None:
    """Only ``"shell"`` is reserved — other bare kinds pass through with their schema."""
    client = RawOpenAIChatClient(model="gpt-4o", async_client=SimpleNamespace(base_url=""))
    prepared = client._prepare_tools_for_openai([_tool_with_schema("filesystem.write", name="write_file")])
    item = _as_dict(prepared[0])
    assert item["type"] == "function"
    assert item["name"] == "write_file"
    assert "command" in item["parameters"]["properties"]


# --------------------------------------------------------------------------- #
# Anthropic (AnthropicClient) — hijacks bare "shell"
# --------------------------------------------------------------------------- #


def test_anthropic_hijacks_bare_shell_kind() -> None:
    """Bare ``kind="shell"`` is forced to the hosted ``bash`` tool (schema + name lost)."""
    client = RawAnthropicClient(model="claude-sonnet-4-5", anthropic_client=SimpleNamespace())
    result = client._prepare_tools_for_anthropic({"tools": [_tool_with_schema(KIND_SHELL)]})
    assert result is not None
    tools = result["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == "bash"
    assert "input_schema" not in tools[0]


def test_anthropic_preserves_chrys_shaped_tools() -> None:
    """``kind=None`` + out-of-band ``chrys_kind`` keeps the tool as a custom tool with schema."""
    client = RawAnthropicClient(model="claude-sonnet-4-5", anthropic_client=SimpleNamespace())
    result = client._prepare_tools_for_anthropic({"tools": [_chrys_shaped_shell_tool()]})
    assert result is not None
    tools = result["tools"]
    assert tools[0]["type"] == "custom"
    assert tools[0]["name"] == "run_shell"
    assert "command" in tools[0]["input_schema"]["properties"]


# --------------------------------------------------------------------------- #
# OpenAI Chat Completions (OpenAIChatCompletionClient) — the SAFE asymmetry
# --------------------------------------------------------------------------- #


def test_chat_completions_never_hijacks_shell_kind() -> None:
    """The Chat Completions path ignores ``kind`` entirely — DeepSeek (a subclass) relies on this."""
    client = RawOpenAIChatCompletionClient(model="gpt-4o", async_client=SimpleNamespace(base_url=""))
    result = client._prepare_tools_for_openai([_tool_with_schema(KIND_SHELL)])
    tools = result["tools"]
    assert len(tools) == 1
    spec = tools[0]
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "run_shell"
    assert "command" in spec["function"]["parameters"]["properties"]

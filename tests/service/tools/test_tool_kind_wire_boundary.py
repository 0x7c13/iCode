# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The wire serializers ignore ``FunctionTool.kind``; Chrys kinds ride out of band.

Chrys records a tool's kind on the chrys-owned ``chrys_kind`` attribute
(``chrys.service.tools.kinds``) and leaves ``FunctionTool.kind`` as ``None``.
The Responses, Anthropic and Chat Completions serializers send every function
tool with its own name and JSON schema whatever ``.kind`` holds, so a
``"shell"`` kind never turns a declared tool into a provider-hosted shell.
These tests pin both halves against the Chrys-owned tool serializers.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import pytest

from chrys.kernel import FunctionTool
from chrys.service.llm.anthropic_messages.request import encode_tools
from chrys.service.llm.chat_completions import request as chat_completions_request
from chrys.service.llm.openai_responses import request as responses_request
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


# --------------------------------------------------------------------------- #
# The out-of-band channel
# --------------------------------------------------------------------------- #


def test_chrys_kinds_stay_off_function_tool_kind() -> None:
    """Chrys records a kind out of band and never writes ``.kind``."""
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


# Factories, so each test serializes a tool no other test has touched.
_TOOLS = pytest.mark.parametrize(
    ("make_tool", "name"),
    [
        (partial(_tool_with_schema, KIND_SHELL), "run_shell"),
        (_chrys_shaped_shell_tool, "run_shell"),
        (partial(_tool_with_schema, "filesystem.write", name="write_file"), "write_file"),
    ],
    ids=["bare-shell-kind", "chrys-shaped", "other-bare-kind"],
)


# --------------------------------------------------------------------------- #
# Each serializer sends the declared function tool
# --------------------------------------------------------------------------- #


@_TOOLS
def test_openai_responses_sends_the_declared_function_tool(make_tool: Callable[[], FunctionTool], name: str) -> None:
    (item,) = responses_request.encode_tools([make_tool()])
    assert item["type"] == "function"
    assert item["name"] == name
    assert "command" in item["parameters"]["properties"]


@_TOOLS
def test_anthropic_sends_the_declared_custom_tool(make_tool: Callable[[], FunctionTool], name: str) -> None:
    result = encode_tools({"tools": [make_tool()]})
    assert result is not None
    (item,) = result["tools"]
    assert item["type"] == "custom"
    assert item["name"] == name
    assert "command" in item["input_schema"]["properties"]


@_TOOLS
def test_chat_completions_sends_the_declared_function_tool(make_tool: Callable[[], FunctionTool], name: str) -> None:
    (spec,) = chat_completions_request.encode_tools([make_tool()])["tools"]
    assert spec["type"] == "function"
    assert spec["function"]["name"] == name
    assert "command" in spec["function"]["parameters"]["properties"]

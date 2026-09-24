# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for MCPAdapter server instructions: the instructions map, reminder rendering, and the exposure flag."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from chrys.service.mcp.adapter import _SEVERED_ENTITY_TAIL_RE, MCP_INSTRUCTIONS_CHAR_LIMIT, MCPAdapter
from chrys.service.profiles.agents.schema import MCPServerConfig
from tests.service.mcp._helpers import _FakeConnectionTool, _function_tool


def _instructions_tool(instructions: str | None) -> MagicMock:
    tool = MagicMock()
    tool._server_instructions = instructions
    return tool


def _adapter_with_instructions(instructions: dict[str, str | None]) -> MCPAdapter:
    """An adapter whose registered servers carry the given instructions (registration order kept)."""
    adapter = MCPAdapter()
    for name, text in instructions.items():
        adapter._servers[name] = _instructions_tool(text)
    return adapter


def _lease_with_instructions(instructions: str) -> MagicMock:
    lease = MagicMock()
    lease.mcp_tool._server_instructions = instructions
    return lease


# ---------------------------------------------------------------------------
# get_server_instructions_map
# ---------------------------------------------------------------------------


def test_get_server_instructions_map_empty() -> None:
    """Empty when no servers have instructions."""
    adapter = MCPAdapter()
    assert adapter.get_server_instructions_map() == {}


def test_get_server_instructions_map_from_servers() -> None:
    """Aggregates from _servers when instructions are set."""
    adapter = _adapter_with_instructions({"srv_a": "Instructions for A.", "srv_b": "Instructions for B."})

    result = adapter.get_server_instructions_map()
    assert result == {"srv_a": "Instructions for A.", "srv_b": "Instructions for B."}


def test_get_server_instructions_map_skips_none() -> None:
    """Servers with None instructions are excluded."""
    adapter = _adapter_with_instructions({"srv": None})

    assert adapter.get_server_instructions_map() == {}


@pytest.mark.parametrize(
    ("exposure", "expected"),
    [
        pytest.param(None, {"leased": "From lease."}, id="exposed"),
        pytest.param(False, {}, id="hidden"),
    ],
)
def test_get_server_instructions_map_from_leases(exposure: bool | None, expected: dict[str, str]) -> None:
    """Aggregates from _leases when _servers lacks the entry; the fallback honors the exposure flag too."""
    adapter = MCPAdapter()
    adapter._leases["leased"] = _lease_with_instructions("From lease.")
    if exposure is not None:
        adapter._instructions_exposure["leased"] = exposure

    assert adapter.get_server_instructions_map() == expected


def test_get_server_instructions_map_servers_win_over_leases() -> None:
    """_servers takes priority over _leases for the same server name."""
    adapter = _adapter_with_instructions({"srv": "From servers."})
    adapter._leases["srv"] = _lease_with_instructions("From leases.")

    # _servers is iterated first, _leases only fills missing names
    assert adapter.get_server_instructions_map() == {"srv": "From servers."}


def test_get_server_instructions_map_respects_expose_instructions_flag() -> None:
    """Servers registered with expose_instructions=False are excluded from the map."""
    adapter = _adapter_with_instructions({"shown": "Visible.", "hidden": "Hidden."})
    adapter._instructions_exposure["shown"] = True
    adapter._instructions_exposure["hidden"] = False

    assert adapter.get_server_instructions_map() == {"shown": "Visible."}
    reminder = adapter.render_instructions_reminder()
    assert reminder is not None
    assert "Visible." in reminder
    assert "Hidden." not in reminder


def test_get_server_instructions_map_defaults_to_exposed_without_registration() -> None:
    """A server with no exposure record (e.g. hand-registered in tests) stays visible."""
    adapter = _adapter_with_instructions({"srv": "Default-visible."})

    assert adapter.get_server_instructions_map() == {"srv": "Default-visible."}


async def test_connect_registers_instructions_exposure_and_disconnect_clears_it() -> None:
    """The production connect path records the flag; disconnect paths drop it."""
    adapter = MCPAdapter()
    hidden_cfg = MCPServerConfig(name="hidden", transport="stdio", command="python", expose_instructions=False)
    shown_cfg = MCPServerConfig(name="shown", transport="stdio", command="python")
    fake_hidden = _FakeConnectionTool(functions=[_function_tool("remote_hidden")])
    fake_hidden._server_instructions = "Hidden."
    fake_shown = _FakeConnectionTool(functions=[_function_tool("remote_shown")])
    fake_shown._server_instructions = "Shown."

    with patch("chrys.service.mcp._connection._create_mcp_tool", side_effect=[fake_hidden, fake_shown]):
        await adapter.connect(hidden_cfg)
        await adapter.connect(shown_cfg)

    assert adapter._instructions_exposure == {"hidden": False, "shown": True}
    assert adapter.get_server_instructions_map() == {"shown": "Shown."}

    await adapter.disconnect("hidden")
    assert adapter._instructions_exposure == {"shown": True}

    await adapter.disconnect_all()
    assert adapter._instructions_exposure == {}


# ---------------------------------------------------------------------------
# render_instructions_reminder
# ---------------------------------------------------------------------------


def test_render_instructions_reminder_empty() -> None:
    """None when no servers have instructions."""
    adapter = MCPAdapter()
    assert adapter.render_instructions_reminder() is None


def test_render_instructions_reminder_single_server() -> None:
    """Single server rendered in XML format."""
    adapter = _adapter_with_instructions({"filesystem": "Use this server for file operations."})

    result = adapter.render_instructions_reminder()
    assert result is not None
    assert result.startswith("<mcp_instructions>")
    assert result.endswith("</mcp_instructions>")
    assert '<server name="filesystem">' in result
    assert "  </server>" in result
    assert "    Use this server for file operations." in result


def test_render_instructions_reminder_multi_server() -> None:
    """Multiple servers each get their own <server> element."""
    adapter = _adapter_with_instructions({"github": "GitHub API access.", "weather": "Weather data provider."})

    result = adapter.render_instructions_reminder()
    assert result is not None
    assert '<server name="github">' in result
    assert "    GitHub API access." in result
    assert '<server name="weather">' in result
    assert "    Weather data provider." in result
    # Both servers present, ordered by iteration
    assert result.index("github") < result.index("weather")


def test_render_instructions_reminder_sorted_by_server_name() -> None:
    """Rendering is name-ordered even when registration order differs."""
    adapter = _adapter_with_instructions({"zeta": "Z instructions.", "alpha": "A instructions."})

    result = adapter.render_instructions_reminder()
    assert result is not None
    assert result.index('<server name="alpha">') < result.index('<server name="zeta">')


def test_render_instructions_reminder_multiline_instructions() -> None:
    """Multi-line instructions are indented per line."""
    adapter = _adapter_with_instructions({"srv": "Line one.\nLine two.\nLine three."})

    result = adapter.render_instructions_reminder()
    assert result is not None
    assert "    Line one." in result
    assert "    Line two." in result
    assert "    Line three." in result
    # Check ordering: each line indented at the same level
    idx_one = result.index("    Line one.")
    idx_two = result.index("    Line two.")
    idx_three = result.index("    Line three.")
    assert idx_one < idx_two < idx_three


def test_render_instructions_reminder_crlf_and_blank_lines() -> None:
    """CRLF newlines are normalized and blank lines carry no trailing spaces."""
    adapter = _adapter_with_instructions({"srv": "First.\r\n\r\nSecond."})

    result = adapter.render_instructions_reminder()
    assert result is not None
    assert "\r" not in result
    lines = result.split("\n")
    assert "    First." in lines
    assert "    Second." in lines
    # The blank instruction line renders empty, not as indent whitespace.
    assert "" in lines[lines.index("    First.") + 1 : lines.index("    Second.")]
    assert not any(line != line.rstrip() for line in lines)


@pytest.mark.parametrize(
    ("server_name", "instructions", "expected"),
    [
        # The quote needs to be escaped for the attribute value
        pytest.param('test"&<>', "Test.", 'name="test&quot;&amp;&lt;&gt;"', id="name"),
        pytest.param("srv", 'Use <tag> & "quotes".', "    Use &lt;tag&gt; &amp; &quot;quotes&quot;.", id="content"),
    ],
)
def test_render_instructions_reminder_escapes_xml(server_name: str, instructions: str, expected: str) -> None:
    """Server names and instruction text with XML special characters are escaped."""
    adapter = _adapter_with_instructions({server_name: instructions})

    result = adapter.render_instructions_reminder()
    assert result is not None
    assert expected in result


@pytest.mark.parametrize(
    ("instructions", "kept_fragment"),
    [
        pytest.param("x" * (MCP_INSTRUCTIONS_CHAR_LIMIT + 500), None, id="oversized"),
        # Each '"' renders as '&quot;' (6 chars); a raw-input cap would admit ~6x the
        # budget.  The leading 'a' shifts the cut point mid-entity so the
        # severed-fragment cleanup path is exercised too.
        pytest.param("a" + '"' * (MCP_INSTRUCTIONS_CHAR_LIMIT + 1), None, id="escaped-output-counts"),
        pytest.param(
            "short intro line\n" + "y" * (MCP_INSTRUCTIONS_CHAR_LIMIT * 2),
            "    short intro line",
            id="multiline-keeps-leading-lines",
        ),
        pytest.param("x\n" + "\n" * (MCP_INSTRUCTIONS_CHAR_LIMIT * 3) + "x", None, id="blank-line-flood"),
    ],
)
def test_render_instructions_reminder_caps_server_controlled_instructions(
    instructions: str,
    kept_fragment: str | None,
) -> None:
    """Server-controlled instructions are truncated at MCP_INSTRUCTIONS_CHAR_LIMIT.

    The budget is charged post-escaping (entity expansion cannot multiply the
    cap), complete leading lines are kept with the cut inside the overflowing
    one, thousands of blank lines cannot bypass the budget via free newline
    separators, and the cut never leaves a severed entity fragment behind.
    """
    adapter = _adapter_with_instructions({"srv": instructions})

    result = adapter.render_instructions_reminder()
    assert result is not None
    assert len(result) < MCP_INSTRUCTIONS_CHAR_LIMIT + 200
    assert f"[instructions truncated: exceeded {MCP_INSTRUCTIONS_CHAR_LIMIT} characters]" in result
    if kept_fragment is not None:
        assert kept_fragment in result
    # The cut must not leave a severed entity fragment at the truncation point.
    kept = result.split("\n")[2]
    assert not kept.endswith("&") and not _SEVERED_ENTITY_TAIL_RE.search(kept)

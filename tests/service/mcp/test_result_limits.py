# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for MCP result caps, spills, prompt exemptions, and binary preflight."""

from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Event as ThreadEvent

import pytest

from chrys.foundation.text.tokenizer import MixedLanguageTokenizer
from chrys.foundation.tool_call_context import get_tool_context, set_tool_context
from chrys.foundation.tool_kinds import KIND_MCP, get_tool_kind, set_tool_kind
from chrys.kernel import Content, FunctionTool
from chrys.kernel.tools import SyncToolCancelledAfterCompletion
from chrys.service.mcp import result_limits
from chrys.service.mcp.cache import clone_mcp_function_tool
from chrys.service.mcp.owned import _MCP_PROMPT_TOOL_KEY
from chrys.service.mcp.result_limits import sanitize_mcp_result_binary, truncate_mcp_result
from chrys.service.tools.spill import TOOL_RESULTS_DIR_NAME

_tokenizer = MixedLanguageTokenizer()


async def _large_text() -> str:
    return "HEAD" + "x" * 10_000 + "TAIL"


def _source_tool(*, prompt: bool = False) -> FunctionTool:
    tool = FunctionTool(
        func=_large_text,
        name="remote",
        description="remote tool",
        input_model={"type": "object", "properties": {}},
        additional_properties={_MCP_PROMPT_TOOL_KEY: True} if prompt else {},
    )
    set_tool_kind(tool, KIND_MCP)
    set_tool_context(tool, {"server_name": "srv", "remote_name": "remote"})
    return tool


async def test_string_result_is_capped_and_spilled(tmp_path: Path) -> None:
    text = await _large_text()

    result = await truncate_mcp_result(text, 100, tmp_path)

    assert "truncated" in result
    assert "Full output saved to:" in result
    assert _tokenizer.count_tokens(result) <= 100
    spills = list((tmp_path / TOOL_RESULTS_DIR_NAME).glob("mcp_*.txt"))
    assert len(spills) == 1
    assert spills[0].read_text() == text


async def test_completed_mcp_result_survives_spill_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chrys.service.tools import spill

    started = ThreadEvent()
    release = ThreadEvent()
    real_write = spill.atomic_write_owner_only_bytes

    def blocking_write(path: Path, payload: bytes) -> None:
        started.set()
        assert release.wait(timeout=5)
        real_write(path, payload)

    monkeypatch.setattr(spill, "atomic_write_owner_only_bytes", blocking_write)

    async def capture_completed() -> str:
        try:
            return await truncate_mcp_result("x" * 10_000, 100, tmp_path)
        except SyncToolCancelledAfterCompletion as exc:
            return exc.completed_result

    task = asyncio.create_task(capture_completed())
    assert await asyncio.to_thread(started.wait, 2)
    task.cancel()
    release.set()

    result = await task

    assert "truncated" in result
    assert "Full output saved to:" in result
    assert list((tmp_path / TOOL_RESULTS_DIR_NAME).glob("mcp_*.txt"))


async def test_list_result_is_copy_bounded_and_keeps_non_text_items(tmp_path: Path) -> None:
    first = Content.from_text("x" * 10_000)
    image = Content.from_data(b"image", "image/png")
    later = Content.from_text("later text")
    source = [first, image, later]

    result = await truncate_mcp_result(source, 100, tmp_path)

    assert result is not source
    assert result[0] is not first
    assert result[1] is image
    assert later not in result
    assert first.text == "x" * 10_000
    assert "[1 further text items omitted.]" in result[0].text
    assert _tokenizer.count_tokens(result[0].text) <= 100
    spill_file = next((tmp_path / TOOL_RESULTS_DIR_NAME).glob("mcp_*.txt"))
    assert spill_file.read_text() == f"{'x' * 10_000}\n\nlater text"


async def test_list_result_keeps_spill_notice_when_first_item_fits_its_slot(tmp_path: Path) -> None:
    first = Content.from_text("中" * 132)
    later = Content.from_text("x" * 10_000)

    result = await truncate_mcp_result([first, later], 100, tmp_path)

    text = next(item.text for item in result if item.type == "text" and item.text is not None)
    assert text != first.text
    assert "[1 further text items omitted.]" in text
    assert "Full output saved to:" in text
    spills = list((tmp_path / TOOL_RESULTS_DIR_NAME).glob("mcp_*.txt"))
    assert len(spills) == 1
    assert str(spills[0]) in text


async def test_defensive_notice_loss_discards_spill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_truncate = result_limits.truncate_content_texts

    def lose_notice(
        texts: list[str],
        budget: int,
        *,
        truncation_suffix: str = "",
    ) -> list[str | None] | None:
        if truncation_suffix:
            return [texts[0], *([None] * (len(texts) - 1))]
        return real_truncate(texts, budget)

    monkeypatch.setattr(result_limits, "truncate_content_texts", lose_notice)

    result = await truncate_mcp_result([Content.from_text("x" * 10_000), Content.from_text("later")], 100, tmp_path)

    assert all("Full output saved to:" not in (item.text or "") for item in result if item.type == "text")
    assert not list((tmp_path / TOOL_RESULTS_DIR_NAME).glob("mcp_*.txt"))


async def test_clone_caps_regular_tool_and_preserves_metadata(tmp_path: Path) -> None:
    source = _source_tool()
    clone = clone_mcp_function_tool(source, result_cap=100, spill_dir=tmp_path)

    result = await clone.func()

    assert "truncated" in result
    assert clone.additional_properties == source.additional_properties
    assert get_tool_kind(clone) == KIND_MCP
    assert get_tool_context(clone) == get_tool_context(source)
    assert clone.parameters() == source.parameters()


async def test_prompt_clone_is_not_producer_capped(tmp_path: Path) -> None:
    clone = clone_mcp_function_tool(_source_tool(prompt=True), result_cap=100, spill_dir=tmp_path)

    result = await clone.func()

    assert result == await _large_text()
    assert not (tmp_path / TOOL_RESULTS_DIR_NAME).exists()


def test_binary_preflight_replaces_oversized_items(monkeypatch: pytest.MonkeyPatch) -> None:
    from mcp import types

    monkeypatch.setattr(result_limits, "MCP_BINARY_MAX_BYTES", 5)
    raw = types.CallToolResult(
        content=[types.ImageContent(type="image", data="A" * 12, mimeType="image/png")],
        isError=False,
    )

    sanitized = sanitize_mcp_result_binary(raw)

    assert isinstance(sanitized.content[0], types.TextContent)
    assert "image omitted" in sanitized.content[0].text
    assert isinstance(raw.content[0], types.ImageContent)


def test_binary_preflight_collapses_aggregate_overflow_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from mcp import types

    monkeypatch.setattr(result_limits, "MCP_BINARY_MAX_BYTES", 10)
    monkeypatch.setattr(result_limits, "MCP_RESULT_MAX_TOTAL_BINARY_BYTES", 15)
    raw = types.CallToolResult(
        content=[
            types.AudioContent(type="audio", data="A" * 12, mimeType="audio/wav"),
            types.TextContent(type="text", text="between"),
            types.ImageContent(type="image", data="A" * 12, mimeType="image/png"),
            types.ImageContent(type="image", data="A" * 12, mimeType="image/png"),
        ],
        isError=False,
    )

    sanitized = sanitize_mcp_result_binary(raw)
    texts = [item.text for item in sanitized.content if isinstance(item, types.TextContent)]

    assert "between" in texts
    assert sum("binary items omitted" in text for text in texts) == 1
    assert any("2 binary items omitted" in text for text in texts)

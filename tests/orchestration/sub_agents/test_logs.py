# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for sub-agent audit log storage helpers."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest

from chrys.foundation.trajectory.metadata import ensure_analytics_item_id
from chrys.kernel import Message
from chrys.service.session import sub_agent_logs
from chrys.service.session.sub_agent_logs import (
    SUB_AGENT_LOG_PREVIEW_MAX_CHARS,
    SUB_AGENT_LOG_TMP_PREFIX,
    SUB_AGENT_LOG_TMP_SUFFIX,
    SUB_AGENT_TRANSCRIPT_OCCURRENCE_IDENTITY,
    SubAgentLogStats,
    SubAgentModelProvenance,
    SubAgentSessionLogWriter,
    make_sub_agent_log_file_name,
    preview_text,
    sanitize_base_url_origin,
    sessions_dir,
)


def test_log_file_name_is_bounded_and_preserves_invocation_id() -> None:
    invocation_id = "a1b2c3d4e5f6"
    filename = make_sub_agent_log_file_name("Explore/" + ("任务" * 200), invocation_id)
    temp_name = f"{SUB_AGENT_LOG_TMP_PREFIX}{filename}{SUB_AGENT_LOG_TMP_SUFFIX}"

    assert filename.endswith(f"_{invocation_id}.json")
    assert len(filename.encode("utf-8")) <= 255
    assert len(temp_name.encode("utf-8")) <= 255


@pytest.mark.parametrize("tool_name", ["", ".", "..", "CON", "AUX", "COM1", "LPT9", "name. ", "bad/name"])
def test_log_file_name_rewrites_platform_unsafe_stems(tool_name: str) -> None:
    filename = make_sub_agent_log_file_name(tool_name, "a1b2c3d4e5f6")
    stem = filename.removesuffix("_a1b2c3d4e5f6.json")

    assert stem
    assert stem not in {".", ".."}
    assert not stem.endswith((" ", "."))
    assert "/" not in stem
    assert stem.upper() not in {"CON", "AUX", "COM1", "LPT9"}


def test_preview_text_truncates_within_cap() -> None:
    text = "x" * (SUB_AGENT_LOG_PREVIEW_MAX_CHARS + 20)
    preview = preview_text(text)

    assert len(preview) == SUB_AGENT_LOG_PREVIEW_MAX_CHARS
    assert preview.endswith("...")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://user:pass@example.com:8443/path?q=1#frag", "https://example.com:8443"),
        ("http://example.com/v1", "http://example.com"),
        ("not a url", ""),
        ("ftp://example.com/path", ""),
    ],
)
def test_sanitize_base_url_origin(raw: str, expected: str) -> None:
    assert sanitize_base_url_origin(raw) == expected


async def test_sub_agent_log_writer_writes_envelope(tmp_path: Path) -> None:
    stats = SubAgentLogStats(tool_call_count=2, total_tokens=100, total_usage_tokens=250)
    writer = SubAgentSessionLogWriter(
        parent_session_dir=tmp_path,
        parent_session_id="parent-session",
        invocation_id="a1b2c3d4e5f6",
        tool_name="Explore",
        agent_profile="Explore",
        agent_display_name="Explore Agent",
        prompt="inspect persistence",
        parent_provider_call_id="provider-call",
        parent_event_call_id="event-call",
        model=SubAgentModelProvenance(
            provider="openai",
            api_style="responses",
            model_id="gpt-test",
            base_url_origin="https://api.openai.com",
        ),
        primary_cwd="/workspace",
        working_dirs=["/workspace"],
        stats=stats,
    )

    message = Message("user", ["hello"])
    ensure_analytics_item_id(message.additional_properties)
    written = await writer.write(status="completed", state={"messages": [message]}, result="done")

    assert written is True
    path = sessions_dir(tmp_path) / writer.log_file_name
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["meta"]["record_type"] == "sub_agent_session"
    assert data["meta"]["status"] == "completed"
    assert data["meta"]["parent_provider_call_id"] == "provider-call"
    assert data["meta"]["parent_event_call_id"] == "event-call"
    assert data["meta"]["tool_call_count"] == 2
    assert data["meta"]["total_usage_tokens"] == 250
    assert data["meta"]["model_base_url_origin"] == "https://api.openai.com"
    assert data["meta"]["transcript_occurrence_identity"] == SUB_AGENT_TRANSCRIPT_OCCURRENCE_IDENTITY
    assert data["state"]["messages"][0]["role"] == "user"


async def test_sub_agent_log_writer_omits_identity_capability_for_incomplete_history(tmp_path: Path) -> None:
    writer = SubAgentSessionLogWriter(
        parent_session_dir=tmp_path,
        parent_session_id="parent-session",
        invocation_id="a1b2c3d4e5f6",
        tool_name="Explore",
        agent_profile="Explore",
        agent_display_name="Explore Agent",
        prompt="inspect legacy persistence",
        parent_provider_call_id="provider-call",
        parent_event_call_id="event-call",
    )

    written = await writer.write(status="completed", state={"messages": [Message("user", ["legacy"])]})

    assert written is True
    data = json.loads((sessions_dir(tmp_path) / writer.log_file_name).read_text(encoding="utf-8"))
    assert "transcript_occurrence_identity" not in data["meta"]


async def test_writer_cancellation_drains_in_flight_write_before_terminal_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    writes: list[str] = []

    def _slow_write(path: Path, envelope: dict) -> None:
        writes.append(envelope["meta"]["status"])
        if envelope["meta"]["status"] == "running":
            started.set()
            assert release.wait(timeout=5)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(envelope), encoding="utf-8")

    monkeypatch.setattr(sub_agent_logs, "_atomic_write_json", _slow_write)
    writer = SubAgentSessionLogWriter(
        parent_session_dir=tmp_path,
        parent_session_id="parent-session",
        invocation_id="a1b2c3d4e5f6",
        tool_name="Explore",
        agent_profile="Explore",
        agent_display_name="Explore",
        prompt="inspect persistence",
        parent_provider_call_id="provider-call",
        parent_event_call_id="event-call",
    )

    initial = asyncio.create_task(writer.write(status="running"))
    assert await asyncio.to_thread(started.wait, 5)
    initial.cancel()
    await asyncio.sleep(0)
    terminal = asyncio.create_task(writer.write(status="cancelled", result="cancelled", ended=True))
    await asyncio.sleep(0)
    assert not terminal.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await initial
    assert await terminal is True

    data = json.loads((sessions_dir(tmp_path) / writer.log_file_name).read_text(encoding="utf-8"))
    assert data["meta"]["status"] == "cancelled"
    assert writes == ["running", "cancelled"]

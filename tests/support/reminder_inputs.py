# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Fixed inputs for the reminder tests that drive a reminder stack (``tests/support/reminder_stack.py``).

``pin_reminder_inputs`` logs clock reads and stubs Python-path discovery,
and ``RUNTIME`` is a fixed runtime environment.  ``ReminderProviders``
stands in for the providers a reminder reads, and the helpers build
messages, spill quotas, archive catalogs and manifest entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from chrys.foundation.models.session_env import SessionEnvironment
from chrys.foundation.models.workspace import WorkingDir
from chrys.foundation.platform import PlatformInfo, ShellInfo
from chrys.kernel import Content, Message
from chrys.service.agent_middleware.reminders import runtime_env, turn_line
from chrys.service.context.compaction.last_words_state import ManifestEntry
from chrys.service.context.compaction.spill import (
    CATALOG_RELATIVE_PATH,
    SpillQuota,
    build_record_filename,
    dropped_turn_relative_path,
)
from tests.support.secure_files import plant_owner_only_bytes

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from datetime import tzinfo

MAX_CONTEXT_TOKENS = 200_000

RUNTIME = SessionEnvironment(
    cwd="/work/example",
    platform=PlatformInfo(
        os_name="linux",
        os_version="24.04",
        arch="amd64",
        shell=ShellInfo(name="bash", path="/usr/bin/bash", args=["-c"], version="5.2.21"),
        config_dir=Path("/home/example/.chrys"),
        data_dir=Path("/home/example/.chrys"),
        extra_shells=(ShellInfo(name="zsh", path="/usr/bin/zsh", args=["-c"], version="5.9"),),
    ),
    session_id="example",
    created_at=datetime(2026, 9, 30, 1, 2, 3, tzinfo=UTC),
    working_dirs=(WorkingDir("/work/example", "app", is_primary=True), WorkingDir("/work/shared")),
)
TODO_A = "Current todo list:\n- [ ] pin the reminder inputs"
TODO_B = "Current todo list:\n- [x] pin the reminder inputs\n- [ ] run the gates"
SKILLS = "<available_skills>\n  <skill>\n    <name>review</name>\n  </skill>\n</available_skills>"
SKILLS_REFRESHED = (
    "<available_skills>\n  <skill>\n    <name>review</name>\n  </skill>\n"
    "  <skill>\n    <name>release</name>\n  </skill>\n</available_skills>"
)
MCP = '<mcp_instructions>\n  <server name="docs">Search before fetching a page.</server>\n</mcp_instructions>'


def _record_path(turn: int, sequence: int, tool: str, record_id: str) -> str:
    return (dropped_turn_relative_path(turn) / build_record_filename(sequence, tool, record_id)).as_posix()


ARCHIVED = tuple(_record_path(1, index, "read_file", f"{index:08x}") for index in range(1, 6))
"""Records earlier turns archived, as the catalog lists them."""


# ---------------------------------------------------------------------------
# Pinned inputs
# ---------------------------------------------------------------------------


def pin_reminder_inputs(monkeypatch: pytest.MonkeyPatch, *, log: list[str] | None = None) -> None:
    """Log clock reads and stub Python-path discovery where the reminder pipeline reads them.

    The clock keeps the real time.  Discovery finds no Python, so no
    executable is probed.  When *log* is given, each read appends
    ``"clock"`` or ``"python_paths"``.
    """

    class _LoggedDateTime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> _LoggedDateTime:
            if log is not None:
                log.append("clock")
            return super().now(tz)

    def _python_execution_paths() -> list[Any]:
        if log is not None:
            log.append("python_paths")
        return []

    monkeypatch.setattr(turn_line, "datetime", _LoggedDateTime)
    monkeypatch.setattr(runtime_env, "_python_execution_paths", _python_execution_paths)


@dataclass
class ReminderProviders:
    """What the providers return; a test changes a field between turns."""

    todo: str | None = None
    skills: str | None = None
    mcp: str | None = None
    file_change: str | None = None
    log: list[str] | None = None

    def read_todo(self) -> str | None:
        self._log("todo")
        return self.todo

    def read_skills(self) -> str | None:
        self._log("skills")
        return self.skills

    def read_mcp(self) -> str | None:
        self._log("mcp")
        return self.mcp

    def drain_file_change(self) -> str | None:
        """The workspace tracker hands its notice over once."""
        self._log("file_change")
        notice, self.file_change = self.file_change, None
        return notice

    def _log(self, name: str) -> None:
        if self.log is not None:
            self.log.append(name)


class LoggingSpillQuota(SpillQuota):
    """A real quota that logs the pointer's count reads."""

    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self.log = log

    def live_record_count(self, *, excluded_relative_paths: Iterable[str] = ()) -> int:
        self.log.append("pointer_count")
        return super().live_record_count(excluded_relative_paths=excluded_relative_paths)


def spill_quota(*, live: Sequence[str]) -> SpillQuota:
    """A quota whose catalog lists *live* records."""
    quota = SpillQuota()
    quota.initialize(0, (), live_relative_paths=list(live))
    return quota


def plant_catalog(session_root: Path) -> None:
    """Create the session's archive catalog, so the pointer can name it."""
    catalog = session_root / CATALOG_RELATIVE_PATH
    catalog.parent.mkdir(parents=True, exist_ok=True)
    plant_owner_only_bytes(catalog, b"")


def manifest_entry(turn: int, sequence: int, tool: str, argument: str) -> ManifestEntry:
    """A dropped tool call whose result was archived."""
    record_id = f"{turn:04x}{sequence:04x}"
    return ManifestEntry(
        record_id=record_id,
        group_id=f"group_{turn}_{sequence}",
        record_dir=dropped_turn_relative_path(turn).as_posix(),
        relative_path=_record_path(turn, sequence, tool, record_id),
        turn=turn,
        round=1,
        sequence=sequence,
        tool=tool,
        display_argument=argument,
        outcome="ok",
        size_chars=1_200,
        assistant_text=False,
        no_record_reason="",
    )


def usage(percent: int) -> dict[str, int]:
    return {"total_token_count": MAX_CONTEXT_TOKENS * percent // 100}


def user(text: str) -> Message:
    return Message(role="user", contents=[Content.from_text(text)])


def assistant(text: str) -> Message:
    return Message(role="assistant", contents=[Content.from_text(text)])


def require[T](value: T | None) -> T:
    if value is None:
        raise AssertionError("expected a value")
    return value

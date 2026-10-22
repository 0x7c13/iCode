# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The hook runs since the process started, across sessions, for the TUI's Hooks panel.

One log lives as long as the engine that owns it; every main-session hook
manager the engine builds writes its runs here, so session switches and
manager replacements keep them. Writes are synchronous; the manager tells its
observer after each one. Details (payload, result file, output) stay in this
process's memory only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from chrys.foundation.models.hook_runs import (
    HookResultFileStatus,
    HookRunApplied,
    HookRunDetail,
    HookRunEffect,
    HookRunIO,
    HookRunRecord,
)

if TYPE_CHECKING:
    from chrys.service.hooks.schema import HookConfig

HOOK_RUN_LOG_MAX_ENDED = 100
"""Ended runs kept; queued and running ones are always kept, and not counted."""

HOOK_RUN_LOG_MAX_DETAIL_CHARS = 16 * 1024 * 1024
"""What the kept details may hold together; the oldest ended runs' details go first."""

HOOK_RUN_PAYLOAD_MAX_CHARS = 64 * 1024
"""Payload text kept per run (payloads are ASCII JSON, so characters are bytes)."""


@dataclass(slots=True)
class _Entry:
    record: HookRunRecord
    detail: HookRunDetail


def _io_size(io: HookRunIO | None) -> int:
    if io is None:
        return 0
    return len(io.payload) + len(io.decision) + len(io.stdout) + len(io.stderr)


class HookRunLog:
    """The hook runs of every session since the process started; the latest ended ones only."""

    def __init__(
        self,
        *,
        max_ended: int = HOOK_RUN_LOG_MAX_ENDED,
        max_detail_chars: int = HOOK_RUN_LOG_MAX_DETAIL_CHARS,
    ) -> None:
        self._max_ended = max_ended
        self._max_detail_chars = max_detail_chars
        self._entries: dict[str, _Entry] = {}
        self._ended: list[str] = []
        """Ended run ids, oldest end first."""
        self._detail_chars = 0
        self._registered = 0
        """Runs registered so far: the next run's number is one more."""
        self._dropped = 0
        """Ended runs dropped past the cap."""

    # ── reads ────────────────────────────────────────────────────────

    def records(self) -> tuple[HookRunRecord, ...]:
        """The runs kept, oldest first, by number: a run keeps its place as it starts and ends."""
        # Registered in number order, and a dict keeps that order.
        return tuple(entry.record for entry in self._entries.values())

    @property
    def dropped(self) -> int:
        """Ended runs no longer kept because more than the cap ended since."""
        return self._dropped

    def record(self, run_id: str) -> HookRunRecord | None:
        entry = self._entries.get(run_id)
        return entry.record if entry is not None else None

    def detail(self, run_id: str) -> HookRunDetail | None:
        entry = self._entries.get(run_id)
        return entry.detail if entry is not None else None

    # ── writes ───────────────────────────────────────────────────────

    def register(
        self,
        *,
        run_id: str,
        session_id: str,
        hook: HookConfig,
        source: str,
        source_path: str,
        payload_text: str,
        tool_name: str,
        tool_call_id: str,
        sub_agent: str,
        retry: bool,
    ) -> None:
        """Add a run queued for a slot."""
        now = time.monotonic()
        self._registered += 1
        record = HookRunRecord(
            run_id=run_id,
            session_id=session_id,
            hook_id=hook.id,
            hook_event=str(hook.event),
            execution_mode=hook.execution.mode,
            detach=hook.execution.detach,
            tool_name=tool_name,
            sub_agent=sub_agent,
            retry=retry,
            registered_at=now,
            started_at=None,
            ended_at=None,
            registered_wall=time.time(),
            state="queued",
            outcome=None,
            exit_code=None,
            effect=None,
            number=self._registered,
        )
        run = hook.run
        if run.type == "script":
            argv: tuple[str, ...] = (run.path, *run.args)
        elif run.type == "command":
            argv = (*run.argv, *run.args)
        else:
            argv = (run.shell,)
        io = HookRunIO(
            payload=payload_text[:HOOK_RUN_PAYLOAD_MAX_CHARS],
            payload_size=len(payload_text.encode("utf-8")),
            result_file=None,
            decision="",
            stdout="",
            stderr="",
            stdout_dropped=0,
            stderr_dropped=0,
        )
        detail = HookRunDetail(
            record=record,
            tool_call_id=tool_call_id,
            timeout_seconds=hook.execution.timeout_seconds,
            on_error=hook.execution.on_error,
            delivery=hook.execution.delivery,
            source=source,
            source_path=source_path,
            run_type=run.type,
            run_argv=argv,
            run_cwd=run.cwd,
            run_env_names=tuple(run.env),
            log_path="",
            shared_log=False,
            io=io,
            applied=None,
        )
        self._entries[run_id] = _Entry(record, detail)
        self._detail_chars += _io_size(io)
        self._trim_details()

    def mark_running(self, run_id: str) -> bool:
        """Mark a queued run as holding its slot; False when it is not queued here."""
        entry = self._entries.get(run_id)
        if entry is None or entry.record.state != "queued":
            return False
        self._set_record(entry, replace(entry.record, state="running", started_at=time.monotonic()))
        return True

    def finish(
        self,
        run_id: str,
        *,
        outcome: str,
        exit_code: int | None = None,
        effect: HookRunEffect | None = None,
        applied: HookRunApplied | None = None,
        result_file: HookResultFileStatus | None = None,
        decision: str = "",
        stdout: str = "",
        stderr: str = "",
        stdout_dropped: int = 0,
        stderr_dropped: int = 0,
        log_path: str = "",
    ) -> bool:
        """End a queued or running run; False when it is not open here."""
        entry = self._entries.get(run_id)
        if entry is None or entry.record.state == "ended":
            return False
        record = replace(
            entry.record,
            state="ended",
            ended_at=time.monotonic(),
            outcome=outcome,
            exit_code=exit_code,
            effect=effect,
        )
        io = entry.detail.io
        if io is not None:
            new_io = replace(
                io,
                result_file=result_file,
                decision=decision,
                stdout=stdout,
                stderr=stderr,
                stdout_dropped=stdout_dropped,
                stderr_dropped=stderr_dropped,
            )
            self._detail_chars += _io_size(new_io) - _io_size(io)
            io = new_io
        shared = bool(log_path) and any(
            other.detail.log_path == log_path for key, other in self._entries.items() if key != run_id
        )
        if shared:
            for key, other in self._entries.items():
                if key != run_id and other.detail.log_path == log_path and not other.detail.shared_log:
                    other.detail = replace(other.detail, shared_log=True)
        entry.record = record
        entry.detail = replace(
            entry.detail,
            record=record,
            io=io,
            applied=applied if applied is not None else HookRunApplied(),
            log_path=log_path,
            shared_log=shared,
        )
        self._ended.append(run_id)
        self._trim_ended()
        self._trim_details()
        return True

    def _set_record(self, entry: _Entry, record: HookRunRecord) -> None:
        entry.record = record
        entry.detail = replace(entry.detail, record=record)

    def _trim_ended(self) -> None:
        while len(self._ended) > self._max_ended:
            entry = self._entries.pop(self._ended.pop(0))
            self._detail_chars -= _io_size(entry.detail.io)
            self._dropped += 1

    def _trim_details(self) -> None:
        """Drop the oldest ended runs' details until the rest fit; queued and running runs keep theirs."""
        if self._detail_chars <= self._max_detail_chars:
            return
        for run_id in self._ended:
            entry = self._entries[run_id]
            if entry.detail.io is None:
                continue
            self._detail_chars -= _io_size(entry.detail.io)
            entry.detail = replace(entry.detail, io=None)
            if self._detail_chars <= self._max_detail_chars:
                return

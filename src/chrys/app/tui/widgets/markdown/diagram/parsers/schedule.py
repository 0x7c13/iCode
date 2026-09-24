# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Bounded parsers for calendar-day Gantt, GitGraph, and packet diagrams."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from chrys.app.tui.widgets.markdown.diagram.model import DiagnosticCode, DiagramIR, DiagramKind, Direction
from chrys.app.tui.widgets.markdown.diagram.specs.schedule import (
    GanttChart,
    GanttTask,
    GitChart,
    GitCommit,
    PacketChart,
    PacketField,
)

from .common import MAX_NODES, _Builder, _clean_label, _source_lines

_UNSUPPORTED = DiagnosticCode.UNSUPPORTED_CHART_STATEMENT
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_DURATION_RE = re.compile(r"(\d{1,6})([dw])")
_QUOTED = r'"(?:\\.|[^"\\])*"'
_GIT_OPTION_RE = re.compile(rf"(id|tag|type)\s*:\s*({_QUOTED}|[^\s]+)")
_PACKET_RE = re.compile(rf"(?:(\d{{1,6}})(?:\s*-\s*(\d{{1,6}}))?|\+(\d{{1,6}}))\s*:\s*({_QUOTED})")
_MAX_PACKET_BITS = 16_384


def _content_lines(source: str) -> list[tuple[int, str]]:
    """Locate headers after comments, retaining physical diagnostic line numbers."""
    return [(line, statement) for line, statement in _source_lines(source) if not statement.startswith("%%")]


@dataclass(frozen=True, slots=True)
class _TaskDraft:
    task_id: str
    label: str
    section: str
    start: str
    end: str
    statuses: tuple[str, ...]
    previous: str | None
    line: int


def _calendar_date(value: str) -> date | None:
    if not _DATE_RE.fullmatch(value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _task_references(draft: _TaskDraft) -> tuple[str, ...]:
    references = draft.start[6:].split() if draft.start.startswith("after ") else []
    if not draft.start and draft.previous:
        references.append(draft.previous)
    if draft.end.startswith("until "):
        references.append(draft.end[6:].strip())
    return tuple(references)


def _resolve_task(draft: _TaskDraft, resolved: dict[str, GanttTask]) -> GanttTask | None:
    if draft.start.startswith("after "):
        starts = [resolved[reference].end for reference in draft.start[6:].split()]
        start = max(starts) if starts else None
    elif not draft.start:
        start = resolved[draft.previous].end if draft.previous else None
    else:
        start = _calendar_date(draft.start)
    if start is None:
        return None
    if draft.end.startswith("until "):
        end = resolved[draft.end[6:].strip()].start
    elif duration := _DURATION_RE.fullmatch(draft.end):
        days = int(duration[1]) * (7 if duration[2] == "w" else 1)
        try:
            end = start + timedelta(days=days)
        except OverflowError:
            return None
    else:
        end = _calendar_date(draft.end)
    if end is None or end < start or (end == start and "milestone" not in draft.statuses):
        return None
    return GanttTask(draft.task_id, draft.label, draft.section, start, end, draft.statuses)


def parse_gantt(source: str) -> DiagramIR:
    """Resolve explicit ISO dates and day/week durations without wall-clock defaults."""
    builder = _Builder(DiagramKind.GANTT, Direction.LEFT_RIGHT)
    title = section = ""
    drafts: list[_TaskDraft] = []
    ids: set[str] = set()
    lines = _content_lines(source)
    if not lines or lines[0][1] != "gantt":
        builder.error(lines[0][0] if lines else 1, _UNSUPPORTED)
    for line, statement in lines[1:]:
        if statement.startswith("title "):
            title = _clean_label(statement[6:])
            continue
        if statement.startswith("section "):
            section = _clean_label(statement[8:])
            continue
        if re.fullmatch(r"dateFormat\s+YYYY-MM-DD|todayMarker\s+off", statement):
            continue
        # Gantt task text ends at the first colon; quotes are literal text.
        delimiter = statement.find(":")
        if delimiter < 1:
            builder.error(line, _UNSUPPORTED)
            continue
        label = _clean_label(statement[:delimiter], quote_chars="")
        tokens = [value.strip() for value in statement[delimiter + 1 :].split(",")]
        statuses: list[str] = []
        while tokens and tokens[0] in {"active", "done", "crit", "milestone"}:
            statuses.append(tokens.pop(0))
        if not label or not 1 <= len(tokens) <= 3 or not all(tokens):
            builder.error(line, _UNSUPPORTED)
            continue
        task_id = tokens[0] if len(tokens) == 3 else f"@task:{len(drafts)}"
        if task_id in ids or (len(tokens) == 3 and not re.fullmatch(r"[\w.-]+", task_id)):
            builder.error(line, _UNSUPPORTED)
            continue
        if len(drafts) >= MAX_NODES:
            builder.error(line, DiagnosticCode.NODE_LIMIT, limit=MAX_NODES)
            break
        ids.add(task_id)
        drafts.append(
            _TaskDraft(
                task_id,
                label,
                section,
                tokens[-2] if len(tokens) > 1 else "",
                tokens[-1],
                tuple(statuses),
                drafts[-1].task_id if drafts else None,
                line,
            )
        )
    resolved: dict[str, GanttTask] = {}
    pending = list(drafts)
    while pending:
        remaining: list[_TaskDraft] = []
        for draft in pending:
            references = _task_references(draft)
            if any(reference not in ids for reference in references):
                builder.error(draft.line, _UNSUPPORTED)
                continue
            if any(reference not in resolved for reference in references):
                remaining.append(draft)
                continue
            task = _resolve_task(draft, resolved)
            if task is None:
                builder.error(draft.line, _UNSUPPORTED)
            else:
                resolved[draft.task_id] = task
        if len(remaining) == len(pending):
            for draft in remaining:
                builder.error(draft.line, _UNSUPPORTED)
            break
        pending = remaining
    if not drafts:
        builder.error(1, _UNSUPPORTED)
    tasks = tuple(resolved[draft.task_id] for draft in drafts if draft.task_id in resolved)
    return builder.finish(GanttChart(title, tasks))


def _git_options(raw: str) -> dict[str, str] | None:
    options: dict[str, str] = {}
    position = 0
    while position < len(raw):
        match = _GIT_OPTION_RE.match(raw, position)
        if match is None or match[1] in options:
            return None
        options[match[1]] = _clean_label(match[2])
        position = match.end()
        if position < len(raw) and not raw[position].isspace():
            return None
        while position < len(raw) and raw[position].isspace():
            position += 1
    return options


def parse_git(source: str) -> DiagramIR:
    """Track real branch heads and merge parents for the common GitGraph commands."""
    builder = _Builder(DiagramKind.GIT, Direction.LEFT_RIGHT)
    lines = _content_lines(source)
    header = re.fullmatch(r"gitGraph(?:\s+(LR|TB|BT))?\s*:?", lines[0][1]) if lines else None
    orientation = (header[1] or "LR") if header else "LR"
    if header is None:
        builder.error(lines[0][0] if lines else 1, _UNSUPPORTED)
    branches: dict[str, int | None] = {"main": None}
    current = "main"
    commits: list[GitCommit] = []
    explicit_ids: set[str] = set()
    for line, statement in lines[1:]:
        parts = statement.split(maxsplit=1)
        command = parts[0]
        arguments = parts[1] if len(parts) > 1 else ""
        if command in {"branch", "checkout", "switch", "merge"}:
            name_match = re.match(rf"({_QUOTED}|[^\s]+)(?:\s+|$)", arguments)
            if name_match is None:
                builder.error(line, _UNSUPPORTED)
                continue
            name = _clean_label(name_match[1])
            rest = arguments[name_match.end() :].strip()
            if not name or (rest and command != "merge"):
                builder.error(line, _UNSUPPORTED)
                continue
            if command == "branch":
                if name in branches:
                    builder.error(line, _UNSUPPORTED)
                elif len(branches) >= MAX_NODES:
                    builder.error(line, DiagnosticCode.NODE_LIMIT, limit=MAX_NODES)
                else:
                    branches[name] = branches[current]
                    current = name
                continue
            if name not in branches:
                builder.error(line, _UNSUPPORTED)
                continue
            if command in {"checkout", "switch"}:
                current = name
                continue
            if name == current or branches[name] is None or branches[name] == branches[current]:
                builder.error(line, _UNSUPPORTED)
                continue
            options = _git_options(rest)
            parents = tuple(parent for parent in (branches[current], branches[name]) if parent is not None)
        elif command == "commit":
            options = _git_options(arguments)
            head = branches[current]
            parents = (head,) if head is not None else ()
        else:
            builder.error(line, _UNSUPPORTED)
            continue
        if options is None or options.get("type", "NORMAL") not in {"NORMAL", "REVERSE", "HIGHLIGHT"}:
            builder.error(line, _UNSUPPORTED)
            continue
        if "id" in options and (not options["id"] or options["id"] in explicit_ids):
            builder.error(line, _UNSUPPORTED)
            continue
        if len(commits) >= MAX_NODES:
            builder.error(line, DiagnosticCode.NODE_LIMIT, limit=MAX_NODES)
            break
        if "id" in options:
            explicit_ids.add(options["id"])
        commits.append(
            GitCommit(
                options.get("id", f"#{len(commits) + 1}"),
                current,
                parents,
                (options["tag"],) if "tag" in options else (),
                options.get("type", "NORMAL"),
            )
        )
        branches[current] = len(commits) - 1
    if not commits:
        builder.error(1, _UNSUPPORTED)
    return builder.finish(GitChart(tuple(branches), tuple(commits), orientation))


def parse_packet(source: str) -> DiagramIR:
    """Retain inclusive absolute offsets, including gaps and row-spanning fields."""
    builder = _Builder(DiagramKind.PACKET, Direction.LEFT_RIGHT)
    fields: list[PacketField] = []
    title = ""
    next_bit = 0
    lines = _content_lines(source)
    if not lines or lines[0][1] not in {"packet", "packet-beta"}:
        builder.error(lines[0][0] if lines else 1, _UNSUPPORTED)
    for line, statement in lines[1:]:
        if statement.startswith("title "):
            title = _clean_label(statement[6:])
            continue
        match = _PACKET_RE.fullmatch(statement)
        if match is None:
            builder.error(line, _UNSUPPORTED)
            continue
        start = next_bit if match[3] is not None else int(match[1])
        end = start + int(match[3]) - 1 if match[3] is not None else int(match[2] or match[1])
        if start < next_bit or end < start:
            builder.error(line, _UNSUPPORTED)
            continue
        if end >= _MAX_PACKET_BITS:
            builder.error(line, DiagnosticCode.CANVAS_LIMIT)
            break
        if len(fields) >= MAX_NODES:
            builder.error(line, DiagnosticCode.NODE_LIMIT, limit=MAX_NODES)
            break
        fields.append(PacketField(start, end, _clean_label(match[4])))
        next_bit = end + 1
    if not fields:
        builder.error(1, _UNSUPPORTED)
    return builder.finish(PacketChart(title, tuple(fields)))

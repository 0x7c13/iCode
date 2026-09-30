# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Golden scenarios that pin what the reminder middleware sends and records.

Each scenario drives a reminder stack (``tests/support/reminder_stack.py``)
through turns, retries, folds and restarts, and records every call as the
model saw it: each message's texts in order, which messages kept their
identity, the reminder records the call wrote and the stack's observable
state.  ``tests/service/agent_middleware/test_reminder_goldens.py`` replays
every scenario against its file in ``reminder_goldens/``.

The files are the baseline a refactor must reproduce byte for byte; they are
written once, reviewed by hand and never regenerated to make a change pass.
The module compares every scenario with its file (a diff per difference,
exit status 1) and writes only the scenarios named after ``--write``::

    uv run python -m tests.support.reminder_goldens
    uv run python -m tests.support.reminder_goldens --write g17_new_scenario

Inputs that differ by machine are pinned (``pin_reminder_inputs``): the
clock, Python-path discovery and the runtime environment.  The two session
roots are real directories (the archive pointer needs a real catalog file),
named ``{root_a}`` and ``{root_b}`` in the files and filled in on the
expected side by exact replacement.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import difflib
import itertools
import json
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import pytest

from chrys.foundation.models.history_markers import HistoryMarkerKind, copy_reminder_record
from chrys.foundation.models.session_env import SessionEnvironment
from chrys.foundation.models.workspace import WorkingDir
from chrys.foundation.platform import PlatformInfo, ShellInfo
from chrys.kernel import EXCLUDED_KEY, ChatResponse, ChatResponseUpdate, Content, Message, ResponseStream
from chrys.kernel.middleware import ChatContext
from chrys.service.agent_middleware.reminders import runtime_env, turn_line
from chrys.service.agent_middleware.reminders.archive_pointer import CATALOG_POINTER_RECORD_COUNT_STATE_KEY
from chrys.service.context.compaction.last_words_state import DropRoundBreakerState, ManifestEntry
from chrys.service.context.compaction.spill import (
    CATALOG_RELATIVE_PATH,
    SpillQuota,
    build_record_filename,
    dropped_turn_relative_path,
)
from tests.support.reminder_calls import establish_request
from tests.support.reminder_stack import ReminderStack, make_reminder_stack, observe, restore_phase4
from tests.support.secure_files import plant_owner_only_bytes

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Sequence
    from datetime import tzinfo

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "service" / "agent_middleware" / "reminder_goldens"

_RECORD = HistoryMarkerKind.SYSTEM_REMINDERS_KEY
MAX_CONTEXT_TOKENS = 200_000
_CLOCK_START = datetime(2026, 9, 30, 1, 2, 3, tzinfo=UTC)
_CLOCK_ZONE = timezone(timedelta(hours=8), "CST")
_PYTHON_PATHS = (
    ("system uv", "/opt/golden/bin/uv", ""),
    ("system Python", "/usr/bin/python3", ""),
    ("your runtime Python", "/opt/golden/runtime/bin/python", "3.14.0"),
)

RUNTIME = SessionEnvironment(
    cwd="/work/golden",
    platform=PlatformInfo(
        os_name="linux",
        os_version="24.04",
        arch="amd64",
        shell=ShellInfo(name="bash", path="/usr/bin/bash", args=["-c"], version="5.2.21"),
        config_dir=Path("/home/golden/.chrys"),
        data_dir=Path("/home/golden/.chrys"),
        extra_shells=(ShellInfo(name="zsh", path="/usr/bin/zsh", args=["-c"], version="5.9"),),
    ),
    session_id="golden",
    created_at=_CLOCK_START,
    working_dirs=(WorkingDir("/work/golden", "app", is_primary=True), WorkingDir("/work/shared")),
)
SUB_AGENTS = ["Explore", "General"]
TOOL_NAMES = ["read_file", "shell", "read_file", " ", "todo_write"]
TODO_A = "Current todo list:\n- [ ] pin the reminder goldens"
TODO_B = "Current todo list:\n- [x] pin the reminder goldens\n- [ ] run the gates"
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
    """Pin the clock and Python-path discovery where the reminder pipeline reads them.

    The clock advances one minute per read, so each turn line differs.  When
    *log* is given, each read appends ``"clock"`` or ``"python_paths"``.
    """
    ticks = itertools.count()

    class _GoldenDateTime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> _GoldenDateTime:
            if log is not None:
                log.append("clock")
            moment = _CLOCK_START + timedelta(minutes=next(ticks))
            return cls(moment.year, moment.month, moment.day, moment.hour, moment.minute, moment.second, tzinfo=tz)

        def astimezone(self, tz: tzinfo | None = None) -> datetime:
            return datetime.astimezone(self, _CLOCK_ZONE if tz is None else tz)

    def _python_execution_paths() -> list[Any]:
        if log is not None:
            log.append("python_paths")
        return [
            runtime_env._PythonExecutionPath(label=label, path=path, version=version)
            for label, path, version in _PYTHON_PATHS
        ]

    monkeypatch.setattr(turn_line, "datetime", _GoldenDateTime)
    monkeypatch.setattr(runtime_env, "_python_execution_paths", _python_execution_paths)
    monkeypatch.setattr(runtime_env, "_runtime_python_uses_process_safe_alias", lambda: False)


@dataclass
class GoldenInputs:
    """What the providers return; a scenario changes a field between turns."""

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


def spill_quota(*, live: Sequence[str] = (), available: Sequence[str] = ()) -> SpillQuota:
    """A quota whose catalog lists *live* records, *available* of them on disk."""
    quota = SpillQuota()
    quota.initialize(0, available, live_relative_paths=list(live))
    return quota


def plant_catalog(session_root: Path) -> None:
    """Create the session's archive catalog, so the pointer can name it."""
    catalog = session_root / CATALOG_RELATIVE_PATH
    catalog.parent.mkdir(parents=True, exist_ok=True)
    plant_owner_only_bytes(catalog, b"")


def manifest_entry(
    turn: int,
    sequence: int,
    tool: str,
    argument: str,
    *,
    size: int = 1_200,
    outcome: str = "ok",
    assistant_text: bool = False,
    no_record_reason: str = "",
) -> ManifestEntry:
    record_id = "" if no_record_reason else f"{turn:04x}{sequence:04x}"
    return ManifestEntry(
        record_id=record_id,
        group_id=f"group_{turn}_{sequence}",
        record_dir=dropped_turn_relative_path(turn).as_posix(),
        relative_path="" if no_record_reason else _record_path(turn, sequence, tool, record_id),
        turn=turn,
        round=1,
        sequence=sequence,
        tool=tool,
        display_argument=argument,
        outcome=outcome,
        size_chars=size,
        assistant_text=assistant_text,
        no_record_reason=no_record_reason,
    )


def usage(percent: int) -> dict[str, int]:
    return {"total_token_count": MAX_CONTEXT_TOKENS * percent // 100}


def user(text: str, **properties: Any) -> Message:
    message = Message(role="user", contents=[Content.from_text(text)])
    message.additional_properties.update(properties)
    return message


def injected(text: str) -> Message:
    return user(text, **{HistoryMarkerKind.INJECTED_KEY: True})


def assistant(text: str) -> Message:
    return Message(role="assistant", contents=[Content.from_text(text)])


def tool_call(call_id: str, name: str, arguments: str) -> Message:
    return Message(role="assistant", contents=[Content.from_function_call(call_id, name, arguments=arguments)])


def tool_result(call_id: str, result: str) -> Message:
    return Message(role="tool", contents=[Content.from_function_result(call_id, result=result)])


def require[T](value: T | None) -> T:
    if value is None:
        raise AssertionError("expected a value")
    return value


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class _CallFailed(Exception):
    """The provider failed after the call's requests (if any) went out."""


async def _no_updates() -> AsyncIterator[ChatResponseUpdate]:
    return
    yield


def _content_text(content: Content) -> str:
    if content.type == "text" and content.text is not None:
        return content.text
    if content.type == "function_call":
        return f"<function_call {content.name}({content.arguments})>"
    if content.type == "function_result":
        return f"<function_result {content.result}>"
    return f"<{content.type}>"


def _record_of(message: Message) -> Any:
    record = message.additional_properties.get(_RECORD)
    return copy.deepcopy(record) if record is not None else None


@dataclass
class GoldenRoots:
    a: Path
    b: Path

    @classmethod
    def under(cls, base: Path) -> GoldenRoots:
        roots = cls(a=(base / "root-a").resolve(), b=(base / "root-b").resolve())
        for root in (roots.a, roots.b):
            plant_catalog(root)
        return roots

    def placeholders(self) -> dict[str, str]:
        return {"{root_a}": self.a.as_posix(), "{root_b}": self.b.as_posix()}


@dataclass
class GoldenRun:
    """One scenario's stacks and the steps it recorded."""

    roots: GoldenRoots
    steps: list[dict[str, Any]] = field(default_factory=list)

    def stack(
        self,
        inputs: GoldenInputs,
        *,
        root: Literal["a", "b"] | None = None,
        quota: SpillQuota | None = None,
        runtime: SessionEnvironment | None = RUNTIME,
        sub_agents: list[str] | None = SUB_AGENTS,
        shell: bool = False,
        tool_names: list[str] | None = None,
        file_read: bool = True,
        pointer: bool = True,
    ) -> ReminderStack:
        session_root = {"a": self.roots.a, "b": self.roots.b, None: None}[root]
        return make_reminder_stack(
            runtime,
            max_context_tokens=MAX_CONTEXT_TOKENS,
            sub_agent_names=sub_agents,
            shell_tool_enabled=shell,
            tool_names=TOOL_NAMES if tool_names is None else tool_names,
            session_root=session_root,
            file_read_available=file_read,
            spill_quota=quota,
            catalog_pointer_enabled=pointer,
            skill_catalog_provider=inputs.read_skills,
            todo_state_provider=inputs.read_todo,
            mcp_instructions_provider=inputs.read_mcp,
            file_change_provider=inputs.drain_file_change,
        )

    def note(self, label: str, **values: Any) -> None:
        self.steps.append({"step": label, **values})

    async def call(
        self,
        label: str,
        stack: ReminderStack,
        messages: list[Message],
        *,
        requests: int = 1,
        fail: bool = False,
        handle: str | None = None,
        answer: str | None = None,
        stream: bool = False,
        poll: bool = False,
        prepare: Callable[[list[Message]], Awaitable[list[Message]]] | None = None,
    ) -> list[Message]:
        """Run one call and record what it sent and wrote.

        *requests* provider requests go out (0: none did); *fail* raises after
        them.  *handle* continues a service-side conversation, *answer* is the
        handle the answer returns, *poll* retrieves a response already
        created, and *prepare* runs where the client compacts.
        """
        options: dict[str, Any] = {}
        if handle is not None:
            options["conversation_id"] = handle
        if poll:
            options["continuation_token"] = "poll_token"
        before = [_record_of(message) for message in messages]
        context = ChatContext(client=None, messages=list(messages), options=options, stream=stream)
        response = ChatResponse(messages=[], conversation_id=answer)

        async def _call_next() -> None:
            for _ in range(requests):
                await establish_request(context, prepare=prepare)
            if fail:
                raise _CallFailed
            if stream:
                context.result = ResponseStream(_no_updates(), finalizer=lambda _updates: response)
            else:
                context.result = response

        failed = False
        try:
            await stack.middleware.process(context, _call_next)
        except _CallFailed:
            failed = True
        result = context.result
        if isinstance(result, ResponseStream):
            for hook in context.stream_result_hooks:
                result.with_result_hook(hook)
            await result.get_final_response()
        sent = cast("list[Message]", context.messages)
        step: dict[str, Any] = {"step": label}
        if failed:
            step["failed"] = True
        step["call"] = [_message_view(message, original) for message, original in zip(sent, messages, strict=True)]
        step["recorded"] = {
            str(index): _record_of(message)
            for index, message in enumerate(messages)
            if _record_of(message) != before[index]
        }
        step["state"] = observe(stack)
        self.steps.append(step)
        return sent


def _message_view(message: Message, original: Message) -> dict[str, Any]:
    view: dict[str, Any] = {"role": message.role, "texts": [_content_text(content) for content in message.contents]}
    if message.additional_properties.get(EXCLUDED_KEY, False):
        view["excluded"] = True
    if message is original:
        view["kept"] = True
    return view


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


async def g01_three_turns(run: GoldenRun) -> None:
    """Turn 1 carries every catalog; turn 2 changes the todo list and loses MCP; turn 3 changes both back."""
    inputs = GoldenInputs(todo=TODO_A, skills=SKILLS, mcp=MCP)
    stack = run.stack(inputs, root="a", quota=spill_quota(live=ARCHIVED[:2]), shell=True)
    mw = stack.middleware

    mw.prepare_turn()
    u1 = user("Pin the reminder goldens.")
    await run.call("turn 1: the first call carries the turn line and every catalog", stack, [u1])
    a1 = tool_call("call_1", "read_file", '{"path": "src/app.py"}')
    t1 = tool_result("call_1", "print('golden')")
    await run.call("turn 1: the tool-loop call adds nothing", stack, [u1, a1, t1])
    f1 = assistant("Pinned.")

    inputs.todo, inputs.mcp = TODO_B, None
    mw.prepare_turn(usage=usage(20))
    u2 = user("Now run the gates.")
    await run.call("turn 2: usage line, todo B, then the MCP withdrawal", stack, [u1, a1, t1, f1, u2])
    f2 = assistant("Gates run.")

    inputs.todo, inputs.mcp = TODO_A, MCP
    mw.prepare_turn(usage=usage(30))
    u3 = user("Back to the first list.")
    await run.call("turn 3: todo A and MCP go again", stack, [u1, a1, t1, f1, u2, f2, u3])


async def g02a_full_order(run: GoldenRun) -> None:
    """One call carries every kind: turn line, warning, switch, the event inbox, file change, catalogs, LAST_WORDS."""
    inputs = GoldenInputs(
        todo=TODO_A,
        skills=SKILLS,
        mcp=MCP,
        file_change="Workspace files changed since the last turn: src/app.py",
    )
    stack = run.stack(inputs, root="a", quota=spill_quota(live=ARCHIVED[:1]))
    mw = stack.middleware

    stack.switch.set_profile_switch("Code", "QA")
    mw.queue_hook_reminders(["global hook: queued for the next turn"], for_next_turn=True)
    scope = mw.create_current_run_scope()
    before_prepare = require(mw.capture_current_run_target(scope))
    mw.queue_hook_reminders_for_current_run(before_prepare, ["scoped hook: queued before prepare"])
    mw.prepare_turn(reminder_scope=scope, usage=usage(75))
    mw.queue_hook_reminders(["mid-turn hook: first attempt"])
    u1 = user("Ship the release.")
    await run.call("attempt 1: fails before any request", stack, [u1], requests=0, fail=True)

    notice = mw.take_undelivered_file_change()
    run.note("attempt 1: the undelivered file change goes back to the tracker", file_change=notice)
    inputs.file_change = notice
    mw.queue_hook_reminders(["global hook: queued for the retry"], for_next_turn=True)
    mw.prepare_turn(reminder_scope=scope, usage=usage(75), preserve_turn_reminders=True, preserve_last_words=True)
    mw.queue_hook_reminders(["mid-turn hook: retry"])
    mw.queue_drained_injection_reminders(["drained injection reminder"])
    stack.last_words.set_last_words("Progress: the build is done; the release notes remain.")
    await run.call("attempt 2: the full send order", stack, [u1])


async def g02b_withdrawals(run: GoldenRun) -> None:
    """Catalogs no longer offered are withdrawn after the unchanged ones, before LAST_WORDS."""
    inputs = GoldenInputs(todo=TODO_A, skills=SKILLS, mcp=MCP)
    quota = spill_quota(live=ARCHIVED[:2])
    first = run.stack(inputs, root="a", quota=quota)
    first.middleware.prepare_turn()
    u1 = user("First turn.")
    await run.call("turn 1: all six catalogs", first, [u1])
    f1 = assistant("Done.")

    # A profile without sub-agents rebuilds the stack; the MCP servers went away.
    rebuilt = run.stack(inputs, root="a", quota=quota, sub_agents=None)
    inputs.mcp = None
    rebuilt.middleware.prepare_turn()
    rebuilt.last_words.set_last_words("Progress: compared the two builds.")
    u2 = user("Second turn.")
    await run.call("turn 2: turn line, two withdrawals, LAST_WORDS", rebuilt, [u1, f1, u2])


async def g03_retry(run: GoldenRun) -> None:
    """A preserving retry replays the record and adds only a re-drained change; a fresh prepare starts a turn."""
    inputs = GoldenInputs(todo=TODO_A, skills=SKILLS, file_change="Workspace files changed: src/a.py")
    stack = run.stack(inputs)
    mw = stack.middleware

    mw.prepare_turn(usage=usage(10))
    u1 = user("Fix the parser.")
    await run.call("attempt 1: fails after its request went out", stack, [u1], fail=True)
    run.note("attempt 1: the delivered file change is not requeued", file_change=mw.take_undelivered_file_change())

    inputs.file_change = "Workspace files changed: src/b.py"
    mw.prepare_turn(usage=usage(10), preserve_turn_reminders=True, preserve_last_words=True)
    await run.call("attempt 2: the record replays; only the new change is added", stack, [u1])

    # Manual Retry replays the opener as a copy carrying its record.
    replay = user("Fix the parser.")
    copy_reminder_record(u1.additional_properties, replay.additional_properties)
    mw.prepare_turn(usage=usage(12))
    await run.call("manual retry: the replayed record already started the turn", stack, [replay])
    f1 = assistant("Fixed.")

    mw.prepare_turn(usage=usage(15))
    u2 = user("Next.")
    await run.call("next turn: a new turn line", stack, [replay, f1, u2])


async def g04_fold(run: GoldenRun) -> None:
    """Folds: catalogs return before LAST_WORDS; a folded current-turn carrier re-sends turn, event and switch."""
    inputs = GoldenInputs(todo=TODO_A, skills=SKILLS, mcp=MCP)
    stack = run.stack(inputs)
    mw = stack.middleware

    mw.prepare_turn()
    u1 = user("Read the code.")
    await run.call("turn 1: catalogs on the opener", stack, [u1])
    f1 = assistant("Read.")

    mw.prepare_turn(usage=usage(40))
    stack.last_words.set_last_words("Progress: read three files.")
    u2 = user("Summarise it.")

    async def fold_turn_one(messages: list[Message]) -> list[Message]:
        u1.additional_properties[EXCLUDED_KEY] = True
        f1.additional_properties[EXCLUDED_KEY] = True
        mw.restore_folded_reminders(messages)
        return messages

    await run.call(
        "turn 2: a fold drops turn 1; its catalogs return before LAST_WORDS",
        stack,
        [u1, f1, u2],
        prepare=fold_turn_one,
    )
    a2 = tool_call("call_2", "read_file", '{"path": "src/b.py"}')
    t2 = tool_result("call_2", "b")
    await run.call("turn 2: the next call renders them from the record", stack, [u1, f1, u2, a2, t2])
    f2 = assistant("Summarised.")

    stack.switch.set_profile_switch("Code", "QA")
    mw.queue_hook_reminders(["hook: before turn 3"], for_next_turn=True)
    mw.prepare_turn(usage=usage(45))
    u3 = user("Review it.")
    history = [u1, f1, u2, a2, t2, f2, u3]
    await run.call("turn 3: turn line, switch and hook", stack, history)
    a3 = tool_call("call_3", "read_file", '{"path": "src/c.py"}')
    t3 = tool_result("call_3", "c")
    i3 = injected("Check the tests too.")
    u3.additional_properties[EXCLUDED_KEY] = True
    await run.call(
        "turn 3: the carrier was folded; the injected message carries them again",
        stack,
        [*history, a3, t3, i3],
    )


async def g05_service_side(run: GoldenRun) -> None:
    """A continued service-side conversation holds its catalogs; other handles, failures and polls do not."""
    inputs = GoldenInputs(todo=TODO_A, skills=SKILLS)
    stack = run.stack(inputs)
    mw = stack.middleware

    mw.prepare_turn()
    await run.call("first call, answered as resp_1", stack, [user("first")], answer="resp_1")
    mw.prepare_turn()
    await run.call("continuing resp_1: catalogs held", stack, [user("second")], handle="resp_1", answer="resp_2")
    mw.prepare_turn()
    await run.call("another handle: catalogs sent", stack, [user("third")], handle="other", answer="resp_3")
    mw.prepare_turn()
    await run.call("failed call: nothing held", stack, [user("fourth")], handle="resp_3", fail=True)
    mw.prepare_turn()
    await run.call("after a failure: catalogs sent", stack, [user("fifth")], handle="resp_3", answer="resp_5")
    mw.prepare_turn()
    await run.call(
        "streamed answer: held once it finalizes",
        stack,
        [user("sixth")],
        handle="resp_5",
        answer="resp_6",
        stream=True,
    )
    mw.prepare_turn()
    seventh = user("seventh")
    await run.call("continuation poll: adds and records nothing", stack, [seventh], handle="resp_6", poll=True)
    await run.call("after the poll: resp_6 still held", stack, [seventh], handle="resp_6", answer="resp_7")


async def g06_switches(run: GoldenRun) -> None:
    """Code → QA → Code → QA across retries of one turn: each switch notice goes out."""
    inputs = GoldenInputs()
    code_tools = ["read_file", "shell", "read_file", " "]
    qa_tools = ["read_file"]
    code = run.stack(inputs, sub_agents=None, tool_names=code_tools)
    code.middleware.prepare_turn()
    u1 = user("First turn.")
    await run.call("Code: first turn", code, [u1])
    f1 = assistant("Done.")
    u2 = user("Second turn.")

    qa = run.stack(inputs, sub_agents=None, tool_names=qa_tools)
    qa.switch.set_profile_switch("Code", "QA")
    qa.middleware.prepare_turn()
    await run.call("QA: the switch notice; the call fails", qa, [u1, f1, u2], fail=True)

    back = run.stack(inputs, sub_agents=None, tool_names=code_tools)
    back.switch.set_profile_switch("QA", "Code")
    back.middleware.prepare_turn(preserve_turn_reminders=True, preserve_last_words=True)
    await run.call("retry as Code: the QA → Code notice; the call fails", back, [u1, f1, u2], fail=True)

    again = run.stack(inputs, sub_agents=None, tool_names=qa_tools)
    again.switch.set_profile_switch("Code", "QA")
    again.middleware.prepare_turn(preserve_turn_reminders=True, preserve_last_words=True)
    await run.call("retry as QA: Code → QA goes again", again, [u1, f1, u2])


async def g07_injection(run: GoldenRun) -> None:
    """Scoped skill refreshes and drained injection reminders land on the injected message."""
    inputs = GoldenInputs(skills=SKILLS)
    stack = run.stack(inputs, sub_agents=None)
    mw = stack.middleware

    scope = mw.create_current_run_scope()
    mw.prepare_turn(reminder_scope=scope)
    u1 = user("Review the change.")
    await run.call("turn start", stack, [u1])
    target = require(mw.capture_current_run_target(scope))

    inputs.skills = SKILLS_REFRESHED
    run.note("scoped skill refresh", accepted=mw.update_skill_catalog_for_current_run(target))
    a1 = tool_call("call_1", "run_skill", '{"name": "release"}')
    t1 = tool_result("call_1", "installed")
    mw.queue_drained_injection_reminders(["skill reference: release", "hook: injection accepted"])
    i1 = injected("Also check the release notes.")
    history = [u1, a1, t1, i1]
    await run.call("the injected message carries the drained reminders and the new catalog", stack, history)

    run.note("scoped skill set back", accepted=mw.set_skill_catalog_for_current_run(target, SKILLS))
    a2 = tool_call("call_2", "read_file", '{"path": "NOTES.md"}')
    t2 = tool_result("call_2", "notes")
    history += [a2, t2]
    await run.call("the catalog changed back: sent again", stack, history)

    mw.update_skill_catalog_for_active_turn()
    a3 = tool_call("call_3", "read_file", '{"path": "CHANGELOG.md"}')
    t3 = tool_result("call_3", "log")
    await run.call("active-turn refresh from the provider", stack, [*history, a3, t3])


async def g08_phase4(run: GoldenRun) -> None:
    """LAST_WORDS: set, refreshed below the middleware, evictions, and a todo list captured at set time."""
    inputs = GoldenInputs(todo=TODO_A)
    quota = spill_quota(live=ARCHIVED[:2])
    stack = run.stack(inputs, root="a", quota=quota, sub_agents=None)
    mw = stack.middleware
    lw = stack.last_words

    mw.prepare_turn()
    u1 = user("Refactor the parser.")
    await run.call("turn start: two archived records", stack, [u1])

    first = [
        manifest_entry(2, 1, "read_file", "src/parser.py"),
        manifest_entry(2, 2, "shell", "uv run pytest tests/parser", outcome="exit 1", size=18_400),
    ]
    written = [entry.relative_path for entry in first]
    quota.initialize(0, written, live_relative_paths=[*ARCHIVED[:2], *written])
    lw.append_manifest(first)
    lw.set_last_words("Progress: the grammar is parsed; the tokenizer remains.")
    inputs.todo = TODO_B
    await run.call("note and manifest; the todo list is the one captured at set time", stack, [u1])

    second = [
        manifest_entry(2, 3, "assistant", "", assistant_text=True, size=640),
        manifest_entry(2, 4, "web_fetch", "https://example.test/spec", no_record_reason="over the record cap"),
    ]

    async def phase4_below(messages: list[Message]) -> list[Message]:
        also_written = [*written, second[0].relative_path]
        quota.initialize(0, also_written, live_relative_paths=[*ARCHIVED[:2], *also_written])
        lw.append_manifest(second)
        lw.set_last_words("Progress: the tokenizer is done; emitting remains.")
        mw.refresh_last_words_reminder(messages)
        return messages

    await run.call("Phase 4 below the middleware: the block is rewritten", stack, [u1], prepare=phase4_below)

    lw.mark_manifest_records_unavailable({first[0].relative_path})
    quota.reclaim_record(first[1].relative_path)
    await run.call("evicted records show as missing", stack, [u1])

    lw.mark_manifest_records_unavailable({second[0].relative_path})
    await run.call("no readable record left: the read affordance goes", stack, [u1])


async def g09_restart(run: GoldenRun) -> None:
    """A restart restores LAST_WORDS and the pointer count; a preserving prepare renders them as before."""
    inputs = GoldenInputs(todo=TODO_A)
    entries = [manifest_entry(2, 1, "read_file", "src/app.py"), manifest_entry(2, 2, "shell", "make test")]
    written = [entry.relative_path for entry in entries]
    before_quota = spill_quota(live=ARCHIVED[:2])
    before = run.stack(inputs, root="a", quota=before_quota, sub_agents=None)
    before.middleware.prepare_turn()
    u1 = user("Build it.")
    await run.call("before the restart: turn start", before, [u1])
    before_quota.initialize(0, written, live_relative_paths=[*ARCHIVED[:2], *written])
    before.last_words.append_manifest(entries)
    before.last_words.set_last_words("Progress: configured; building next.")
    before.last_words.set_drop_round_breaker(DropRoundBreakerState(attempts=1, side_call_tokens=900))
    await run.call("before the restart: LAST_WORDS", before, [u1])
    saved = {
        "last_words": before.last_words.get_last_words(),
        "last_words_manifest": before.last_words.get_last_words_manifest(),
        "last_words_breaker": before.last_words.get_last_words_breaker_state(),
        CATALOG_POINTER_RECORD_COUNT_STATE_KEY: before.pointer.record_count_state(),
    }

    # More records were archived since; the restored turn keeps its turn-start count.
    quota = spill_quota(live=[*ARCHIVED, *written], available=written)
    after = run.stack(inputs, root="a", quota=quota, sub_agents=None)
    restore_phase4(after, saved, available_relative_paths=set(written))
    run.note("restored, before prepare", state=observe(after))
    after.middleware.prepare_turn(preserve_turn_reminders=True, preserve_last_words=True)
    await run.call("after the restart: the same request", after, [u1])


async def g10_legacy_records(run: GoldenRun) -> None:
    """Legacy records replay verbatim, malformed entries are dropped, a legacy ``name`` survives a rewrite."""
    inputs = GoldenInputs(todo=TODO_A)
    stack = run.stack(inputs, root="a", quota=spill_quota(live=ARCHIVED[:3]), sub_agents=None)
    old_pointer = (
        "Earlier context compaction archived 3 records from previous turns; "
        "catalog: /old/session/compactions/dropped/catalog.jsonl (contains each record's relative path)."
    )
    u1 = user(
        "Old turn.",
        **{
            _RECORD: [
                {"kind": "turn", "text": "[Turn Start] Local time: 2026-01-01 08:00:00 CST"},
                {"kind": "turn", "text": "[Runtime Environment]\n  Working directory: /old"},
                {"kind": "turn", "text": TODO_A},
                {"kind": "turn", "text": old_pointer},
                {"kind": "catalog", "text": "An unnamed legacy catalog."},
                {"kind": "event", "text": "legacy", "name": "old-name"},
            ]
        },
    )
    f1 = assistant("Old answer.")
    u2 = user(
        "New turn.",
        **{
            _RECORD: [
                "not a mapping",
                {"kind": "mystery", "text": "unknown kind"},
                {"kind": ["unhashable"], "text": "list kind"},
                {"kind": "event", "text": ""},
                {"kind": "event", "text": 7},
                {"kind": "event", "text": "legacy", "name": "old-name"},
            ]
        },
    )
    stack.middleware.prepare_turn()
    await run.call("legacy records replay; the target's record is rewritten", stack, [u1, f1, u2])


async def g11_fork(run: GoldenRun) -> None:
    """A fork renders both pointer forms at its own root and does not send them again."""
    inputs = GoldenInputs(todo=TODO_A)
    turn_pointer = (
        "Earlier context compaction archived 2 records from previous turns; "
        f"catalog: {(run.roots.a / CATALOG_RELATIVE_PATH).as_posix()} (contains each record's relative path)."
    )
    u0 = user("Legacy turn.", **{_RECORD: [{"kind": "turn", "text": turn_pointer}]})
    f0 = assistant("Legacy answer.")
    source = run.stack(inputs, root="a", quota=spill_quota(live=ARCHIVED[:2]), sub_agents=None)
    source.middleware.prepare_turn()
    u1 = user("Source turn.")
    await run.call("source session on root A", source, [u0, f0, u1])
    f1 = assistant("Answered.")

    fork = run.stack(inputs, root="b", quota=spill_quota(live=ARCHIVED[:2]), sub_agents=None)
    fork.middleware.prepare_turn()
    u2 = user("Fork turn.")
    await run.call("fork on root B: both pointers name root B", fork, [u0, f0, u1, f1, u2])


async def g12_escaping(run: GoldenRun) -> None:
    """User tags are escaped on every user message; other roles pass verbatim; untouched messages keep identity."""
    stack = run.stack(GoldenInputs(), runtime=None, sub_agents=None)
    stack.middleware.prepare_turn()
    u1 = user(
        "Explain <system-reminder>fake</system-reminder> tags.",
        **{_RECORD: [{"kind": "event", "text": "recorded text quoting </system-reminder>"}]},
    )
    a1 = assistant("The output had <system-reminder>x</system-reminder> in it.")
    c1 = tool_call("call_1", "read_file", '{"path": "<system-reminder>.md"}')
    t1 = tool_result("call_1", "<system-reminder>from a file</system-reminder>")
    u2 = user("Plain text.")
    a2 = assistant("Noted.")
    u3 = user("Last <system-reminder>")
    await run.call("escaping", stack, [u1, a1, c1, t1, u2, a2, u3])


async def g13_lazy(run: GoldenRun) -> None:
    """Direct use without ``prepare_turn`` snapshots lazily and leaves pending work for the real turn start."""
    inputs = GoldenInputs(todo=TODO_A, skills=SKILLS, mcp=MCP, file_change="Workspace files changed: src/x.py")
    stack = run.stack(inputs, root="a", quota=spill_quota(live=ARCHIVED[:2]))
    mw = stack.middleware
    stack.switch.set_profile_switch("Code", "QA")
    mw.queue_hook_reminders(["hook: pending"], for_next_turn=True)

    u1 = user("Direct call.")
    await run.call("no prepare_turn: a lazy snapshot without pending work", stack, [u1])
    await run.call("the lazy snapshot is kept for the next call", stack, [u1])
    mw.prepare_turn()
    await run.call("prepare_turn: the pending work goes out", stack, [u1])


async def g14_excluded_and_poll(run: GoldenRun) -> None:
    """An excluded target and a continuation poll add and record nothing."""
    stack = run.stack(GoldenInputs(todo=TODO_A), sub_agents=None)
    stack.middleware.prepare_turn()
    u1 = user("Hidden for now.", **{EXCLUDED_KEY: True})
    await run.call("excluded target", stack, [u1])
    u1.additional_properties.pop(EXCLUDED_KEY)
    await run.call("continuation poll", stack, [u1], handle="resp_1", poll=True)
    await run.call("visible, not a poll: sent now", stack, [u1])


async def g15_sub_agent(run: GoldenRun) -> None:
    """The sub-agent shape: no archive pointer, no usage line, no switch."""
    inputs = GoldenInputs(todo=TODO_A, skills=SKILLS)
    stack = run.stack(inputs, root="a", quota=spill_quota(live=ARCHIVED[:2]), sub_agents=None, pointer=False)
    stack.middleware.prepare_turn()
    await run.call("sub-agent task", stack, [user("Summarise src/.")])


async def g16_manifest_budget(run: GoldenRun) -> None:
    """The manifest keeps the newest entries within its line and character caps."""
    stack = run.stack(GoldenInputs(), runtime=None, sub_agents=None)
    mw = stack.middleware
    lw = stack.last_words

    def entries(count: int, argument: Callable[[int], str]) -> list[ManifestEntry]:
        return [manifest_entry(3, index, "read_file", argument(index)) for index in range(1, count + 1)]

    for label, manifest in (
        ("48 entries fit the line cap exactly", entries(48, lambda index: f"src/m{index}.py")),
        ("49 entries: the oldest two give way", entries(49, lambda index: f"src/m{index}.py")),
        ("long arguments: the character cap", entries(24, lambda index: f"src/{index:03d}/" + "d" * 250)),
    ):
        mw.prepare_turn()
        lw.append_manifest(manifest)
        lw.set_last_words("Progress: checking the manifest budget.")
        await run.call(label, stack, [user(label)])


SCENARIOS: dict[str, Callable[[GoldenRun], Awaitable[None]]] = {
    "g01_three_turns": g01_three_turns,
    "g02a_full_order": g02a_full_order,
    "g02b_withdrawals": g02b_withdrawals,
    "g03_retry": g03_retry,
    "g04_fold": g04_fold,
    "g05_service_side": g05_service_side,
    "g06_switches": g06_switches,
    "g07_injection": g07_injection,
    "g08_phase4": g08_phase4,
    "g09_restart": g09_restart,
    "g10_legacy_records": g10_legacy_records,
    "g11_fork": g11_fork,
    "g12_escaping": g12_escaping,
    "g13_lazy": g13_lazy,
    "g14_excluded_and_poll": g14_excluded_and_poll,
    "g15_sub_agent": g15_sub_agent,
    "g16_manifest_budget": g16_manifest_budget,
}


async def run_scenario(name: str, base: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], GoldenRoots]:
    """Run scenario *name* with session roots under *base*; return its JSON-ready result."""
    pin_reminder_inputs(monkeypatch)
    run = GoldenRun(roots=GoldenRoots.under(base))
    scenario = SCENARIOS[name]
    await scenario(run)
    result = {"scenario": name, "about": (scenario.__doc__ or "").strip(), "steps": run.steps}
    return json.loads(json.dumps(result)), run.roots


def _replace_strings(value: Any, replace: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return replace(value)
    if isinstance(value, list):
        return [_replace_strings(item, replace) for item in value]
    if isinstance(value, dict):
        return {key: _replace_strings(item, replace) for key, item in value.items()}
    return value


def load_golden(name: str, roots: GoldenRoots) -> dict[str, Any]:
    """The golden for scenario *name*, with the root placeholders filled in."""
    expected = json.loads((GOLDEN_DIR / f"{name}.json").read_text(encoding="utf-8"))

    def fill(text: str) -> str:
        for placeholder, path in roots.placeholders().items():
            text = text.replace(placeholder, path)
        return text

    return _replace_strings(expected, fill)


def _scrub(result: dict[str, Any], roots: GoldenRoots) -> dict[str, Any]:
    def scrub(text: str) -> str:
        for placeholder, path in roots.placeholders().items():
            text = text.replace(path, placeholder)
        return text

    return _replace_strings(result, scrub)


def _render_golden(name: str) -> str:
    with tempfile.TemporaryDirectory() as base, pytest.MonkeyPatch.context() as monkeypatch:
        result, roots = asyncio.run(run_scenario(name, Path(base), monkeypatch))
    return json.dumps(_scrub(result, roots), indent=1, ensure_ascii=False) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Compare every scenario with its golden file, or write only the scenarios ``--write`` names."""
    parser = argparse.ArgumentParser(prog="python -m tests.support.reminder_goldens")
    parser.add_argument("--write", nargs="+", choices=list(SCENARIOS), default=[], metavar="SCENARIO")
    args = parser.parse_args(argv)
    if args.write:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        for name in args.write:
            (GOLDEN_DIR / f"{name}.json").write_text(_render_golden(name), encoding="utf-8", newline="\n")
            sys.stdout.write(f"wrote {name}.json\n")
        return 0
    status = 0
    for name in SCENARIOS:
        path = GOLDEN_DIR / f"{name}.json"
        actual = _render_golden(name)
        if not path.is_file():
            sys.stdout.write(f"missing {name}.json\n")
            status = 1
            continue
        expected = path.read_text(encoding="utf-8")
        if actual == expected:
            sys.stdout.write(f"ok {name}.json\n")
            continue
        status = 1
        sys.stdout.writelines(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                actual.splitlines(keepends=True),
                fromfile=f"{name}.json (golden)",
                tofile=f"{name}.json (now)",
            )
        )
    return status


if __name__ == "__main__":
    sys.exit(main())

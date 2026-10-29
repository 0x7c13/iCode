# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""What a tool group's row says when the group made one call and it succeeded: what that call did, in its own words.

Two or more calls, or one that failed or was interrupted, keep their counts: the row then sums up, and a failure
keeps its count in view.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any, Final
from urllib.parse import urlsplit

from chrys.app.tui.i18n import render_str
from chrys.app.tui.widgets.chat.fold_rows import duration_words, text_preview, tool_category
from chrys.foundation.i18n import Localizer, MessageDef, MessageRef, msg
from chrys.foundation.i18n.formatting import sanitize_legacy_scalar
from chrys.foundation.platform.files import surrogate_safe_text
from chrys.foundation.tool_kinds import KIND_DOC_CONVERTER

_READ = msg("tui.tool_card.group.single.read", fallback="Read {path}")
_VIEWED_IMAGE = msg("tui.tool_card.group.single.view_image", fallback="Viewed image {path}")
_READ_DOCUMENT = msg("tui.tool_card.group.single.read_document", fallback="Read document {path}")
_SEARCHED = msg("tui.tool_card.group.single.search", fallback="Searched for {pattern}")
_SEARCHED_IN = msg("tui.tool_card.group.single.search_in", fallback="Searched for {pattern} in {path}")
_FOUND_FILES = msg("tui.tool_card.group.single.find_files", fallback="Found files matching {pattern}")
_EDITED = msg("tui.tool_card.group.single.edit", fallback="Edited {path}")
_WROTE = msg("tui.tool_card.group.single.write", fallback="Wrote {path}")
_RAN_SHELL_COMMAND = msg("tui.tool_card.group.single.shell_command", fallback="Ran {shell} command: {command}")
"""A command of the shell the tool is named after, as the model explained it or, unexplained, its first line."""
_RAN_COMMAND = msg("tui.tool_card.group.single.command", fallback="Ran command: {command}")
"""A command whose tool names no shell the shell tool runs: a provider-hosted one, an external agent's, or the
``execute`` of older sessions."""
_SEARCHED_WEB = msg("tui.tool_card.group.single.web_search", fallback="Searched the web: {query}")
_FETCHED = msg("tui.tool_card.group.single.fetch", fallback="Fetched {url}")
_RAN_AGENT = msg("tui.tool_card.group.single.agent", fallback="Ran agent {agent}")
_LOADED_SKILL = msg("tui.tool_card.group.single.load_skill", fallback="Loaded skill {skill}")
_READ_SKILL_RESOURCE = msg("tui.tool_card.group.single.skill_resource", fallback="Read skill resource {resource}")
_RAN_SKILL_SCRIPT = msg("tui.tool_card.group.single.skill_script", fallback="Ran skill script {script}")
_CALLED_MCP_TOOL = msg("tui.tool_card.group.single.mcp", fallback="Called MCP tool {tool}")
_ASKED = msg("tui.tool_card.group.single.question", fallback="Asked you: {question}")
_ASKED_SEVERAL = msg(
    "tui.tool_card.group.single.questions",
    fallback="Asked you {count} question",
    plural_fallback="Asked you {count} questions",
)
"""One call that asked several questions at once."""
_UPDATED_TODOS = msg("tui.tool_card.group.single.todo", fallback="Updated the todo list")
_SLEPT_FOR = msg("tui.tool_card.group.single.sleep", fallback="Slept {duration}: {reason}")
"""A sleep that said why; one that did not keeps the group's ``Slept {duration}``."""
_CALLED = msg("tui.tool_card.group.single.other", fallback="Called {tool}")

_SHELL_LABELS: Final[dict[str, str]] = {"git_bash": "Git Bash"}


@dataclass(frozen=True)
class SingleCallRow:
    """The row's words for the one call, and the path within them that the row cuts in its middle when narrow."""

    words: str
    subject: str = ""


type _Row = Callable[[MessageRef], SingleCallRow]


def single_call_row(
    localizer: Localizer,
    tool_name: str,
    tool_kind: str,
    args: dict[str, Any] | None,
    *,
    hosted_family: str = "",
    slept_ms: int = 0,
    workspace_cwd: str = "",
) -> SingleCallRow | None:
    """What one succeeded call did, from its name, kind and arguments; None when those do not say enough to tell.

    A path inside *workspace_cwd* reads relative to it.
    """
    args = args or {}
    category = tool_category(tool_kind, hosted_family)
    name = sanitize_legacy_scalar(tool_name)
    hosted = bool(hosted_family)

    def row(reference: MessageRef) -> SingleCallRow:
        return SingleCallRow(render_str(localizer, reference))

    def path_row(words: MessageDef) -> SingleCallRow | None:
        path = _path(args, "path", workspace_cwd)
        return SingleCallRow(render_str(localizer, words.bind(path=path)), path) if path else None

    match category:
        case "read":
            if tool_name == "view_image":
                return path_row(_VIEWED_IMAGE)
            return path_row(_READ_DOCUMENT if tool_kind == KIND_DOC_CONVERTER else _READ)
        case "edit":
            # Another agent's edits, moves and deletes read as the counts read them: edited.
            return path_row(_WROTE if tool_name == "write_file" else _EDITED)
        case "search":
            if not (pattern := _line(args, "pattern")):
                return None
            if tool_name == "glob":
                return row(_FOUND_FILES.bind(pattern=pattern))
            if path := _path(args, "path", workspace_cwd):
                return row(_SEARCHED_IN.bind(pattern=pattern, path=path))
            return row(_SEARCHED.bind(pattern=pattern))
        case "command":
            command = _line(args, "reason") or _first_line(args, "command")
            if not command:
                return None
            shell = "" if hosted else _shell_label(tool_name)
            if shell:
                return row(_RAN_SHELL_COMMAND.bind(shell=shell, command=command))
            return row(_RAN_COMMAND.bind(command=command))
        case "web_search":
            query = _line(args, "query")
            return row(_SEARCHED_WEB.bind(query=query)) if query else None
        case "fetch":
            url = _url(args)
            return row(_FETCHED.bind(url=url)) if url else None
        case "agent":
            agent = _line(args, "agent_name") or _line(args, "agent") or name
            return row(_RAN_AGENT.bind(agent=agent))
        case "skill":
            return _skill_row(row, tool_name, args)
        case "mcp":
            # A hosted call that gave no name is named after its family, which names no tool.
            return row(_CALLED_MCP_TOOL.bind(tool=name)) if name and tool_name != hosted_family else None
        case "question":
            return _question_row(row, args)
        case "todo":
            return row(_UPDATED_TODOS.bind())
        case "sleep":
            reason = _line(args, "reason")
            if not reason:
                return None
            return row(_SLEPT_FOR.bind(duration=duration_words(localizer, slept_ms // 1000), reason=reason))
        case _:
            return row(_CALLED.bind(tool=name)) if name and not hosted else None


def _skill_row(row: _Row, tool_name: str, args: dict[str, Any]) -> SingleCallRow | None:
    skill = _line(args, "skill_name")
    if tool_name == "load_skill":
        return row(_LOADED_SKILL.bind(skill=skill)) if skill else None
    if tool_name == "read_skill_resource":
        resource = _line(args, "resource_name")
        resource = f"{skill}/{resource}" if skill and resource else resource or skill
        return row(_READ_SKILL_RESOURCE.bind(resource=resource)) if resource else None
    if tool_name == "run_skill_script":
        script = _line(args, "script_name")
        script = f"{skill}/{script}" if skill and script else script or skill
        return row(_RAN_SKILL_SCRIPT.bind(script=script)) if script else None
    return None


def _question_row(row: _Row, args: dict[str, Any]) -> SingleCallRow | None:
    questions = args.get("questions")
    if not isinstance(questions, list) or not questions:
        return None
    if len(questions) > 1:
        return row(_ASKED_SEVERAL.bind(count=len(questions)))
    question = questions[0]
    text = (_line(question, "question") or _line(question, "header")) if isinstance(question, dict) else ""
    return row(_ASKED.bind(question=text)) if text else None


def _shell_label(tool_name: str) -> str:
    """The shell *tool_name* names, as the row says it; empty for a tool not named after one.

    The shell tool takes its shell's name (``zsh``, ``pwsh``, ``git_bash``); another agent's shell call carries a title
    of its own, which may read like a name (``run_command``) yet names no shell. A shell the shell tool does not
    list, as an unusual ``$SHELL`` may be, reads as a command alone.
    """
    # Imported here, as the renderers' registry imports it: its module loads what only running a command needs.
    from chrys.service.tools.builtins.shell import SHELL_TOOL_NAMES

    if tool_name not in SHELL_TOOL_NAMES:
        return ""
    return _SHELL_LABELS.get(tool_name, tool_name)


def _line(args: dict[str, Any], key: str) -> str:
    """*key*'s text on one line, markup-safe; empty when it is absent or not text."""
    value = args.get(key)
    return text_preview(value) if isinstance(value, str) else ""


def _first_line(args: dict[str, Any], key: str) -> str:
    """The first line of *key*'s text, ``…`` standing for the lines after it."""
    value = args.get(key)
    if not isinstance(value, str):
        return ""
    first, _, rest = value.strip().partition("\n")
    first = text_preview(first)
    return f"{first} …" if first and rest.strip() else first


def _path(args: dict[str, Any], key: str, workspace_cwd: str) -> str:
    """*key*'s path as the row shows it: inside *workspace_cwd*, relative to it; elsewhere, as the call gave it.

    Empty for the working folder itself, which tells nothing the chat does not show.
    """
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        return ""
    path = PurePath(value)
    if workspace_cwd and path.is_absolute() and path.is_relative_to(workspace_cwd):
        relative = path.relative_to(workspace_cwd)
        # One that climbs out of the folder (``src/../../etc``) does not lie inside it.
        if ".." not in relative.parts:
            value = str(relative)
    if not PurePath(value).parts:
        return ""
    return sanitize_legacy_scalar(surrogate_safe_text(value))


def _url(args: dict[str, Any]) -> str:
    """The page *args* fetched, without its scheme, credentials, query or fragment: its host and path.

    Empty for one with no host to name: what else it holds may be a secret.
    """
    value = args.get("url")
    if not isinstance(value, str):
        return ""
    try:
        parts = urlsplit(value.strip())
    except ValueError:
        return ""
    host = parts.netloc.rpartition("@")[2]
    return text_preview(host + (parts.path if parts.path != "/" else "")) if host else ""

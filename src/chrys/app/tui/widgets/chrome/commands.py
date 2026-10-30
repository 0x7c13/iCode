# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Slash command registry for the TUI input bar."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from textual.content import Content

    from chrys.foundation.i18n import MessageRef

_SLASH_COMMAND_CANDIDATE_RE = re.compile(r"^[/\uff0f][A-Za-z][A-Za-z0-9_-]*(?:\s|$)")
_SLASH_NAME_WITH_ARGUMENTS_RE = re.compile(r"[/\uff0f]\S+\s")


@dataclass(frozen=True, slots=True)
class ManPageHeading:
    """An unindented localized section heading."""

    message: MessageRef


@dataclass(frozen=True, slots=True)
class ManPageProseBlock:
    """A localized prose block rendered with man-page indentation."""

    message: MessageRef


@dataclass(frozen=True, slots=True)
class ManPageVerbatimBlock:
    """An untranslated syntax or example block rendered with indentation."""

    text: str
    indent: int = 4


@dataclass(frozen=True, slots=True)
class ManPageRows:
    """Localized row prose paired with untranslated prefixes."""

    rows: tuple[tuple[str, MessageRef], ...]
    indent: int = 0


type ManPageSegment = ManPageHeading | ManPageProseBlock | ManPageVerbatimBlock | ManPageRows


@dataclass(frozen=True, slots=True)
class ManPageSpec:
    """Locale-neutral specification for one command man page."""

    name: str
    segments: tuple[ManPageSegment, ...]


def is_slash_command_candidate(text: str) -> bool:
    """Return True when text starts with a command-shaped slash token."""
    return "\n" not in text and "\r" not in text and _SLASH_COMMAND_CANDIDATE_RE.match(text) is not None


def is_slash_name_with_arguments(text: str) -> bool:
    """Whether *text* is one line naming a slash token and then arguments: ``/translate formal hello``."""
    return "\n" not in text and _SLASH_NAME_WITH_ARGUMENTS_RE.match(text) is not None


def completion_token_start(text: str, trigger: str) -> int | None:
    """Where the token an open suggestion list completes starts in *text*, or None when *text* holds none.

    The token runs from there to the end of the text: a list opens on a
    trigger typed at the end of the draft and filters on everything after it,
    so a pick replaces exactly what the list matched. A slash token (``/`` or
    ``/name `` for a second-level list) is the draft's leading one, since a
    slash list stays open only while the draft starts with it and a shortcut
    name may itself hold ``/``. Any other trigger (``@``) starts at its last
    occurrence. The fullwidth form of the first character (U+FF0F, U+FF20)
    counts as the trigger.
    """
    if not trigger:
        return None
    head = trigger[0]
    variants = (trigger, chr(ord(head) + 0xFEE0) + trigger[1:]) if 0x21 <= ord(head) <= 0x7E else (trigger,)
    if head == "/":
        return 0 if text.startswith(variants) else None
    start = max(text.rfind(variant) for variant in variants)
    return start if start >= 0 else None


@dataclass
class SlashCommandDef:
    """A slash command definition.

    Attributes:
        name: Command name without the leading ``/``.
        description: Locale-neutral short help text shown in the suggestion list.
        action: Callback invoked when the command is executed.
            Receives an optional argument string (e.g. theme name).
        subcommands: Optional callable that returns ``(value, label)`` pairs
            for a second-level suggestion list (e.g. available themes).
        synopsis: Optional command usage shown in the man page.
        options_help: Optional structured OPTIONS rows. Each row pairs an
            untranslated, alignment-preserving syntax prefix with localized prose.
        man_page: Optional locale-neutral detailed help body, either one prose
            reference or an ordered tuple of structured man-page segments.
    """

    name: str
    description: MessageRef
    action: Callable[[str], None] = field(default=lambda _arg: None)
    subcommands: Callable[[], Sequence[tuple[str, str | Content]]] | None = None
    initial: Callable[[], str] | None = None
    aliases: list[str] = field(default_factory=list)
    hidden: bool = False
    allow_while_running: bool = False
    synopsis: str | None = None
    options_help: tuple[tuple[str, MessageRef], ...] | None = None
    man_page: MessageRef | tuple[ManPageSegment, ...] | None = None

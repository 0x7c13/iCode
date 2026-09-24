# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Rich renderer for provider-hosted shell execution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from chrys.app.tui.widgets.chat.renderers.hosted_generic import (
    HOSTED_FILES,
    HOSTED_OUTPUT,
    HostedToolCall,
    _artifact_summary,
    _bounded_text,
    _display_text,
    _first_value,
    _hosted_card_css,
    _parsed_mapping,
    _sequence,
)
from chrys.app.tui.widgets.chat.renderers.skill import SKILL_EXIT
from chrys.foundation.i18n import msg
from chrys.foundation.i18n.formatting import sanitize_legacy_block, sanitize_legacy_scalar

_HOSTED_SHELL_TIMED_OUT_STATUS = msg("tui.hosted.shell.status.timed_out", fallback="timed out")
_HOSTED_SHELL_COMMANDS = msg("tui.hosted.shell.commands", fallback="Commands:")
_HOSTED_SHELL_STDOUT = msg("tui.hosted.shell.stdout", fallback="stdout:")
_HOSTED_SHELL_STDERR = msg("tui.hosted.shell.stderr", fallback="stderr:")
_HOSTED_SHELL_EXIT = msg("tui.hosted.shell.exit", fallback="Exit: {exit_code}")
_HOSTED_SHELL_TIMED_OUT = msg("tui.hosted.shell.timed_out", fallback="Timed out: {answer}")
_HOSTED_SHELL_YES = msg("tui.hosted.shell.yes", fallback="yes")
_HOSTED_SHELL_NO = msg("tui.hosted.shell.no", fallback="no")
_HOSTED_SHELL_EMPTY = msg("tui.hosted.shell.empty", fallback="No shell output")


def _commands(args: Mapping[str, Any]) -> list[str]:
    values = _sequence(args.get("commands"))
    if values:
        return [_display_text(value) for value in values]
    command = _first_value(args, "command", "input")
    return [_display_text(command)] if command is not None else []


def _shell_details(result: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    details = dict(metadata)
    parsed = _parsed_mapping(result)
    for key in ("stdout", "stderr", "exit_code", "timed_out"):
        if key not in details and key in parsed:
            details[key] = parsed[key]
    outputs = _sequence(parsed.get("outputs")) or _sequence(metadata.get("outputs"))
    if outputs:
        first = outputs[0]
        if isinstance(first, Mapping):
            for key in ("stdout", "stderr", "exit_code", "timed_out"):
                if key not in details and key in first:
                    details[key] = first[key]
    return details


class HostedShellToolCall(HostedToolCall):
    """Provider-hosted shell card without local approval or path chrome."""

    DEFAULT_CSS = _hosted_card_css("HostedShellToolCall")

    def __init__(
        self,
        call_id: str,
        tool_name: str,
        args_summary: str = "",
        args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(call_id, tool_name, args_summary, args=args)
        self._details: dict[str, Any] = {}

    def _label_details(self) -> list[str]:
        details: list[str] = []
        commands = _commands(self.args)
        if commands:
            first = sanitize_legacy_scalar(_bounded_text(commands[0]))
            details.append(f"{first} +{len(commands) - 1}" if len(commands) > 1 else first)
        exit_code = self._details.get("exit_code")
        if isinstance(exit_code, int) and not isinstance(exit_code, bool):
            details.append(self._render_message(SKILL_EXIT.bind(code=exit_code)))
        elif self._details.get("timed_out") is True:
            details.append(self._render_message(_HOSTED_SHELL_TIMED_OUT_STATUS.bind()))
        return details

    def _format_complete_details(self, result: str) -> str:
        sections: list[str] = []
        commands = _commands(self.args)
        if commands:
            sections.append(
                f"{self._render_message(_HOSTED_SHELL_COMMANDS.bind())}\n"
                f"{sanitize_legacy_block(_bounded_text(chr(10).join(commands)))}"
            )
        stdout = _first_value(self._details, "stdout")
        stderr = _first_value(self._details, "stderr")
        if stdout is not None:
            sections.append(
                f"{self._render_message(_HOSTED_SHELL_STDOUT.bind())}\n"
                f"{sanitize_legacy_block(_bounded_text(_display_text(stdout)))}"
            )
        if stderr is not None:
            sections.append(
                f"{self._render_message(_HOSTED_SHELL_STDERR.bind())}\n"
                f"{sanitize_legacy_block(_bounded_text(_display_text(stderr)))}"
            )
        exit_code = self._details.get("exit_code")
        timed_out = self._details.get("timed_out")
        if exit_code is not None:
            sections.append(
                self._render_message(
                    _HOSTED_SHELL_EXIT.bind(exit_code=sanitize_legacy_scalar(_display_text(exit_code)))
                )
            )
        if timed_out is not None:
            answer = self._render_message((_HOSTED_SHELL_YES if timed_out is True else _HOSTED_SHELL_NO).bind())
            sections.append(self._render_message(_HOSTED_SHELL_TIMED_OUT.bind(answer=answer)))
        if stdout is None and stderr is None and result:
            sections.append(
                f"{self._render_message(HOSTED_OUTPUT.bind())}\n{sanitize_legacy_block(_bounded_text(result))}"
            )
        artifact_text = _artifact_summary(self.artifacts, self._render_message)
        if artifact_text:
            sections.append(f"{self._render_message(HOSTED_FILES.bind())}\n{artifact_text}")
        return "\n\n".join(sections) or self._render_message(_HOSTED_SHELL_EMPTY.bind())

    def set_complete(self, result: str, duration_ms: int = 0, **kwargs: Any) -> None:
        metadata = kwargs.get("metadata")
        self._details = _shell_details(result, metadata if isinstance(metadata, Mapping) else {})
        super().set_complete(result, duration_ms, **kwargs)

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Generic and tool-discovery renderers for provider-hosted tools."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from typing import Any

from rich.text import Text
from textual.widgets import Static

from chrys.app.tui.i18n import render_str, widget_localizer
from chrys.app.tui.util.static_update import update_static_in_place
from chrys.app.tui.widgets.chat.tool_call import ToolCall, fmt_duration
from chrys.app.tui.widgets.chat.tool_renderers import hosted_failure_display_text, hosted_family_display_title
from chrys.foundation.hosted_tools import HostedToolStatus, normalize_hosted_tool_status
from chrys.foundation.i18n import MessageRef, msg
from chrys.foundation.i18n.formatting import format_message, sanitize_legacy_block, sanitize_legacy_scalar

HOSTED_STATUS_COMPLETED = msg("tui.hosted.status.completed", fallback="Completed")
HOSTED_STATUS_FAILED = msg("tui.hosted.status.failed", fallback="Failed")
HOSTED_STATUS_INTERRUPTED = msg("tui.hosted.status.interrupted", fallback="Interrupted")
HOSTED_RUNNING = msg("tui.hosted.status.running", fallback="running")
HOSTED_ARGUMENTS = msg("tui.hosted.arguments", fallback="Arguments:")
HOSTED_RESULT = msg("tui.hosted.result", fallback="Result:")
HOSTED_ARTIFACTS = msg("tui.hosted.artifacts", fallback="Artifacts:")
HOSTED_OUTPUT = msg("tui.hosted.output", fallback="Output:")
HOSTED_FILES = msg("tui.hosted.files", fallback="Files:")
HOSTED_QUERY = msg("tui.hosted.query", fallback="Query:")
_HOSTED_UNNAMED_ARTIFACT = msg("tui.hosted.artifact.unnamed", fallback="unnamed artifact")
_HOSTED_NO_OUTPUT = msg("tui.hosted.empty.no_output", fallback="No output")
_HOSTED_DISCOVERED_TOOLS = msg("tui.hosted.discovery.discovered_tools", fallback="Discovered tools:")
_HOSTED_DISCOVERED_COUNT = msg(
    "tui.hosted.discovery.discovered_count",
    fallback="{count} discovered",
    plural_fallback="{count} discovered",
)
_HOSTED_MORE_ARTIFACTS = msg(
    "tui.hosted.artifact.more",
    fallback="… {count} more",
    plural_fallback="… {count} more",
)
_HOSTED_NO_TOOL_DEFINITIONS = msg(
    "tui.hosted.discovery.empty",
    fallback="No tool definitions returned",
)

_HOSTED_DETAIL_MAX = 500


def _bounded_text(value: str) -> str:
    return value if len(value) <= _HOSTED_DETAIL_MAX else f"{value[:_HOSTED_DETAIL_MAX]}…"


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _bounded_json(value: object) -> str:
    return _bounded_text(_json_text(value))


def _parsed_mapping(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return {str(key): item for key, item in parsed.items()} if isinstance(parsed, Mapping) else {}


def _display_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return _json_text(value)


def _first_value(values: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = values.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return list(value)
    return []


def _artifact_summary(
    artifacts: list[dict[str, Any]],
    render_message: Callable[[MessageRef], str] = format_message,
) -> str:
    if not artifacts:
        return ""
    lines: list[str] = []
    for artifact in artifacts[:5]:
        name = _first_value(artifact, "path", "name", "id") or render_message(_HOSTED_UNNAMED_ARTIFACT.bind())
        details = [str(value) for key in ("mime", "size") if (value := artifact.get(key)) not in (None, "")]
        suffix = f" ({', '.join(details)})" if details else ""
        lines.append(sanitize_legacy_block(f"- {name}{suffix}"))
    if len(artifacts) > 5:
        lines.append(f"- {render_message(_HOSTED_MORE_ARTIFACTS.bind(count=len(artifacts) - 5))}")
    return _bounded_text("\n".join(lines))


def _hosted_card_css(type_name: str) -> str:
    """Return the base hosted-card rules scoped to one concrete widget type."""
    return f"""
    {type_name} {{
        padding: 0 0 0 2;
        height: auto;
        margin: 0 0 1 0;
    }}
    {type_name} #tc-label {{
        height: auto;
    }}
    {type_name} #tc-body {{
        height: auto;
        color: $text-muted;
    }}
    {type_name} > #tc-panel {{
        height: auto;
        margin: 0 0 0 2;
        border: round $tui-border-primary 50%;
        border-title-color: $text;
        border-title-style: not bold;
        border-subtitle-align: right;
        border-subtitle-color: $text-muted;
        padding: 0 1;
    }}
    {type_name}.-success > #tc-panel {{
        border: round $tui-border-success 30%;
        border-title-color: $success;
        border-subtitle-color: $success;
    }}
    {type_name}.-error > #tc-panel {{
        border: round $tui-border-error 50%;
        border-title-color: $text-error;
        border-subtitle-color: $text-error;
    }}
    {type_name}.-rejected > #tc-panel {{
        border: round $tui-border-warning 50%;
        border-title-color: $warning;
        border-subtitle-color: $warning;
    }}
    """


class HostedToolCall(ToolCall):
    """Bounded provider-neutral fallback for hosted tool families."""

    DEFAULT_CSS = _hosted_card_css("HostedToolCall")

    def _render_message(self, reference: MessageRef) -> str:
        return render_str(widget_localizer(self), reference)

    def _status_text(self) -> str:
        canonical = normalize_hosted_tool_status(self.canonical_status)
        canonical_messages = {
            HostedToolStatus.COMPLETED: HOSTED_STATUS_COMPLETED,
            HostedToolStatus.FAILED: HOSTED_STATUS_FAILED,
            HostedToolStatus.INTERRUPTED: HOSTED_STATUS_INTERRUPTED,
        }
        if definition := canonical_messages.get(canonical):
            return self._render_message(definition.bind())
        if not self.provider_status and canonical == HostedToolStatus.RUNNING:
            return self._render_message(HOSTED_RUNNING.bind())
        status = sanitize_legacy_scalar(self.provider_status or self.canonical_status or HOSTED_RUNNING.fallback)
        return status[:1].upper() + status[1:]

    def _title_text(self) -> str:
        # The hosted adapter backfills a missing provider tool name with the
        # family code; that code is still the wire-level identity, so it is
        # shown as-is ("openai/image") and the localized family title only
        # covers calls that carry no name at all.
        name = self.tool_name or self.provider_item_type
        if not name:
            name = hosted_family_display_title(self.hosted_family, render_message=self._render_message) or "tool"
        name = sanitize_legacy_scalar(name)
        provider = sanitize_legacy_scalar(self.provider)
        return f"{provider}/{name}" if provider else name

    def _label_details(self) -> list[str]:
        return []

    def _label_text(self, duration_ms: int = 0) -> Text:
        label = Text("• ", style="bold")
        label.append(self._title_text(), style="bold")
        for detail in self._label_details():
            label.append(" · ", style="dim")
            label.append(detail, style="dim")
        if duration_ms:
            label.append(f" ({fmt_duration(duration_ms)})", style="dim")
        return label

    def _refresh_status_subtitle(self) -> None:
        status = self._status_text()
        with suppress(Exception):
            self.query_one("#tc-panel").border_subtitle = Text(status)

    def update_args(self, args: dict[str, Any]) -> None:
        """Refresh both the argument panel and family summary."""
        super().update_args(args)
        with suppress(Exception):
            self.query_one("#tc-label", Static).update(self._label_text())

    def update_hosted_status(self, status: str, provider_status: str) -> None:
        """Refresh the collapsed lifecycle summary."""
        self.canonical_status = status
        self.provider_status = provider_status or self.provider_status
        with suppress(Exception):
            self.query_one("#tc-label", Static).update(self._label_text())
        self._refresh_status_subtitle()

    def update_progress(self, lines: list[str]) -> None:
        """Show a bounded progress snapshot without terminalizing the card."""
        preview = sanitize_legacy_block(_bounded_text("\n".join(lines)))
        self._body_pinned = bool(preview)
        with suppress(Exception):
            update_static_in_place(
                self.query_one("#tc-body", Static), Text(preview) if preview else self._render_spinner()
            )

    def _truncate_result(self, text: str) -> str:
        """Keep hosted result details intentionally compact."""
        return _bounded_text(text)

    def _format_complete_details(self, result: str) -> str:
        sections: list[str] = []
        if self.args:
            sections.append(
                f"{self._render_message(HOSTED_ARGUMENTS.bind())} {sanitize_legacy_block(_bounded_json(self.args))}"
            )
        if result:
            sections.append(
                f"{self._render_message(HOSTED_RESULT.bind())}\n{sanitize_legacy_block(_bounded_text(result))}"
            )
        artifact_text = _artifact_summary(self.artifacts, self._render_message)
        if artifact_text:
            sections.append(f"{self._render_message(HOSTED_ARTIFACTS.bind())}\n{artifact_text}")
        return "\n\n".join(sections) or self._render_message(_HOSTED_NO_OUTPUT.bind())

    def set_error(self, error: str) -> None:
        """Render a bounded provider error payload."""
        super().set_error(
            _bounded_text(hosted_failure_display_text(error, self.metadata, render_message=self._render_message))
        )
        self._refresh_status_subtitle()

    def set_complete(self, result: str, duration_ms: int = 0, **kwargs: Any) -> None:
        """Render bounded arguments, result text, and artifact descriptors."""
        artifacts = kwargs.get("artifacts")
        self.artifacts = (
            [
                {str(key): value for key, value in artifact.items()}
                for artifact in artifacts
                if isinstance(artifact, dict)
            ]
            if isinstance(artifacts, list)
            else []
        )
        display_result = hosted_failure_display_text(
            result,
            kwargs.get("metadata"),
            render_message=self._render_message,
        )
        display = self._format_complete_details(display_result)
        super().set_complete(display, duration_ms, **kwargs)
        self._refresh_status_subtitle()


class HostedToolDiscoveryCall(HostedToolCall):
    """Thin hosted fallback specialized for provider tool discovery."""

    DEFAULT_CSS = _hosted_card_css("HostedToolDiscoveryCall")

    def __init__(
        self,
        call_id: str,
        tool_name: str,
        args_summary: str = "",
        args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(call_id, tool_name, args_summary, args=args)
        self._discovered_count: int | None = None

    def _label_details(self) -> list[str]:
        details: list[str] = []
        scope = _first_value(self.args, "namespace", "query")
        if scope is not None:
            details.append(sanitize_legacy_scalar(_bounded_text(_display_text(scope))))
        if self._discovered_count is not None:
            details.append(self._render_message(_HOSTED_DISCOVERED_COUNT.bind(count=self._discovered_count)))
        return details

    def _format_complete_details(self, result: str) -> str:
        parsed = _parsed_mapping(result)
        definitions = _first_value(parsed, "tools", "selected", "definitions")
        sections = (
            [f"{self._render_message(HOSTED_QUERY.bind())} {sanitize_legacy_block(_bounded_json(self.args))}"]
            if self.args
            else []
        )
        if definitions is not None:
            sections.append(
                f"{self._render_message(_HOSTED_DISCOVERED_TOOLS.bind())}\n"
                f"{sanitize_legacy_block(_bounded_json(definitions))}"
            )
        elif result:
            sections.append(
                f"{self._render_message(_HOSTED_DISCOVERED_TOOLS.bind())}\n"
                f"{sanitize_legacy_block(_bounded_text(result))}"
            )
        return "\n\n".join(sections) or self._render_message(_HOSTED_NO_TOOL_DEFINITIONS.bind())

    def set_complete(self, result: str, duration_ms: int = 0, **kwargs: Any) -> None:
        parsed = _parsed_mapping(result)
        self._discovered_count = None
        for key in ("tools", "selected", "definitions"):
            if key in parsed:
                self._discovered_count = len(_sequence(parsed[key]))
                break
        super().set_complete(result, duration_ms, **kwargs)

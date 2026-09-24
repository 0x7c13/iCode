# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Rich renderer for provider-hosted search and fetch calls."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from chrys.app.tui.widgets.chat.renderers.hosted_generic import (
    HOSTED_RESULT,
    HostedToolCall,
    _bounded_json,
    _bounded_text,
    _display_text,
    _first_value,
    _hosted_card_css,
    _parsed_mapping,
    _sequence,
)
from chrys.foundation.i18n import msg
from chrys.foundation.i18n.formatting import sanitize_legacy_block, sanitize_legacy_scalar

_HOSTED_SEARCH_RESULT_COUNT = msg(
    "tui.hosted.search.result_count",
    fallback="{count} result",
    plural_fallback="{count} results",
)
_HOSTED_SEARCH_QUERIES = msg("tui.hosted.search.queries", fallback="Queries:")
_HOSTED_SEARCH_QUERY_ACTION = msg("tui.hosted.search.query_action", fallback="Query/action:")
_HOSTED_SEARCH_RESULTS = msg("tui.hosted.search.results", fallback="Results: {n}")
_HOSTED_SEARCH_CITATIONS = msg(
    "tui.hosted.search.citations",
    fallback="Citations and URLs:",
)
_HOSTED_SEARCH_EMPTY = msg(
    "tui.hosted.search.empty",
    fallback="No search details returned",
)


def _search_subject(args: Mapping[str, Any]) -> str:
    queries = _sequence(args.get("queries"))
    if queries:
        subject = _display_text(queries[0])
        return f"{subject} +{len(queries) - 1}" if len(queries) > 1 else subject
    action = args.get("action")
    if isinstance(action, Mapping):
        value = _first_value(action, "query", "url", "type")
        if value is not None:
            return _display_text(value)
    value = _first_value(args, "query", "url", "input")
    return _display_text(value)


def _search_action_type(args: Mapping[str, Any]) -> str:
    action_type = args.get("type")
    if isinstance(action_type, str) and action_type.strip():
        return action_type.strip()
    action = args.get("action")
    if isinstance(action, Mapping):
        action_type = action.get("type")
        if isinstance(action_type, str) and action_type.strip():
            return action_type.strip()
    return ""


def _result_items(parsed: Mapping[str, Any]) -> list[Any]:
    for key in ("results", "citations", "items"):
        items = _sequence(parsed.get(key))
        if items:
            return items
    return []


def _result_count(result: str, metadata: Mapping[str, Any]) -> int | None:
    for key in ("result_count", "results_count", "count"):
        value = metadata.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    parsed = _parsed_mapping(result)
    for key in ("results", "citations", "items"):
        if key in parsed:
            return len(_sequence(parsed[key]))
    return None


def _is_action_echo(result: str) -> bool:
    candidate = result.removeprefix("Error: ").strip()
    return set(_parsed_mapping(candidate)) == {"action"}


def _citation_lines(parsed: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for item in _result_items(parsed)[:5]:
        if isinstance(item, Mapping):
            title = _first_value(item, "title", "filename", "name", "text")
            url = _first_value(item, "url", "uri")
            if title is not None and url is not None:
                lines.append(f"- {_display_text(title)} — {_display_text(url)}")
            elif url is not None:
                lines.append(f"- {_display_text(url)}")
            else:
                lines.append(f"- {_bounded_json(item)}")
        else:
            lines.append(f"- {_display_text(item)}")
    return lines


class HostedSearchToolCall(HostedToolCall):
    """Search/fetch card with query, result-count, and citation summaries."""

    DEFAULT_CSS = _hosted_card_css("HostedSearchToolCall")

    def __init__(
        self,
        call_id: str,
        tool_name: str,
        args_summary: str = "",
        args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(call_id, tool_name, args_summary, args=args)
        self._result_count: int | None = None

    def _label_details(self) -> list[str]:
        details: list[str] = []
        subject = _search_subject(self.args)
        if subject:
            details.append(sanitize_legacy_scalar(_bounded_text(subject)))
        if self._result_count is not None:
            details.append(self._render_message(_HOSTED_SEARCH_RESULT_COUNT.bind(count=self._result_count)))
        return details

    def _border_title_text(self) -> str:
        return sanitize_legacy_scalar(_search_action_type(self.args) or self.tool_name)

    def _format_complete_details(self, result: str) -> str:
        parsed = _parsed_mapping(result)
        sections: list[str] = []
        queries = _sequence(self.args.get("queries"))
        if queries:
            sections.append(
                f"{self._render_message(_HOSTED_SEARCH_QUERIES.bind())} {sanitize_legacy_block(_bounded_json(queries))}"
            )
        elif subject := _search_subject(self.args):
            sections.append(
                f"{self._render_message(_HOSTED_SEARCH_QUERY_ACTION.bind())} "
                f"{sanitize_legacy_block(_bounded_text(subject))}"
            )
        if self._result_count is not None:
            sections.append(self._render_message(_HOSTED_SEARCH_RESULTS.bind(n=self._result_count)))
        citations = _citation_lines(parsed)
        if citations:
            sections.append(
                f"{self._render_message(_HOSTED_SEARCH_CITATIONS.bind())}\n"
                f"{sanitize_legacy_block(_bounded_text(chr(10).join(citations)))}"
            )
        elif result and not _is_action_echo(result):
            sections.append(
                f"{self._render_message(HOSTED_RESULT.bind())}\n{sanitize_legacy_block(_bounded_text(result))}"
            )
        return "\n\n".join(sections) or self._render_message(_HOSTED_SEARCH_EMPTY.bind())

    def set_complete(self, result: str, duration_ms: int = 0, **kwargs: Any) -> None:
        metadata = kwargs.get("metadata")
        self._result_count = _result_count(result, metadata if isinstance(metadata, Mapping) else {})
        super().set_complete(result, duration_ms, **kwargs)

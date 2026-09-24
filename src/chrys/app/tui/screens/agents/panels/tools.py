# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tools configuration panel — composable widget for the Tools tab."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Label

from chrys.app.tui.i18n import render_str, widget_localizer
from chrys.app.tui.widgets import Checkbox
from chrys.foundation.i18n import MessageDef, msg

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from chrys.service.profiles.agents.schema import ToolsConfig

_BUILTIN_TOOL_CATEGORIES = msg("tui.agent_config.tools.categories", fallback="Builtin Tool Categories")
_CATEGORIES_DESCRIPTION = msg(
    "tui.agent_config.tools.categories_description",
    fallback="Toggle categories to enable or disable",
)
_FILESYSTEM_READ = msg("tui.agent_config.tools.filesystem_read", fallback="Filesystem Read")
_FILESYSTEM_READ_DESCRIPTION = msg(
    "tui.agent_config.tools.filesystem_read_description",
    fallback="Read files from the filesystem",
)
_FILESYSTEM_WRITE = msg("tui.agent_config.tools.filesystem_write", fallback="Filesystem Write")
_FILESYSTEM_WRITE_DESCRIPTION = msg(
    "tui.agent_config.tools.filesystem_write_description",
    fallback="Create and edit files on the filesystem",
)
_SEARCH = msg("tui.agent_config.tools.search", fallback="File search")
_SEARCH_DESCRIPTION = msg(
    "tui.agent_config.tools.search_description",
    fallback="Search file contents and find files by pattern",
)
_SHELL = msg("tui.agent_config.tools.shell", fallback="Shell")
_SHELL_DESCRIPTION = msg(
    "tui.agent_config.tools.shell_description",
    fallback="Execute shell commands in a subprocess",
)
_ASK_USER = msg("tui.agent_config.tools.ask_user", fallback="Ask User")
_ASK_USER_DESCRIPTION = msg(
    "tui.agent_config.tools.ask_user_description",
    fallback="Prompt the user with one or more questions for input or clarification",
)
_SLEEP = msg("tui.agent_config.tools.sleep", fallback="Sleep")
_SLEEP_DESCRIPTION = msg(
    "tui.agent_config.tools.sleep_description",
    fallback="Pause with a skippable countdown, up to 3600 seconds",
)
_DOCUMENT_CONVERTER = msg(
    "tui.agent_config.tools.document_converter",
    fallback="Document Converter (PDF, Office)",
)
_DOCUMENT_CONVERTER_DESCRIPTION = msg(
    "tui.agent_config.tools.document_converter_description",
    fallback="Convert PDF, DOCX, PPTX, XLSX, XLS to Markdown",
)
_TODO_LIST = msg("tui.agent_config.tools.todo_list", fallback="Todo List")
_TODO_LIST_DESCRIPTION = msg(
    "tui.agent_config.tools.todo_list_description",
    fallback="Track multi-step work as a live task checklist",
)

_WEB_SEARCH = msg("tui.agent_config.tools.web_search", fallback="Web search")
_WEB_SEARCH_DESCRIPTION = msg(
    "tui.agent_config.tools.web_search_description",
    fallback=(
        "Search the web for links and snippets. Uses Exa (exa.ai) unless the agent profile names other providers; "
        "queries are sent to that service and may carry details from your conversation, so mind your data and privacy"
    ),
)
_WEB_FETCH = msg("tui.agent_config.tools.web_fetch", fallback="Web fetch")
_WEB_FETCH_DESCRIPTION = msg(
    "tui.agent_config.tools.web_fetch_description",
    fallback="Read a page's text by URL",
)

# All available builtin tool categories (from service/tools/registry.py).
# (category_key, display_label, description, enabled)
_ALL_CATEGORIES: list[tuple[str, MessageDef, MessageDef, bool]] = [
    ("filesystem.read", _FILESYSTEM_READ, _FILESYSTEM_READ_DESCRIPTION, True),
    ("filesystem.write", _FILESYSTEM_WRITE, _FILESYSTEM_WRITE_DESCRIPTION, True),
    ("search", _SEARCH, _SEARCH_DESCRIPTION, True),
    ("web_search", _WEB_SEARCH, _WEB_SEARCH_DESCRIPTION, True),
    ("web_fetch", _WEB_FETCH, _WEB_FETCH_DESCRIPTION, True),
    ("shell", _SHELL, _SHELL_DESCRIPTION, True),
    ("ask_user", _ASK_USER, _ASK_USER_DESCRIPTION, True),
    ("sleep", _SLEEP, _SLEEP_DESCRIPTION, True),
    ("doc_converter", _DOCUMENT_CONVERTER, _DOCUMENT_CONVERTER_DESCRIPTION, True),
    ("todo", _TODO_LIST, _TODO_LIST_DESCRIPTION, True),
]


def _cat_id(cat: str) -> str:
    """Sanitise a category name for use as a CSS-safe widget ID."""
    return cat.replace("_", "-").replace(".", "-")


class ToolsConfigPanel(VerticalScroll):
    """Composable widget for the Tools configuration tab.

    Displays builtin tool categories as checkboxes.
    """

    DEFAULT_CSS = """
    ToolsConfigPanel {
        height: 1fr;
        /* Left inset only: the scrollbar is carved out inside the padding,
           so right padding would sit right of the scrollbar. The right inset
           rides child margins below instead. */
        padding: 0 0 0 2;
        scrollbar-size-vertical: 1;
        scrollbar-gutter: stable;
    }
    ToolsConfigPanel > * {
        margin-right: 1;
    }
    ToolsConfigPanel .tc-section-title {
        color: $secondary;
        text-style: bold;
        height: 1;
    }
    ToolsConfigPanel .tc-section-desc {
        color: $text-muted;
        height: 1;
        margin: 0 1 1 0;
    }
    ToolsConfigPanel .tc-categories {
        height: auto;
    }
    ToolsConfigPanel Checkbox {
        height: 1;
        padding: 0;
        border: none;
        background: transparent;
        margin: 0;
    }
    ToolsConfigPanel Checkbox > .toggle--button {
        color: $foreground 35%;
        background: $foreground 6%;
    }
    ToolsConfigPanel Checkbox.-on > .toggle--button {
        color: $success;
        background: $foreground 6%;
    }
    ToolsConfigPanel .tc-cat-desc {
        color: $text-muted;
        height: auto;
        width: 1fr;
        margin: 0 0 1 4;
        text-wrap: wrap;
        text-overflow: fold;
    }
    """

    def __init__(self, tools_config: ToolsConfig | None = None, *, read_only: bool = False) -> None:
        from chrys.service.profiles.agents.schema import ToolsConfig as TC

        self._tools = tools_config or TC()
        self._active_categories: set[str] = set(self._tools.builtins)
        self._read_only = read_only
        super().__init__()

    def compose(self) -> ComposeResult:
        localizer = widget_localizer(self)
        yield Label(render_str(localizer, _BUILTIN_TOOL_CATEGORIES.bind()), classes="tc-section-title")
        yield Label(render_str(localizer, _CATEGORIES_DESCRIPTION.bind()), classes="tc-section-desc")
        with Vertical(classes="tc-categories"):
            for cat, label, desc, enabled in _ALL_CATEGORIES:
                cb = Checkbox(
                    render_str(localizer, label.bind()),
                    value=cat in self._active_categories,
                    id=f"tc-cat-{_cat_id(cat)}",
                )
                cb.disabled = self._read_only or not enabled
                yield cb
                yield Label(render_str(localizer, desc.bind()), classes="tc-cat-desc")

    def get_config(self) -> list[str]:
        """Return list of active builtin category names."""
        result: list[str] = []
        for cat, _label, _desc, _enabled in _ALL_CATEGORIES:
            try:
                cb = self.query_one(f"#tc-cat-{_cat_id(cat)}", Checkbox)
            except NoMatches:
                if cat in self._active_categories:
                    result.append(cat)
            else:
                if cb.value:
                    result.append(cat)
        # Categories without a switch here (another version's tools) pass through a save untouched.
        known = {cat for cat, *_ in _ALL_CATEGORIES}
        result.extend(dict.fromkeys(cat for cat in self._tools.builtins if cat not in known))
        return result

    def validate(self) -> list[str]:
        """Validate tools configuration."""
        return []

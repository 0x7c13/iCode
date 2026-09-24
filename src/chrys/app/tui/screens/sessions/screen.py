# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""SessionsScreen — modal for browsing and restoring persisted sessions.

Opens as a centered modal overlay (Ctrl+S).  Displays all persisted sessions
in a sortable DataTable: click a column header to sort (click again to flip
direction), type in the bottom search box to filter across every column plus
each session's user prompts (rows found only through prompt text render in
italics, list after the column matches, and carry the matched context in
their tooltip), and forked sessions nest under their parent as a tree.  Session metadata is
scanned asynchronously and rendered once complete so sorting, filtering, and
tree parentage appear in a stable order.  Resume to load, Delete to remove.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual import on, work
from textual.containers import HorizontalGroup, VerticalGroup
from textual.message import Message
from textual.widgets import Button, DataTable, Input

from chrys.app.tui.binding_display import CLOSE_BINDING, DELETE_BINDING, localized_binding
from chrys.app.tui.i18n import LocaleController, render_str, widget_localizer
from chrys.app.tui.screens.dialogs.base import BaseDialog
from chrys.app.tui.screens.sessions.presenter import (
    COLUMNS,
    WORKFLOW_COLUMNS,
    SessionRow,
    WorkflowSessionPick,
    build_session_rows,
    build_workflow_rows,
    column_by_key,
    format_tokens,
    last_interaction_display,
    profile_display,
    title_display,
)
from chrys.app.tui.util.rich_style import rich_style_from_textual_color
from chrys.app.tui.widgets import ChrysLoadingIndicator, DialogButtonRow, DialogButtonSpec, HatchedEmptyState
from chrys.app.tui.widgets.input import EnhancedInput
from chrys.foundation.branding import APP_DISPLAY_NAME
from chrys.foundation.i18n import DisplayPath, MessageRef, msg
from chrys.foundation.util.session_ids import session_short_id

if TYPE_CHECKING:
    from textual import events
    from textual.app import ComposeResult
    from textual.timer import Timer

    from chrys.service.state.store import SessionMeta, StateStore


_SEARCH_DEBOUNCE_SECONDS = 0.15
"""Delay between the last search keystroke and the table re-render."""

_TREE_LEVEL_WIDTH = 2
"""Cells one fork level adds to the Session ID column ('└ ' / '│ ')."""

_DELETE_SESSION_TITLE = msg("tui.sessions.title.delete", fallback="Delete Session")
_SESSION_OPEN_ELSEWHERE = msg(
    "tui.sessions.delete.open_elsewhere",
    fallback="Session is open in another {app_name} instance.",
)
_SESSIONS_TITLE = msg("tui.sessions.title", fallback="Chat Sessions")
_WORKFLOW_SESSIONS_TITLE = msg("tui.sessions.title.workflow", fallback="Workflow Sessions")
_TOOLTIP_WORKFLOW = msg("tui.sessions.tooltip.workflow", fallback="Workflow: {workflow}")
_TOOLTIP_STATUS = msg("tui.sessions.tooltip.status", fallback="Status: {status}")
_TOOLTIP_RUN = msg("tui.sessions.tooltip.run", fallback="Run: {run_id}")
_NO_SAVED_SESSIONS = msg("tui.sessions.empty", fallback="No saved sessions.")
_SEARCH_PLACEHOLDER = msg(
    "tui.sessions.search_placeholder",
    fallback="Search sessions… (matches any column & your prompts)",
)
_RESUME = msg("tui.sessions.button.resume", fallback="Resume")
_DELETE = msg("tui.sessions.button.delete", fallback="Delete")
_CLOSE = msg("tui.sessions.button.close", fallback="Close")
_LOADING_SESSIONS = msg("tui.sessions.loading", fallback="Loading sessions")
_SESSION_COUNT = msg("tui.sessions.count", fallback="{count_text} sessions")
_WORKFLOW_SESSION_COUNT = msg("tui.sessions.workflow_count", fallback="{count_text} workflow sessions")
_TOOLTIP_TITLE = msg("tui.sessions.tooltip.title", fallback="Title: {title}")
_TOOLTIP_AGENT = msg("tui.sessions.tooltip.agent", fallback="Agent: {agent}")
_TOOLTIP_PROMPT_MATCH = msg("tui.sessions.tooltip.prompt_match", fallback="Prompt match: {prompt}")
_TOOLTIP_DIRECTORY = msg("tui.sessions.tooltip.directory", fallback="Directory: {path}")
_TOOLTIP_TURNS = msg("tui.sessions.tooltip.turns", fallback="Turns: {turns}")
_TOOLTIP_TOTAL_TOKENS = msg("tui.sessions.tooltip.total_tokens", fallback="Total tokens: {tokens}")
_TOOLTIP_LAST_INTERACTION = msg(
    "tui.sessions.tooltip.last_interaction",
    fallback="Last interaction: {interaction}",
)
_TOOLTIP_FORKED_FROM = msg("tui.sessions.tooltip.forked_from", fallback="Forked from: {session_id}")
_DELETE_SESSION_MESSAGE = msg(
    "tui.sessions.delete.confirm_message",
    fallback='Delete session\n"{session_id}"?\n\nThis cannot be undone.',
    multiline=True,
)


class _SessionTable(DataTable):
    """DataTable subclass that shows per-row tooltips on hover.

    Textual's tooltip system is per-widget: when the mouse moves within the
    same widget and the tooltip is already visible, the screen hides it
    (screen.py ``_handle_mouse_move`` line ~1657).  To support per-ROW
    tooltips we defer the tooltip update via ``call_later`` so it runs
    *after* the screen's hide logic, then immediately trigger the tooltip
    display.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._row_tooltips: dict[int, Text] = {}
        self._last_hover_row: int = -1

    def set_row_tooltips(self, tooltips: dict[int, Text]) -> None:
        """Set tooltip text for each row index."""
        self._row_tooltips = tooltips

    def _on_mouse_move(self, event: events.MouseMove) -> None:
        # Textual dispatches DataTable._on_mouse_move after this method; do
        # not call super() here or hover bookkeeping runs twice.
        meta = event.style.meta
        if meta and "row" in meta:
            row_idx = meta["row"]
            if row_idx != self._last_hover_row:
                self._last_hover_row = row_idx
                tip = self._row_tooltips.get(row_idx)
                # Defer so it runs after screen._handle_mouse_move hides tooltip
                self.call_later(self._apply_row_tooltip, tip)
        else:
            self._last_hover_row = -1
            self.tooltip = None

    def _apply_row_tooltip(self, tip: Text | None) -> None:
        """Apply the tooltip and force immediate display."""
        self.tooltip = tip
        if tip is not None:
            import contextlib

            with contextlib.suppress(Exception):
                self.screen._handle_tooltip_timer(self)


class _SearchInput(EnhancedInput):
    """Filter box: Escape clears the query, then hands focus back."""

    class Escaped(Message):
        """Posted when Escape is pressed on an already-empty search box."""

    async def _on_key(self, event: events.Key) -> None:
        # No super() call: Textual dispatches the EnhancedInput/Input base
        # handlers through the MRO on its own.
        if event.key != "escape":
            return
        event.stop()
        event.prevent_default()
        if self.value:
            self.value = ""
        else:
            self.post_message(self.Escaped())


def _next_cursor_row_after_delete(deleted_row: int, remaining_count: int) -> int | None:
    """Return the row to highlight after deleting one row from a table."""
    if remaining_count <= 0:
        return None
    return min(deleted_row, remaining_count - 1)


def _next_session_id_after_delete(session_ids: list[str], deleted_row: int) -> str | None:
    """Return the visible neighbor to highlight after deleting one row."""
    if not 0 <= deleted_row < len(session_ids):
        return None
    next_row = deleted_row + 1
    if next_row < len(session_ids):
        return session_ids[next_row]
    previous_row = deleted_row - 1
    if previous_row >= 0:
        return session_ids[previous_row]
    return None


class SessionsScreen(BaseDialog[str | WorkflowSessionPick | None]):
    """Modal for browsing persisted sessions.

    Returns a chat session id or a specific workflow run to open; None closes the modal.
    """

    BINDINGS: ClassVar[list] = [
        localized_binding("escape", "dismiss", CLOSE_BINDING),
        localized_binding("delete", "delete_session", DELETE_BINDING),
    ]

    CSS_PATH = "screen.tcss"

    #: Sentinel prefix: tells caller to delete this session and start fresh.
    DELETE_AND_NEW_PREFIX = "__delete_and_new__:"

    def __init__(
        self,
        state_store: StateStore,
        current_session_id: str = "",
        *,
        locale_controller: LocaleController | None = None,
        workflow_mode: bool = False,
    ) -> None:
        self._state_store = state_store
        self._current_session_id = current_session_id
        self._locale_controller = locale_controller
        self._workflow_mode = workflow_mode
        self._columns = WORKFLOW_COLUMNS if workflow_mode else COLUMNS
        self._sessions: list[SessionMeta] = []
        self._rendered_sessions: list[SessionMeta] = []
        self._session_ids: list[str] = []
        self._rows: list[SessionRow] = []
        self._sort_column: str = column_by_key("").key
        self._sort_reverse: bool = column_by_key("").default_reverse
        self._loading = True
        # Becomes True once the cursor has jumped to the currently loaded
        # session (or the initial scan finished without finding it).
        self._cursor_initialized = False
        self._search_timer: Timer | None = None
        super().__init__()

    def compose(self) -> ComposeResult:
        localizer = widget_localizer(self)
        with VerticalGroup(id="container") as container:
            title = _WORKFLOW_SESSIONS_TITLE if self._workflow_mode else _SESSIONS_TITLE
            container.border_title = Text(render_str(localizer, title.bind()))
            with VerticalGroup(id="sessions-loading-state"):
                yield ChrysLoadingIndicator(id="sessions-loading")
            yield HatchedEmptyState(render_str(localizer, _NO_SAVED_SESSIONS.bind()), id="empty-note")
            table = _SessionTable(id="sessions", cursor_type="row")
            table.display = False
            yield table
            with HorizontalGroup(id="footer") as footer:
                footer.display = False
                # The themed border lives on the wrapper so the input's
                # background stays inside it (a border directly on the Input
                # would share its background; compact=True would strip the
                # border entirely with `border: none !important`).
                with HorizontalGroup(id="search-box"):
                    yield _SearchInput(
                        placeholder=render_str(localizer, _SEARCH_PLACEHOLDER.bind()),
                        id="search",
                        compact=True,
                    )
                yield DialogButtonRow(
                    DialogButtonSpec(
                        Text(render_str(localizer, _RESUME.bind())), id="resume", variant="primary", disabled=True
                    ),
                    DialogButtonSpec(
                        Text(render_str(localizer, _DELETE.bind())), id="delete", variant="error", disabled=True
                    ),
                    DialogButtonSpec(Text(render_str(localizer, _CLOSE.bind())), id="cancel", variant="warning"),
                    id="buttons",
                )

    def on_mount(self) -> None:
        self._load_sessions()

    # ------------------------------------------------------------------
    # Loading & rendering
    # ------------------------------------------------------------------

    @work(exclusive=True, group="load-sessions")
    async def _load_sessions(
        self,
        preferred_cursor_row: int | None = None,
        preferred_session_id: str | None = None,
    ) -> None:
        """Stream session metas from the store, then render one stable table.

        Batches arrive from :meth:`StateStore.stream_session_metas`, but the
        UI waits for the full set before sorting/filtering/tree assembly.
        Rendering partial batches makes rows jump as later sessions and fork
        parents arrive.
        """
        self._loading = True
        self._update_chrome(
            visible_rows=len(self._rows),
            filtered=bool(self.query_one("#search", _SearchInput).value.strip()),
            total_sessions=len(self._rendered_sessions),
        )
        self._sessions = []
        async for batch in self._state_store.stream_session_metas(kind="workflow" if self._workflow_mode else "chat"):
            self._sessions.extend(batch)
        self._loading = False
        self._rendered_sessions = list(self._sessions)
        self._render_table(
            preferred_cursor_row=preferred_cursor_row,
            preferred_session_id=preferred_session_id,
        )
        self._cursor_initialized = True

    def _render_table(
        self,
        preferred_cursor_row: int | None = None,
        preferred_session_id: str | None = None,
        *,
        prefer_live_selection: bool = True,
        source_sessions: list[SessionMeta] | None = None,
    ) -> None:
        """Rebuild the table from the in-memory metas (sort + filter + tree)."""
        table = self.query_one("#sessions", _SessionTable)
        query = self.query_one("#search", _SearchInput).value
        previous_selected = self._get_selected_session_id()

        render_source = source_sessions
        if render_source is None:
            render_source = self._rendered_sessions

        build_rows = build_workflow_rows if self._workflow_mode else build_session_rows
        rows = build_rows(
            render_source,
            sort_column=self._sort_column,
            sort_reverse=self._sort_reverse,
            query=query,
            render_message=self._render_message,
        )
        self._rows = rows
        self._session_ids = [row.key for row in rows]

        table.clear(columns=True)
        max_depth = max((row.depth for row in rows), default=0)
        for column in self._columns:
            label = self._render_message(column.label.bind())
            if column.key == self._sort_column:
                label += " ↓" if self._sort_reverse else " ↑"
            width = column.width
            if width is not None:
                if column.key == "id":
                    width += _TREE_LEVEL_WIDTH * max_depth
                width = max(width, cell_len(label))
            table.add_column(Text(label), width=width, key=column.key)

        highlight = query.strip()
        match_style = self._match_style()
        row_tooltips: dict[int, Text] = {}
        for row_index, row in enumerate(rows):
            table.add_row(*self._row_texts(row, highlight, match_style), key=row.key)
            row_tooltips[row_index] = self._row_tooltip(row)
        table.set_row_tooltips(row_tooltips)

        self._restore_cursor(
            previous_selected,
            preferred_cursor_row,
            preferred_session_id,
            prefer_live_selection=prefer_live_selection,
        )
        self._rendered_sessions = list(render_source)
        has_rows = bool(rows)
        self.query_one("#resume", Button).disabled = not has_rows
        self.query_one("#delete", Button).disabled = not has_rows
        self._update_chrome(visible_rows=len(rows), filtered=bool(highlight), total_sessions=len(render_source))

    def _match_style(self) -> Style:
        """Search-match highlight style, themed to the warning color."""
        warning = self.app.theme_variables.get("warning", "yellow")
        return rich_style_from_textual_color(warning, reverse=True)

    def _row_texts(self, row: SessionRow, highlight: str, match_style: Style) -> list[Text]:
        """Cell renderables for one row, with tree guides and match marks."""
        texts: list[Text] = []
        for column in self._columns:
            value = Text(row.cells[column.key], justify="right" if column.numeric else "left")
            if highlight:
                value.highlight_words([highlight], match_style, case_sensitive=False)
                if not row.matched:
                    # Ancestor kept only to anchor a matching fork subtree.
                    value.stylize("dim")
                elif row.prompt_only:
                    # Match lives in prompt text — nothing in the row
                    # highlights, so italics flag it (tooltip has the
                    # matched context).
                    value.stylize("italic")
            if column.key == "id" and row.tree_prefix:
                value = Text(row.tree_prefix, style="dim").append_text(value)
            texts.append(value)
        return texts

    def _row_tooltip(self, row: SessionRow) -> Text:
        """Hover tooltip, one labelled line per fact.

        The agent profile lives only here since its column was dropped
        (rarely distinguishing between rows).
        """
        meta = row.meta
        if row.workflow is not None:
            parts = [
                self._render_message(_TOOLTIP_WORKFLOW.bind(workflow=row.workflow.title)),
                self._render_message(_TOOLTIP_STATUS.bind(status=row.cells["status"])),
                self._render_message(_TOOLTIP_RUN.bind(run_id=row.workflow.run_id)),
            ]
        else:
            assert meta.kind == "chat"
            parts = [
                self._render_message(_TOOLTIP_TITLE.bind(title=title_display(meta))),
                self._render_message(_TOOLTIP_AGENT.bind(agent=profile_display(meta))),
            ]
        if row.prompt_snippet:
            parts.insert(1, self._render_message(_TOOLTIP_PROMPT_MATCH.bind(prompt=row.prompt_snippet)))
        if meta.primary_cwd:
            parts.append(self._render_message(_TOOLTIP_DIRECTORY.bind(path=DisplayPath(meta.primary_cwd))))
        if meta.kind == "chat":
            parts.append(self._render_message(_TOOLTIP_TURNS.bind(turns=meta.turn_count)))
            parts.append(self._render_message(_TOOLTIP_TOTAL_TOKENS.bind(tokens=format_tokens(meta.total_tokens))))
        updated = row.workflow.updated_at if row.workflow is not None else meta.updated_at
        parts.append(
            self._render_message(_TOOLTIP_LAST_INTERACTION.bind(interaction=last_interaction_display(updated)))
        )
        if meta.kind == "chat" and meta.parent_session_id:
            parts.append(
                self._render_message(_TOOLTIP_FORKED_FROM.bind(session_id=session_short_id(meta.parent_session_id)))
            )
        return Text("\n".join(parts))

    def _restore_cursor(
        self,
        previous_selected: str | None,
        preferred_cursor_row: int | None,
        preferred_session_id: str | None,
        *,
        prefer_live_selection: bool,
    ) -> None:
        """Keep the highlighted session stable across re-renders.

        The jump to the currently loaded session happens at most once, on
        the initial open (``_cursor_initialized``) — a later reload (e.g.
        after a delete) must not yank the cursor back to it.  Delete refreshes
        prefer a concrete neighbor session id, with row number only as a last
        fallback if that neighbor disappeared during the reload.  Final
        reload renders prefer live user selection; optimistic delete renders
        force the precomputed neighbor for immediate visual stability.
        """
        table = self.query_one("#sessions", _SessionTable)
        if table.row_count == 0:
            return
        target_id = previous_selected if previous_selected in self._session_ids else None
        if preferred_session_id is not None:
            if (not prefer_live_selection or target_id is None) and preferred_session_id in self._session_ids:
                target_id = preferred_session_id
            elif not prefer_live_selection:
                target_id = None
            if target_id is not None:
                table.move_cursor(row=self._session_ids.index(target_id))
            elif not self._loading and preferred_cursor_row is not None:
                table.move_cursor(row=min(preferred_cursor_row, table.row_count - 1))
            else:
                table.move_cursor(row=0)
            return

        if target_id is None and not self._cursor_initialized:
            target_id = next((row.key for row in self._rows if row.meta.session_id == self._current_session_id), None)
            self._cursor_initialized = True
        if target_id is not None:
            table.move_cursor(row=self._session_ids.index(target_id))
        elif preferred_cursor_row is not None:
            table.move_cursor(row=min(preferred_cursor_row, table.row_count - 1))
        else:
            table.move_cursor(row=0)

    def _update_chrome(self, *, visible_rows: int, filtered: bool, total_sessions: int) -> None:
        """Empty-state visibility plus the live count in the border."""
        show_loading = self._loading and total_sessions == 0
        has_sessions = total_sessions > 0
        container = self.query_one("#container")
        if has_sessions or show_loading:
            container.remove_class("-empty")
        else:
            container.add_class("-empty")
        self.query_one("#sessions-loading-state").display = show_loading
        self.query_one("#empty-note").display = not show_loading and not has_sessions
        self.query_one("#sessions").display = not show_loading and has_sessions
        self.query_one("#footer").display = not show_loading and has_sessions

        if show_loading:
            container.border_subtitle = Text(self._render_message(_LOADING_SESSIONS.bind()))
        elif has_sessions:
            count = f"{visible_rows}/{total_sessions}" if filtered else f"{total_sessions}"
            label = _WORKFLOW_SESSION_COUNT if self._workflow_mode else _SESSION_COUNT
            container.border_subtitle = Text(self._render_message(label.bind(count_text=count)))
        else:
            container.border_subtitle = ""

    # ------------------------------------------------------------------
    # Sorting & searching
    # ------------------------------------------------------------------

    @on(DataTable.HeaderSelected, "#sessions")
    def _on_header_selected(self, event: DataTable.HeaderSelected) -> None:
        """Click a column header — sort by it; click again to flip direction."""
        key = event.column_key.value
        if key is None:
            return
        column = column_by_key(str(key), workflow=self._workflow_mode)
        if self._sort_column == column.key:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = column.key
            self._sort_reverse = column.default_reverse
        self._render_table()

    @on(Input.Changed, "#search")
    def _on_search_changed(self, event: Input.Changed) -> None:
        """Debounce filter re-renders while the user is still typing."""
        if self._search_timer is not None:
            self._search_timer.stop()
        self._search_timer = self.set_timer(_SEARCH_DEBOUNCE_SECONDS, self._render_table)

    @on(Input.Submitted, "#search")
    def _on_search_submitted(self, event: Input.Submitted) -> None:
        """Enter in the search box — jump to the filtered results."""
        self.query_one("#sessions", _SessionTable).focus()

    @on(_SearchInput.Escaped)
    def _on_search_escaped(self, event: _SearchInput.Escaped) -> None:
        """Escape on an empty search box — return focus to the table."""
        self.query_one("#sessions", _SessionTable).focus()

    # ------------------------------------------------------------------
    # Selection & dismissal
    # ------------------------------------------------------------------

    def _get_selected_session_id(self) -> str | None:
        """Return the session ID stored as the highlighted row key."""
        table = self.query_one("#sessions", DataTable)
        if table.row_count == 0:
            return None
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        if row_key is not None and row_key.value is not None:
            return str(row_key.value)
        return None

    def _resume_selected(self) -> None:
        key = self._get_selected_session_id()
        row = next((row for row in self._rows if row.key == key), None)
        if row is not None:
            self.dismiss(WorkflowSessionPick(row.meta.session_id) if row.workflow is not None else row.meta.session_id)

    @on(DataTable.RowHighlighted)
    def _on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Enable buttons when a row is highlighted."""
        self.query_one("#resume", Button).disabled = False
        self.query_one("#delete", Button).disabled = False

    @on(DataTable.RowSelected)
    def _on_row_selected(self, event: DataTable.RowSelected) -> None:
        """Double-click or Enter on a row — resume that session."""
        self._resume_selected()

    @on(Button.Pressed, "#resume")
    def _on_resume(self, event: Button.Pressed) -> None:
        self._resume_selected()

    @on(Button.Pressed, "#cancel")
    def _on_cancel(self, event: Button.Pressed) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#delete")
    def _on_delete_button(self, event: Button.Pressed) -> None:
        self.action_delete_session()

    def action_delete_session(self) -> None:
        """Delete key or Delete button — confirm removing the highlighted session."""
        session_id = self._get_selected_session_id()
        if session_id:
            table = self.query_one("#sessions", DataTable)
            selected_row = table.cursor_coordinate.row
            session_ids = list(self._session_ids)
            preferred_session_id = _next_session_id_after_delete(session_ids, selected_row)
            preferred_cursor_row = _next_cursor_row_after_delete(selected_row, len(session_ids) - 1)

            from chrys.app.tui.screens.dialogs.confirm import ConfirmDialog

            dialog = ConfirmDialog(
                title=self._render_message(_DELETE_SESSION_TITLE.bind()),
                message=self._render_message(_DELETE_SESSION_MESSAGE.bind(session_id=session_short_id(session_id))),
                confirm_label=self._render_message(_DELETE.bind()),
                confirm_variant="error",
                locale_controller=self._locale_controller,
            )
            self.app.push_screen(  # ty: ignore[no-matching-overload]  # Textual's callback overload rejects the work-decorated handler result.
                dialog,
                callback=lambda confirmed: (
                    self._do_delete(
                        session_id,
                        preferred_cursor_row=preferred_cursor_row,
                        preferred_session_id=preferred_session_id,
                    )
                    if confirmed
                    else None
                ),
            )

    def _render_message(self, reference: MessageRef) -> str:
        return render_str(widget_localizer(self), reference)

    @work(thread=False)
    async def _do_delete(
        self,
        session_id: str,
        *,
        preferred_cursor_row: int | None,
        preferred_session_id: str | None,
    ) -> None:
        """Delete a session from the store and refresh the table."""
        if session_id == self._current_session_id:
            # Don't delete here — let MainScreen handle it via the engine
            # so shutdown() won't re-save the file.
            self.dismiss(f"{self.DELETE_AND_NEW_PREFIX}{session_id}")
        else:
            try:
                await self._state_store.delete_session(session_id)
            except TimeoutError:
                self.notify(
                    render_str(
                        widget_localizer(self),
                        _SESSION_OPEN_ELSEWHERE.bind(app_name=APP_DISPLAY_NAME),
                    ),
                    title=render_str(
                        widget_localizer(self),
                        _DELETE_SESSION_TITLE.bind(),
                    ),
                    severity="warning",
                    timeout=4,
                    markup=False,
                )
                return
            optimistic_sessions = [meta for meta in self._rendered_sessions if meta.session_id != session_id]
            self._render_table(
                preferred_cursor_row=preferred_cursor_row,
                preferred_session_id=preferred_session_id,
                prefer_live_selection=False,
                source_sessions=optimistic_sessions,
            )
            self._load_sessions(
                preferred_cursor_row=preferred_cursor_row,
                preferred_session_id=preferred_session_id,
            )

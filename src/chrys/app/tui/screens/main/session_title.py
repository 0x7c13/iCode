# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The session title on the chat border and the terminal tab, and its editing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from chrys.app.tui.terminal.title import TerminalTabText, set_app_terminal_title
from chrys.app.tui.util.spinner import SPINNER_FRAMES, STATIC_SPINNER

if TYPE_CHECKING:
    from pathlib import Path

    from textual.timer import Timer

    from chrys.app.tui.i18n import LocaleController
    from chrys.app.tui.screens.main.state import RunState

_ACTIVITY_INTERVAL_SECONDS = 0.3
_COMPLETED_MARK = "✓"
_FAILED_MARK = "✗"
# The live run waits for the user: an answer or approval, or a retry decision.
QUESTION_MARK: Final = "?"
RETRY_MARK: Final = "!"

type AttentionMark = Literal["", "?", "!"]
type WorkflowTabPhase = Literal["idle", "running", "awaiting_retry", "question", "completed", "failed"]

# Phases of a live workflow run: the tab polls while one is shown.
_LIVE_WORKFLOW_PHASES: frozenset[WorkflowTabPhase] = frozenset({"running", "awaiting_retry", "question"})
_WORKFLOW_PHASE_MARKS: dict[WorkflowTabPhase, str] = {
    "awaiting_retry": RETRY_MARK,
    "question": QUESTION_MARK,
    "completed": _COMPLETED_MARK,
    "failed": _FAILED_MARK,
}


@dataclass(frozen=True, slots=True)
class WorkflowTab:
    """What the terminal tab shows in workflow mode.

    ``title`` is the workflow's title; when it is empty the tab shows ``cwd``,
    the workflow project directory. ``phase`` is the state of the run this
    screen launched last (``idle`` before one, or after it was cancelled).
    """

    title: str
    cwd: str
    phase: WorkflowTabPhase = "idle"


class SessionTitleController:
    """Own the session title and project it onto the chat border and the terminal tab.

    Three titles feed the display title on the border: ``custom`` is user-set
    and wins; ``generated`` is the latest post-turn LLM summary; ``fallback``
    mirrors the persisted first-user-message title so the border has something
    before a summary lands. The terminal tab shows a submitted prompt's preview
    until the titles or the cwd change, which bring back the display title (or
    the cwd when there is none); a custom title pins the tab, so prompt
    previews never replace it. A spinner prefixes the tab while the agent runs
    (``?`` or ``!`` instead while the run waits for the user: ``attention``),
    and the last turn's result mark after it ends.
    In workflow mode the tab shows the workflow's ``WorkflowTab`` instead; the
    chat state keeps updating underneath and comes back when it is cleared.
    The session handler persists titles; this controller only displays them.
    """

    def __init__(
        self,
        *,
        run: RunState,
        app: Callable[[], object],
        workspace_cwd: Callable[[], str],
        show_display_title: Callable[[str], None],
        set_interval: Callable[[float, Callable[[], None]], Timer],
        current_session_id: Callable[[], str],
        push_screen: Callable[[object, Callable[[str | None], None]], object],
        start_custom_title_save: Callable[[str, str], object],
        session_dir: Callable[[str], Path | None],
        animations_enabled: Callable[[], bool],
        attention: Callable[[], AttentionMark],
        locale_controller: LocaleController | None = None,
    ) -> None:
        self._run = run
        self._app = app
        self._workspace_cwd = workspace_cwd
        self._show_display_title = show_display_title
        self._set_interval = set_interval
        self._current_session_id = current_session_id
        self._push_screen = push_screen
        self._start_custom_title_save = start_custom_title_save
        self._session_dir = session_dir
        self._animations_enabled = animations_enabled
        self._attention = attention
        self._locale_controller = locale_controller
        self._custom = ""
        self._generated = ""
        self._fallback = ""
        # Cleaned when set, so each repaint only adds the mark. The chat tab
        # shows the workspace cwd until something sets it.
        self._chat_tab_text: TerminalTabText | None = None
        self._workflow_tab_text: TerminalTabText | None = None
        self._result_mark = ""
        self._activity_frame = 0
        self._activity_timer: Timer | None = None
        self._workflow_tab: WorkflowTab | None = None
        self._written_title: str | None = None

    # ------------------------------------------------------------------ #
    # Session title
    # ------------------------------------------------------------------ #

    @property
    def custom_title(self) -> str:
        return self._custom

    @property
    def display_title(self) -> str:
        """User-facing session title: custom wins, then generated, then fallback."""
        return self._custom or self._generated or self._fallback

    def set_session_title_state(
        self,
        *,
        custom: str | None = None,
        generated: str | None = None,
        fallback: str | None = None,
    ) -> None:
        """Update the titles (``None`` leaves a field unchanged) and refresh."""
        if custom is not None:
            self._custom = custom
        if generated is not None:
            self._generated = generated
        if fallback is not None:
            self._fallback = fallback
        self._refresh_display_title()

    def reset_session_title_state(self) -> None:
        """Clear every title and the result mark (new or blank session)."""
        self._result_mark = ""
        self._custom = ""
        self._generated = ""
        self._fallback = ""
        self._refresh_display_title()

    def _refresh_display_title(self) -> None:
        self._show_display_title(self.display_title)
        self.set_terminal_title_for_cwd()

    def open_editor(self) -> None:
        """Push the custom-title dialog for the current session (border click or /rename)."""
        session_id = self._current_session_id()
        if not session_id:
            return
        from chrys.app.tui.screens.dialogs.session_title import SessionTitleDialog

        dialog = SessionTitleDialog(
            session_id=session_id,
            session_folder=self._session_dir(session_id),
            custom_title=self._custom,
            auto_title=self._generated or self._fallback,
            locale_controller=self._locale_controller,
        )

        def on_result(result: str | None) -> None:
            # Pin the edit to the session the dialog was opened for — the
            # UI may have restored another session while it was open.
            if result is not None:
                self._start_custom_title_save(result, session_id)

        self._push_screen(dialog, on_result)

    def apply_custom_title(self, custom_title: str) -> None:
        """Apply a non-empty ``/rename <title>`` argument without the dialog."""
        session_id = self._current_session_id()
        if session_id:
            self._start_custom_title_save(custom_title, session_id)

    # ------------------------------------------------------------------ #
    # Terminal tab
    # ------------------------------------------------------------------ #

    def set_terminal_title_for_cwd(self, cwd: str | None = None) -> None:
        display = self.display_title
        if display:
            # A session title pins the terminal tab; cwd changes (workspace
            # updates, restores) must not unpin it. The tab falls back to
            # the cwd once the title state clears.
            self._chat_tab_text = TerminalTabText.for_message(display)
        else:
            self._chat_tab_text = TerminalTabText.for_cwd(cwd or self._workspace_cwd())
        self._render_terminal_title()

    def set_terminal_title_for_user_message(self, text: str) -> None:
        self._result_mark = ""
        if not self._fallback and text.strip():
            # Mirror the persisted first-user-message title so the border
            # shows a title before the first auto-summary lands.
            self._fallback = " ".join(text.split())
            self._refresh_display_title()
        if self._custom:
            # A custom title pins the terminal tab; prompt previews must
            # not replace it.
            self._chat_tab_text = TerminalTabText.for_message(self._custom)
        else:
            self._chat_tab_text = TerminalTabText.for_message(text)
        self._render_terminal_title()

    def mark_terminal_title_completed(self) -> None:
        self._result_mark = _COMPLETED_MARK
        if not self._run.agent_running:
            self._render_terminal_title()

    def mark_terminal_title_failed(self) -> None:
        self._result_mark = _FAILED_MARK
        if not self._run.agent_running:
            self._render_terminal_title()

    def clear_terminal_title_result(self) -> None:
        if not self._result_mark:
            return
        self._result_mark = ""
        self._render_terminal_title()

    def set_workflow_tab(self, tab: WorkflowTab | None) -> None:
        """Show workflow mode's tab; ``None`` brings the chat tab back."""
        if tab == self._workflow_tab:
            return
        if tab is not None and tab.phase in _LIVE_WORKFLOW_PHASES and not self._live:
            self._activity_frame = 0
        self._workflow_tab = tab
        if tab is None:
            self._workflow_tab_text = None
        else:
            self._workflow_tab_text = (
                TerminalTabText.for_message(tab.title) if tab.title else TerminalTabText.for_cwd(tab.cwd)
            )
        self.sync_activity()

    def run_started(self) -> None:
        """A new turn drops the last result mark and restarts the spinner."""
        self._result_mark = ""
        self._activity_frame = 0

    def sync_activity(self) -> None:
        """Poll the tab exactly while a run is live, then repaint it.

        Each tick advances the spinner and picks up waiting marks, which
        change without telling this controller.
        """
        if self._live:
            if self._activity_timer is None:
                self._activity_timer = self._set_interval(_ACTIVITY_INTERVAL_SECONDS, self._advance_activity)
        else:
            self.stop_activity()
        self._render_terminal_title()

    def stop_activity(self) -> None:
        """Stop the spinner's timer; the titles and the result mark stay."""
        if self._activity_timer is not None:
            self._activity_timer.stop()
            self._activity_timer = None

    def _advance_activity(self) -> None:
        if not self._live or self._activity_timer is None:
            return
        if self._animations_enabled() and not self._waiting_mark():
            self._activity_frame = (self._activity_frame + 1) % len(SPINNER_FRAMES)
        self._render_terminal_title()

    @property
    def _live(self) -> bool:
        """A run is live: the shown workflow tab's, else the chat agent's."""
        if self._workflow_tab is not None:
            return self._workflow_tab.phase in _LIVE_WORKFLOW_PHASES
        return self._run.agent_running

    def _waiting_mark(self) -> str:
        """``!`` or ``?`` while the live run waits for the user; ``!`` wins."""
        marks: set[str] = {self._attention()}
        if self._workflow_tab is not None:
            marks.add(_WORKFLOW_PHASE_MARKS.get(self._workflow_tab.phase, ""))
        if RETRY_MARK in marks:
            return RETRY_MARK
        return QUESTION_MARK if QUESTION_MARK in marks else ""

    @property
    def _indicator(self) -> str:
        if self._live:
            if waiting := self._waiting_mark():
                return waiting
            return SPINNER_FRAMES[self._activity_frame] if self._animations_enabled() else STATIC_SPINNER
        if self._workflow_tab is not None:
            return _WORKFLOW_PHASE_MARKS.get(self._workflow_tab.phase, "")
        return self._result_mark

    def _terminal_title(self) -> str:
        text = self._workflow_tab_text if self._workflow_tab_text is not None else self._chat_tab_text
        if text is None:
            text = TerminalTabText.for_cwd(self._workspace_cwd())
        return text.with_mark(self._indicator)

    def _render_terminal_title(self) -> None:
        # Write only changes: the workflow side re-sends the tab on every
        # run event, and the poll ticks while nothing changes.
        title = self._terminal_title()
        if title == self._written_title:
            return
        if set_app_terminal_title(self._app(), title):
            self._written_title = title

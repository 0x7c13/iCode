# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""ConnectionTestDialog — modal shown while testing an external connection.

Shared by the agent-config Test buttons: MCP servers and ACP agents both
run their one-shot connection tests behind this dialog (``subject_label``
names the subject in the border captions).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual import on
from textual.binding import Binding
from textual.containers import HorizontalGroup, VerticalGroup
from textual.widgets import Button, LoadingIndicator, Static

from chrys.app.tui.i18n import render_str, widget_localizer
from chrys.app.tui.screens.dialogs.base import BaseDialog
from chrys.app.tui.widgets import DialogButtonRow, DialogButtonSpec, StableAutoHeightScroll
from chrys.app.tui.widgets.markdown import VirtualizedMarkdown
from chrys.foundation.i18n import MessageRef, msg

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.geometry import Size


_TESTING_CONNECTION = msg(
    "tui.connection_test.testing",
    fallback="Testing {subject} Connection",
)
_OK = msg("tui.connection_test.button.ok", fallback="Ok")
_SUCCESS = msg("tui.connection_test.success", fallback="Connection Successful")
_FAILED = msg("tui.connection_test.failed", fallback="Connection Failed")
_MCP_SERVER = msg("tui.connection_test.default_subject", fallback="MCP Server")
_DEFAULT_MCP_SERVER = _MCP_SERVER.bind()


class _ResultMarkdown(VirtualizedMarkdown, can_focus=False):
    """Markdown result body that lets the wrapping dialog scroll own scrolling.

    Mirroring the ask-user dialog pattern: the widget grows to its virtual
    height so the ``StableAutoHeightScroll`` wrapper scrolls it — that keeps
    the scrollbar flush with the container edge while the markdown's own
    horizontal padding insets the content.
    """

    def watch_virtual_size(self, _old: Size, new: Size) -> None:
        if new.height > 0:
            self.styles.height = new.height


class ConnectionTestDialog(BaseDialog[None]):
    """Modal dialog shown during an external connection test (MCP or ACP).

    Lifecycle:
    - Opens with a ``LoadingIndicator`` and no buttons (the user cannot
      dismiss while the test runs — Esc is a no-op).
    - Once the caller invokes :meth:`set_result`, the spinner is replaced
      by the result message and an ``Ok`` button appears.  Esc only
      dismisses when ``allow_esc=True`` is passed (timeout case);
      otherwise the user must explicitly click ``Ok``.
    """

    BINDINGS: ClassVar[list] = [
        Binding("escape", "dismiss_if_allowed", show=False, priority=True),
    ]

    CSS_PATH = "connection_test.tcss"

    def __init__(self, server_name: str = "", *, subject_label: MessageRef | str = _DEFAULT_MCP_SERVER) -> None:
        self._server_name = server_name.strip()
        self._subject_label = subject_label
        self._resolved = False
        self._esc_allowed = False
        self._mounted = False
        self._pending_result: tuple[bool, str, bool, bool] | None = None
        super().__init__()

    def compose(self) -> ComposeResult:
        localizer = widget_localizer(self)
        subject = (
            self._subject_label if isinstance(self._subject_label, str) else render_str(localizer, self._subject_label)
        )
        with VerticalGroup(id="connection-test-container") as container:
            # Identity rides the top border; the lifecycle state (testing →
            # success/failure) rides the bottom one.
            container.border_title = Text(self._server_name or subject)
            container.border_subtitle = Text(render_str(localizer, _TESTING_CONNECTION.bind(subject=subject)))
            with VerticalGroup(id="connection-test-inner"):
                yield LoadingIndicator(id="connection-test-loading")
                message = Static("", id="connection-test-message", markup=False)
                message.display = False
                yield message
            buttons = DialogButtonRow(
                DialogButtonSpec(Text(render_str(localizer, _OK.bind())), id="connection-test-ok", variant="primary"),
                id="connection-test-buttons",
            )
            buttons.display = False
            yield buttons

    def on_mount(self) -> None:
        self._mounted = True
        if self._pending_result is not None:
            success, message, allow_esc, markdown = self._pending_result
            self._pending_result = None
            self._apply_result(success, message, allow_esc, markdown)

    def set_result(self, success: bool, message: str, allow_esc: bool = False, *, markdown: bool = False) -> None:
        """Swap the loading state for a final result + Ok button.

        ``allow_esc`` enables Esc-to-dismiss for the result state — pass
        ``True`` only for timeout outcomes; otherwise the user must
        explicitly click ``Ok``.

        ``markdown`` renders ``message`` as Markdown in a scrollable body and
        widens the dialog — used for structured reports such as the ACP
        handshake summary. Plain results keep the compact text layout.

        Safe to call before the dialog has finished mounting — the result is
        buffered and applied in :meth:`on_mount`.
        """
        if self._resolved:
            return
        self._resolved = True

        if not self._mounted:
            self._pending_result = (success, message, allow_esc, markdown)
            return

        self._apply_result(success, message, allow_esc, markdown)

    def _apply_result(self, success: bool, message: str, allow_esc: bool, markdown: bool) -> None:
        self._esc_allowed = allow_esc

        if markdown:
            # The Markdown body replaces the whole inner group (spinner and
            # plain-text message included) and owns its own scrolling; the
            # ``-markdown`` class widens the container to report size.
            self.add_class("-markdown")
            self.query_one("#connection-test-inner", VerticalGroup).remove()
            container = self.query_one("#connection-test-container", VerticalGroup)
            container.mount(
                StableAutoHeightScroll(
                    _ResultMarkdown(message, id="connection-test-markdown", open_links=False),
                    id="connection-test-markdown-scroll",
                ),
                before="#connection-test-buttons",
            )
        else:
            loading = self.query_one("#connection-test-loading", LoadingIndicator)
            loading.display = False
            loading.remove()

            message_widget = self.query_one("#connection-test-message", Static)
            message_widget.update(message)
            message_widget.display = True

        buttons = self.query_one("#connection-test-buttons", HorizontalGroup)
        buttons.display = True

        container = self.query_one("#connection-test-container", VerticalGroup)
        localizer = widget_localizer(self)
        container.border_subtitle = Text(render_str(localizer, (_SUCCESS if success else _FAILED).bind()))

        self.add_class("-success" if success else "-error")
        self.query_one("#connection-test-ok", Button).focus()

    @on(Button.Pressed, "#connection-test-ok")
    def _on_ok(self, event: Button.Pressed) -> None:
        self.dismiss(None)

    def action_dismiss_if_allowed(self) -> None:
        if self._resolved and self._esc_allowed:
            self.dismiss(None)

    def _allow_click_outside_dismiss(self) -> bool:
        return self._resolved and self._esc_allowed

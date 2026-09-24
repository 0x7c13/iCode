# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Rich renderer for provider-hosted MCP calls."""

from __future__ import annotations

from chrys.app.tui.widgets.chat.renderers.hosted_generic import (
    HOSTED_ARGUMENTS,
    HOSTED_OUTPUT,
    HostedToolCall,
    _bounded_json,
    _bounded_text,
    _display_text,
    _first_value,
    _hosted_card_css,
)
from chrys.foundation.i18n import msg
from chrys.foundation.i18n.formatting import sanitize_legacy_block, sanitize_legacy_scalar

_HOSTED_MCP_EMPTY = msg("tui.hosted.mcp.empty", fallback="Output: No output")


class HostedMcpToolCall(HostedToolCall):
    """Hosted MCP card with server, tool, arguments, and output."""

    DEFAULT_CSS = _hosted_card_css("HostedMcpToolCall")

    def _label_details(self) -> list[str]:
        server = _first_value(self.args, "server", "server_name")
        return [sanitize_legacy_scalar(_bounded_text(_display_text(server)))] if server is not None else []

    def _format_complete_details(self, result: str) -> str:
        sections: list[str] = []
        if self.args:
            sections.append(
                f"{self._render_message(HOSTED_ARGUMENTS.bind())} {sanitize_legacy_block(_bounded_json(self.args))}"
            )
        sections.append(
            f"{self._render_message(HOSTED_OUTPUT.bind())}\n{sanitize_legacy_block(_bounded_text(result))}"
            if result
            else self._render_message(_HOSTED_MCP_EMPTY.bind())
        )
        return "\n\n".join(sections)

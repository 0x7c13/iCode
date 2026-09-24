# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Rich renderer for provider-hosted code execution."""

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
)
from chrys.foundation.i18n import msg
from chrys.foundation.i18n.formatting import sanitize_legacy_block

_HOSTED_CODE_TITLE = msg("tui.hosted.code.title", fallback="code")
_HOSTED_CODE_INPUT = msg("tui.hosted.code.input", fallback="Code/input:")
_HOSTED_CODE_STDOUT = msg("tui.hosted.code.stdout", fallback="stdout:")
_HOSTED_CODE_STDERR = msg("tui.hosted.code.stderr", fallback="stderr:")
_HOSTED_CODE_IMAGES = msg("tui.hosted.code.images", fallback="Images: {n}")
_HOSTED_CODE_EMPTY = msg("tui.hosted.code.empty", fallback="No code output")


def _code_value(args: Mapping[str, Any]) -> str:
    return _display_text(_first_value(args, "code", "input", "command"))


class HostedCodeToolCall(HostedToolCall):
    """Hosted code card with input, streams, files, and image previews."""

    DEFAULT_CSS = _hosted_card_css("HostedCodeToolCall")

    def _border_title_text(self) -> str:
        """Keep code arguments in the body instead of duplicating them in the border."""
        return self._render_message(_HOSTED_CODE_TITLE.bind())

    def _label_details(self) -> list[str]:
        """The ``provider/code`` label is sufficient for the collapsed summary."""
        return []

    def _format_complete_details(self, result: str) -> str:
        sections: list[str] = []
        code = _code_value(self.args)
        if code:
            sections.append(
                f"{self._render_message(_HOSTED_CODE_INPUT.bind())}\n{sanitize_legacy_block(_bounded_text(code))}"
            )
        stdout = _first_value(self.metadata or {}, "stdout")
        stderr = _first_value(self.metadata or {}, "stderr")
        if stdout is not None:
            sections.append(
                f"{self._render_message(_HOSTED_CODE_STDOUT.bind())}\n"
                f"{sanitize_legacy_block(_bounded_text(_display_text(stdout)))}"
            )
        if stderr is not None:
            sections.append(
                f"{self._render_message(_HOSTED_CODE_STDERR.bind())}\n"
                f"{sanitize_legacy_block(_bounded_text(_display_text(stderr)))}"
            )
        if stdout is None and stderr is None and result:
            sections.append(
                f"{self._render_message(HOSTED_OUTPUT.bind())}\n{sanitize_legacy_block(_bounded_text(result))}"
            )
        artifact_text = _artifact_summary(self.artifacts, self._render_message)
        if artifact_text:
            sections.append(f"{self._render_message(HOSTED_FILES.bind())}\n{artifact_text}")
        if self.image_contents:
            sections.append(self._render_message(_HOSTED_CODE_IMAGES.bind(n=len(self.image_contents))))
        return "\n\n".join(sections) or self._render_message(_HOSTED_CODE_EMPTY.bind())

    def set_complete(self, result: str, duration_ms: int = 0, **kwargs: Any) -> None:
        image_contents = kwargs.get("image_contents")
        self.image_contents = list(image_contents) if isinstance(image_contents, list) else []
        metadata = kwargs.get("metadata")
        self.metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        super().set_complete(result, duration_ms, **kwargs)

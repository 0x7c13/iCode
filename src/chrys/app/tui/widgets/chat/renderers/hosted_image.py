# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Rich renderer for provider-hosted image generation."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from typing import Any

from rich.text import Text
from textual.widgets import Static

from chrys.app.tui.widgets.chat.renderers.hosted_generic import (
    HOSTED_RESULT,
    HostedToolCall,
    _bounded_text,
    _display_text,
    _first_value,
    _hosted_card_css,
)
from chrys.foundation.i18n import msg
from chrys.foundation.i18n.formatting import sanitize_legacy_block, sanitize_legacy_scalar

_HOSTED_IMAGE_PARTIAL_PREVIEW = msg(
    "tui.hosted.image.partial_preview",
    fallback="Partial preview · {count} image",
    plural_fallback="Partial preview · {count} images",
)
_HOSTED_IMAGE_PROMPT = msg("tui.hosted.image.prompt", fallback="Prompt/operation:")
_HOSTED_IMAGE_EMPTY = msg("tui.hosted.image.empty", fallback="No image output")


def _image_subject(args: Mapping[str, Any]) -> str:
    return _display_text(_first_value(args, "prompt", "operation", "action", "input"))


class HostedImageToolCall(HostedToolCall):
    """Hosted image card with non-terminal partial preview refreshes."""

    DEFAULT_CSS = _hosted_card_css("HostedImageToolCall")

    def _label_details(self) -> list[str]:
        subject = _image_subject(self.args)
        return [sanitize_legacy_scalar(_bounded_text(subject))] if subject else []

    def update_progress(self, lines: list[str]) -> None:
        """Refresh partial text/images while preserving the running state."""
        preview = sanitize_legacy_block(_bounded_text("\n".join(lines)))
        if not preview and self.image_contents:
            count = len(self.image_contents)
            preview = self._render_message(_HOSTED_IMAGE_PARTIAL_PREVIEW.bind(count=count))
        self._body_pinned = bool(preview)
        with suppress(Exception):
            body = self.query_one("#tc-body", Static)
            body.display = True
            body.update(Text(preview) if preview else self._render_spinner())
        self._render_image_contents(list(self.image_contents))
        with suppress(Exception):
            self.query_one("#tc-label", Static).update(self._label_text())

    def _format_complete_details(self, result: str) -> str:
        sections: list[str] = []
        subject = _image_subject(self.args)
        if subject:
            sections.append(
                f"{self._render_message(_HOSTED_IMAGE_PROMPT.bind())}\n{sanitize_legacy_block(_bounded_text(subject))}"
            )
        if result:
            sections.append(
                f"{self._render_message(HOSTED_RESULT.bind())}\n{sanitize_legacy_block(_bounded_text(result))}"
            )
        return "\n\n".join(sections) or (
            "" if self.image_contents else self._render_message(_HOSTED_IMAGE_EMPTY.bind())
        )

    def set_complete(self, result: str, duration_ms: int = 0, **kwargs: Any) -> None:
        image_contents = kwargs.get("image_contents")
        self.image_contents = list(image_contents) if isinstance(image_contents, list) else []
        metadata = kwargs.get("metadata")
        self.metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        has_details = bool(self._format_complete_details(result))
        super().set_complete(result, duration_ms, **kwargs)
        with suppress(Exception):
            self.query_one("#tc-body", Static).display = has_details

    def set_error(self, error: str) -> None:
        """Keep the body visible when a previously image-only card fails."""
        super().set_error(error)
        with suppress(Exception):
            self.query_one("#tc-body", Static).display = True

    def restore_error_payload(
        self,
        error: str,
        *,
        image_contents: list[Any],
        metadata: dict[str, Any] | None,
        artifacts: list[dict[str, Any]],
    ) -> None:
        """Restore partial structured output before reapplying the failed state."""
        self.image_contents = list(image_contents)
        self.metadata = dict(metadata or {})
        self.artifacts = list(artifacts)
        self._render_image_contents(self.image_contents)
        self.set_error(error)

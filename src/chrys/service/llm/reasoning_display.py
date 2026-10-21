# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The readable text of a reasoning content, the same for every wire protocol.

Each codec leaves a reasoning content's displayable text in ``Content.text``,
streamed or not: plaintext reasoning (Chat Completions ``reasoning_content`` /
``reasoning``, Responses reasoning text, Anthropic thinking) and summaries
(Responses reasoning summaries, Chat Completions ``reasoning_details``
summaries). A summary carries :data:`REASONING_SUMMARY_KEY` so it is never
shown as the full reasoning. Opaque state — signatures, encrypted payloads,
redacted thinking, ``protected_data`` — has no text and is never shown.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chrys.foundation.models.invocations import ContentChannel
    from chrys.kernel import Content

REASONING_SUMMARY_KEY = "reasoning_summary"
"""Content ``additional_properties`` flag: the text is a summary of the reasoning, not the reasoning itself."""


def readable_reasoning(content: Content) -> tuple[ContentChannel, str] | None:
    """The channel and text a reasoning content shows, or ``None`` when it has nothing readable."""
    if content.type != "text_reasoning" or not content.text:
        return None
    if content.additional_properties.get(REASONING_SUMMARY_KEY):
        return "reasoning_summary", content.text
    return "reasoning", content.text

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""What each protocol's reasoning contents show: their text, on the reasoning or summary channel."""

from __future__ import annotations

import pytest

from chrys.kernel import Content
from chrys.service.llm.reasoning_display import REASONING_SUMMARY_KEY, readable_reasoning


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        # Chat Completions plaintext and Anthropic thinking (with its signature).
        (Content.from_text_reasoning(text="chain"), ("reasoning", "chain")),
        (Content.from_text_reasoning(text="thinking", protected_data="sig"), ("reasoning", "thinking")),
        # Responses reasoning text and summary.
        (
            Content.from_text_reasoning(id="rs", text="private", additional_properties={"reasoning_text": True}),
            ("reasoning", "private"),
        ),
        (
            Content.from_text_reasoning(id="rs", text="visible", additional_properties={REASONING_SUMMARY_KEY: True}),
            ("reasoning_summary", "visible"),
        ),
        # Opaque state: a signature delta, an encrypted backfill, redacted thinking.
        (Content.from_text_reasoning(text=None, protected_data="sig"), None),
        (Content.from_text_reasoning(id="rs", text="", protected_data="encrypted"), None),
        (
            Content.from_text_reasoning(
                text=None, protected_data="opaque", additional_properties={"anthropic_redacted_thinking": True}
            ),
            None,
        ),
        (Content.from_text("answer"), None),
    ],
)
def test_readable_reasoning_shows_text_only_on_its_channel(content: Content, expected: tuple[str, str] | None) -> None:
    assert readable_reasoning(content) == expected

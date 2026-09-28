# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Unlabeled code blocks guess their lexer from a bounded, line-aligned prefix."""

from __future__ import annotations

from typing import Any

import pygments.lexers
import pytest
from pygments.lexer import Lexer
from textual.highlight import highlight

from chrys.app.tui.widgets.markdown.blocks import MarkdownBlock
from chrys.app.tui.widgets.markdown.parser import _create_markdown_parser, _parse_tokens
from chrys.app.tui.widgets.syntax_theme import NoErrorHighlightTheme

_SAMPLE_LIMIT = 2048


@pytest.fixture
def guessed_samples(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every text Pygments is asked to guess a lexer for."""
    samples: list[str] = []
    real_guess = pygments.lexers.guess_lexer

    def recording_guess(text: str, **options: Any) -> Lexer:
        samples.append(text)
        return real_guess(text, **options)

    # textual.highlight.guess_language imports guess_lexer from this module at call time.
    monkeypatch.setattr(pygments.lexers, "guess_lexer", recording_guess)
    return samples


def _only_code_block(markdown: str) -> MarkdownBlock:
    blocks = [
        block for block in _parse_tokens(_create_markdown_parser().parse(markdown)) if block.block_type == "fence"
    ]
    assert len(blocks) == 1
    return blocks[0]


def test_a_long_unlabeled_code_block_guesses_from_a_line_aligned_prefix(guessed_samples: list[str]) -> None:
    code = "\n".join(
        f"2026-09-27 05:{index % 60:02d}:12,345 INFO worker[{index % 7}] processed batch {index} status=ok"
        for index in range(2000)
    )

    block = _only_code_block(f"```\n{code}\n```\n")

    assert len(guessed_samples) == 1
    sample = guessed_samples[0]
    assert 0 < len(sample) <= _SAMPLE_LIMIT
    assert code.startswith(sample)
    assert code[len(sample)] == "\n"
    lexer = pygments.lexers.guess_lexer(sample)
    expected = highlight(code, language=lexer.aliases[0], theme=NoErrorHighlightTheme)
    assert block.content.plain == expected.plain
    assert block.content.spans == expected.spans


def test_a_short_unlabeled_code_block_is_guessed_and_highlighted_as_a_whole(guessed_samples: list[str]) -> None:
    code = "import os\n\n\ndef main() -> str:\n    return os.getcwd()"

    block = _only_code_block(f"```\n{code}\n```\n")

    assert guessed_samples == [code]
    expected = highlight(code, language=None, theme=NoErrorHighlightTheme)
    assert block.content.plain == expected.plain
    assert block.content.spans == expected.spans


def test_a_labeled_code_block_guesses_nothing(guessed_samples: list[str]) -> None:
    code = "\n".join(f"value_{index} = {index}" for index in range(500))

    block = _only_code_block(f"```python\n{code}\n```\n")

    assert guessed_samples == []
    assert block.code_language == "python"

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for shared tool-output cleaning and truncation."""

from __future__ import annotations

import pytest

from chrys.foundation.text.tokenizer import MixedLanguageTokenizer
from chrys.foundation.text.tool_output import (
    process_carriage_returns,
    strip_ansi,
    truncate_content_texts,
    truncate_output,
)

_tokenizer = MixedLanguageTokenizer()


@pytest.mark.parametrize(
    ("text", "budget"),
    [
        ("\n".join(f"line {index}" for index in range(500)), 100),
        ("{" + '"value":"' + "x" * 20_000 + '"}', 100),
        ("中" * 10_000, 100),
        (("中abc" * 5_000), 100),
    ],
    # Explicit ids: the raw values would become 20k-120k char test IDs, and
    # pytest exports the ID via PYTEST_CURRENT_TEST, which Windows caps at
    # 32767 chars.
    ids=["many-lines", "long-json-line", "cjk", "mixed-cjk-ascii"],
)
@pytest.mark.parametrize("suffix", ["", "[Full output saved to: /tmp/result.txt]"])
def test_truncate_output_is_a_hard_bound(text: str, budget: int, suffix: str) -> None:
    result = truncate_output(text, budget, truncation_suffix=suffix)

    assert _tokenizer.count_tokens(result) <= budget
    assert "truncated" in result


@pytest.mark.parametrize("budget", [1, 10, 50])
def test_truncate_output_is_total_for_positive_tiny_budgets(budget: int) -> None:
    result = truncate_output("x" * 10_000, budget, truncation_suffix="[suffix-too-large-for-tiny-budget]")

    assert _tokenizer.count_tokens(result) <= budget


def test_truncate_output_rejects_zero_budget() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        truncate_output("large", 0)


def test_suffix_is_only_added_when_truncation_occurs() -> None:
    assert truncate_output("short", 100, truncation_suffix="[suffix]") == "short"


def test_in_budget_text_is_unchanged_when_suffix_would_not_fit() -> None:
    text = "中" * 132

    result = truncate_output(text, 100, truncation_suffix="[path notice]" * 100)

    assert _tokenizer.count_tokens(text) <= 100
    assert result == text


def test_single_giant_line_preserves_head_and_tail_payload() -> None:
    text = "HEAD" + "x" * 10_000 + "TAIL"
    result = truncate_output(text, 100)

    assert result.startswith("HEAD")
    assert result.endswith("TAIL")
    assert "truncated" in result
    assert _tokenizer.count_tokens(result) <= 100


def test_line_truncation_preserves_complete_lines_and_ratio() -> None:
    lines = [f"line {index}" for index in range(500)]
    result = truncate_output("\n".join(lines), 100)
    payload_lines = [line for line in result.splitlines() if "truncated" not in line]

    assert all(line in lines for line in payload_lines)
    marker_index = next(index for index, line in enumerate(result.splitlines()) if "truncated" in line)
    assert len(result.splitlines()[marker_index + 1 :]) > marker_index


def test_non_additive_join_is_recounted() -> None:
    text = "\n".join(["中" * 10 for _ in range(100)])
    result = truncate_output(text, 50, truncation_suffix="中" * 10)

    assert _tokenizer.count_tokens(result) <= 50


def test_truncate_content_texts_returns_none_when_aggregate_fits() -> None:
    assert truncate_content_texts(["one", "two"], 100) is None


def test_truncate_content_texts_preserves_shape_and_one_marker() -> None:
    texts = ["x" * 31_960, "second item" * 10, "third item" * 10]
    result = truncate_content_texts(texts, 8_000, truncation_suffix="[spill notice]")

    assert result is not None
    assert len(result) == len(texts)
    assert result[0] is not None
    assert result[1:] == [None, None]
    joined = "\n".join(value for value in result if value is not None)
    assert joined.count("truncated") == 1
    assert "[2 further text items omitted.]" in joined
    assert _tokenizer.count_tokens(joined) <= 8_000


def test_first_truncation_slot_has_no_phantom_prefix_cost() -> None:
    result = truncate_content_texts(["中" * 1_000, "tail"], 100)

    assert result is not None
    assert result[0] is not None
    assert _tokenizer.count_tokens(result[0]) > 90
    assert _tokenizer.count_tokens(result[0]) <= 100


def test_multi_item_truncation_forces_omission_footer_when_current_item_fits() -> None:
    first = "中" * 132

    result = truncate_content_texts([first, "x" * 10_000], 100)

    assert result is not None
    assert result[0] is not None
    assert result[0] != first
    assert "truncated" in result[0]
    assert "[1 further text items omitted.]" in result[0]
    assert result[1] is None
    assert _tokenizer.count_tokens(result[0]) <= 100


def test_keep_whole_r9_literal_counterexample_uses_complete_join_count() -> None:
    first = "中" * 25
    suffix = "x" * 255
    marker = f"[...truncated ~{len(first)} chars (estimated {_tokenizer.count_tokens(first)} tokens total)...]"
    fixed_footer = "\n".join((marker, "[1 further text items omitted.]", suffix))

    assert _tokenizer.count_tokens(first) == 14
    assert _tokenizer.count_tokens(fixed_footer) == 85
    assert _tokenizer.count_tokens(first) + _tokenizer.count_tokens("\n") + _tokenizer.count_tokens(fixed_footer) == 100
    assert _tokenizer.count_tokens(f"{first}\n{fixed_footer}") == 101

    result = truncate_content_texts([first, "tail" * 1_000], 100, truncation_suffix=suffix)

    assert result is not None
    assert result[0] is not None
    assert result[0] != first
    assert suffix in result[0]
    assert result[1] is None
    assert _tokenizer.count_tokens(result[0]) <= 100


def test_cleaning_helpers_match_shell_semantics() -> None:
    assert strip_ansi("\x1b[31mred\x1b[0m") == "red"
    assert strip_ansi("\x1b]8;;https://example.com\x1b\\link\x1b]8;;\x1b\\") == "link"
    # The end of a sequence whose start is in other output.
    assert strip_ansi("title\x1b\\rest") == "titlerest"
    assert process_carriage_returns("10%\r50%\r100%\r\ndone") == "100%\ndone"


def test_long_carriage_return_runs_preserve_completed_and_overwritten_lines() -> None:
    # Performance sentinel: this size makes quadratic scanning visible in
    # duration reports, without an assertion tied to the test host's speed.
    carriage_returns = "\r" * 100_000
    assert process_carriage_returns(f"line{carriage_returns}\nnext{carriage_returns}") == "line\n"
    assert process_carriage_returns(f"old{carriage_returns}new") == "new"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("line1\r\r\nline2\r\r\n", "line1\nline2\n", id="pty-crlf"),
        pytest.param("line\r\r\r\n", "line\n", id="multiple-crs-before-newline"),
        pytest.param("title\r\r\n\r\r\nbody\r\r\n", "title\n\nbody\n", id="blank-lines"),
        pytest.param("pty\r\r\nlf\ncrlf\r\n", "pty\nlf\ncrlf\n", id="mixed-line-endings"),
        pytest.param("first\r\r\nlast", "first\nlast", id="unterminated-last-line"),
        pytest.param("10%\r50%\r100%\r\r\ndone", "100%\ndone", id="progress-before-newline"),
        pytest.param("line\r\r\nsomething\r", "line\n", id="bare-trailing-cr-still-overwrites"),
    ],
)
def test_process_carriage_returns_preserves_completed_lines(text: str, expected: str) -> None:
    assert process_carriage_returns(text) == expected

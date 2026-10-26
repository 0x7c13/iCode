# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for MixedLanguageTokenizer."""

from __future__ import annotations

import pytest

from chrys.foundation.text.tokenizer import _CJK_IDEOGRAPH_RANGES, TOKEN_UNITS_PER_TOKEN, MixedLanguageTokenizer

_tokenizer = MixedLanguageTokenizer()


def test_mixed_language_tokenizer_returns_positive():
    """The built-in estimator always returns at least one token."""
    t = MixedLanguageTokenizer()
    assert t.count_tokens("") == 1
    assert t.count_tokens("hello") >= 1
    assert t.count_tokens("你好") >= 1


def test_mixed_language_tokenizer_weights_cjk_more_than_ascii():
    tokenizer = MixedLanguageTokenizer()
    assert tokenizer.count_tokens("你" * 100) > tokenizer.count_tokens("a" * 100)


def test_counts_a_quarter_token_per_character_and_more_per_cjk_ideograph() -> None:
    ideographs = "".join(chr(first) + chr(last) for first, last in _CJK_IDEOGRAPH_RANGES)
    neighbours = "".join(chr(first - 1) + chr(last + 1) for first, last in _CJK_IDEOGRAPH_RANGES)

    assert _tokenizer.count_tokens("a" * 100) == 25
    assert _tokenizer.count_tokens(ideographs * 10) == 60
    assert _tokenizer.count_tokens(neighbours * 10) == 25
    # 7 other characters and 4 ideographs: 1.75 + 2.4 tokens.
    assert _tokenizer.count_tokens("中文 text, 混合") == 4


@pytest.mark.parametrize(
    ("text", "units"), [("", 0), ("a", 25), ("abc", 75), ("汉", 60), ("汉a\n", 110), ("汉a🚀\n", 135)]
)
def test_token_units_preserve_fractional_estimates(text: str, units: int) -> None:
    assert TOKEN_UNITS_PER_TOKEN == 100
    assert _tokenizer.count_token_units(text) == units
    assert _tokenizer.count_tokens(text) == max(1, units // TOKEN_UNITS_PER_TOKEN)


def test_token_units_do_not_depend_on_chunk_boundaries() -> None:
    text = "a汉字\n" + chr(0x20000) + "b"
    assert _tokenizer.count_token_units(text) == 255
    for split in range(len(text) + 1):
        chunks = ("", text[:split], "", text[split:])
        assert sum(_tokenizer.count_token_units(chunk) for chunk in chunks) == 255

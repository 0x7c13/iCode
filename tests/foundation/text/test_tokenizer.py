# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for MixedLanguageTokenizer."""

from __future__ import annotations

from chrys.foundation.text.tokenizer import _CJK_IDEOGRAPH_RANGES, MixedLanguageTokenizer

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

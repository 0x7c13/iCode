# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for MixedLanguageTokenizer."""

from __future__ import annotations

from chrys.foundation.text.tokenizer import MixedLanguageTokenizer

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

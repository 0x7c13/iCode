# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Cache-hit token extraction in the usage middleware."""

from __future__ import annotations

from chrys.service.context.middleware.usage import extract_cache_hit_tokens


def test_extract_returns_none_when_no_cache_field_present() -> None:
    assert extract_cache_hit_tokens({"input_token_count": 100, "output_token_count": 50}) is None


def test_extract_reads_anthropic_field() -> None:
    assert extract_cache_hit_tokens({"anthropic.cache_read_input_tokens": 1234}) == 1234


def test_extract_reads_openai_responses_field() -> None:
    assert extract_cache_hit_tokens({"openai.cached_input_tokens": 777}) == 777


def test_extract_reads_openai_chat_completions_field() -> None:
    """``prompt/cached_tokens`` covers OpenAI Chat Completions and vLLM."""
    assert extract_cache_hit_tokens({"prompt/cached_tokens": 4096}) == 4096


def test_extract_reads_deepseek_field() -> None:
    assert extract_cache_hit_tokens({"deepseek.prompt_cache_hit_tokens": 5000}) == 5000


def test_deepseek_native_field_wins_over_mirrored_chat_completions_field() -> None:
    """DeepSeek can mirror cache hits into ``prompt_tokens_details.cached_tokens``;
    first-match-wins avoids double-counting both keys."""
    usage = {
        "deepseek.prompt_cache_hit_tokens": 10_000,
        "prompt/cached_tokens": 10_000,
    }
    assert extract_cache_hit_tokens(usage) == 10_000


def test_extract_returns_zero_when_provider_reports_zero() -> None:
    """A reported 0 is distinct from absent — UI should show 0, not ``-``."""
    assert extract_cache_hit_tokens({"anthropic.cache_read_input_tokens": 0}) == 0

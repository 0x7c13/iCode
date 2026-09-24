# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Token counting for compaction budget management."""

from __future__ import annotations


class MixedLanguageTokenizer:
    """Fast character-ratio token estimator for mixed CJK and other text."""

    def count_tokens(self, text: str) -> int:
        estimate = 0.0
        for ch in text:
            cp = ord(ch)
            if (
                0x4E00 <= cp <= 0x9FFF  # CJK Unified Ideographs
                or 0x3400 <= cp <= 0x4DBF  # CJK Unified Ideographs Extension A
                or 0x20000 <= cp <= 0x2A6DF  # CJK Unified Ideographs Extension B
                or 0xF900 <= cp <= 0xFAFF  # CJK Compatibility Ideographs
                or 0x2F800 <= cp <= 0x2FA1F  # CJK Compatibility Ideographs Supplement
            ):
                estimate += 0.6
            else:
                estimate += 0.25
        return max(1, int(estimate))

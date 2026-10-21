# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Token counting for compaction budget management."""

from __future__ import annotations

import re

_CJK_IDEOGRAPH_RANGES = (
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x2F800, 0x2FA1F),  # CJK Compatibility Ideographs Supplement
)
_CJK_IDEOGRAPH_RUN = re.compile(
    "[" + "".join(f"{chr(first)}-{chr(last)}" for first, last in _CJK_IDEOGRAPH_RANGES) + "]+"
)


class MixedLanguageTokenizer:
    """Fast character-ratio token estimator for mixed CJK and other text.

    A CJK ideograph counts 0.6 tokens and any other character 0.25.
    """

    def count_tokens(self, text: str) -> int:
        # Counted in C: compaction estimates whole conversations on the event
        # loop, where a per-character Python loop held it for seconds.
        ideographs = sum(map(len, _CJK_IDEOGRAPH_RUN.findall(text)))
        return max(1, (len(text) * 25 + ideographs * 35) // 100)

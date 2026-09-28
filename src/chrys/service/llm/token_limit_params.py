# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Chat Completions request parameter that carries the output-token cap, per provider.

The client classes take their ``TOKEN_LIMIT_PARAM`` from this table and the Models
screen labels the Max Output Tokens field from it, so the label always names what
the client sends. It imports no provider SDK: labelling a form must not load one.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

CHAT_COMPLETIONS_TOKEN_LIMIT_PARAMS: Final[Mapping[str, str]] = MappingProxyType(
    {
        # Real OpenAI hard-rejects the legacy ``max_tokens`` spelling on current models.
        "openai": "max_completion_tokens",
        # These OpenAI-compatible endpoints document only the legacy spelling.
        "deepseek-openai": "max_tokens",
        "glm-openai": "max_tokens",
    }
)

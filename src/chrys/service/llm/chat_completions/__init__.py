# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Chat Completions API clients, for OpenAI and the compatible DeepSeek and GLM endpoints."""

from __future__ import annotations

from .client import ChatCompletionsClient, DeepSeekChatCompletionsClient, GlmChatCompletionsClient

__all__ = ["ChatCompletionsClient", "DeepSeekChatCompletionsClient", "GlmChatCompletionsClient"]

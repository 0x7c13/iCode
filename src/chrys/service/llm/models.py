# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Model registry — known models and their configurations.

TODO: Implement model config registry for the settings panel.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """Information about a known LLM model."""

    provider: str  # "anthropic", "openai", "deepseek-openai", "glm-openai"
    model_id: str
    display_name: str
    context_window: int  # max tokens
    supports_tools: bool = True


# Known models (extend as needed)
KNOWN_MODELS: list[ModelInfo] = [
    ModelInfo("anthropic", "claude-sonnet-4-5-20250929", "Claude Sonnet 4.5", 200_000),
    ModelInfo("anthropic", "claude-opus-4-6", "Claude Opus 4.6", 200_000),
    ModelInfo("openai", "gpt-5.4", "GPT-5.4", 128_000),
    ModelInfo("openai", "o3", "o3", 200_000),
]

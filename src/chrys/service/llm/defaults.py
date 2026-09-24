# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared defaults for Chrys LLM adapters."""

from __future__ import annotations

from typing import Any, Final

# Adapter-level fill-in for requests that reach the wire without a cap
# (Anthropic requires ``max_tokens``).  Profile-driven calls never hit this:
# ``effective_chat_options`` injects the profile's ``max_output_tokens``
# (default ``chrys.service.profiles.models.schema.DEFAULT_MAX_OUTPUT_TOKENS``,
# a deliberately distinct constant) into every live request.
FALLBACK_MAX_OUTPUT_TOKENS: Final[int] = 16 * 1024


def apply_default_max_tokens(run_options: dict[str, Any]) -> None:
    """Set Chrys' fallback output cap unless the request already provides one."""
    if run_options.get("max_tokens"):
        return
    run_options["max_tokens"] = FALLBACK_MAX_OUTPUT_TOKENS

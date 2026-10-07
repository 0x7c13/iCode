# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""What Chrys knows about each provider id, as data.

The client factory reads its credentials and endpoint fallbacks from this
table, the Chat Completions clients take their output-cap parameter from it,
and the Models screen labels its form from it. It imports no provider SDK:
labelling a form must not load one.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

from chrys.service.profiles.models.schema import API_STYLE_CHAT_COMPLETIONS, API_STYLE_RESPONSES

if TYPE_CHECKING:
    from collections.abc import Mapping

type SdkFamily = Literal["openai", "anthropic"]


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """One provider id's SDK, credentials, endpoint and wire-dialect facts."""

    label: str
    """Provider name in configuration errors."""
    sdk: SdkFamily
    """Which provider SDK carries the requests."""
    api_key_env: str
    """Environment variable consulted when the profile carries no API key."""
    base_url_env: str
    """Environment variable consulted when the profile carries no base URL."""
    default_base_url: str
    """The endpoint used when neither the profile nor the environment names one."""
    native_sdk: bool
    """The SDK is the provider's own and applies the default endpoint itself.

    The client factory then leaves the base URL to the SDK instead of passing
    ``default_base_url``.
    """
    api_styles: frozenset[str] | None
    """Wire APIs the provider speaks; None when it ignores ``api_style``."""
    chat_completions_max_output_param: str | None
    """Chat Completions parameter that carries the output-token cap."""


_BOTH_API_STYLES: Final = frozenset({API_STYLE_CHAT_COMPLETIONS, API_STYLE_RESPONSES})

PROVIDERS: Final[dict[str, ProviderSpec]] = {
    "anthropic": ProviderSpec(
        label="Anthropic",
        sdk="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        base_url_env="ANTHROPIC_BASE_URL",
        default_base_url="https://api.anthropic.com",
        native_sdk=True,
        api_styles=None,
        chat_completions_max_output_param=None,
    ),
    "openai": ProviderSpec(
        label="OpenAI",
        sdk="openai",
        api_key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_BASE_URL",
        default_base_url="https://api.openai.com/v1",
        native_sdk=True,
        api_styles=_BOTH_API_STYLES,
        # Real OpenAI hard-rejects the legacy ``max_tokens`` spelling on current models.
        chat_completions_max_output_param="max_completion_tokens",
    ),
    "deepseek-openai": ProviderSpec(
        label="DeepSeek",
        sdk="openai",
        api_key_env="DEEPSEEK_API_KEY",
        base_url_env="DEEPSEEK_BASE_URL",
        default_base_url="https://api.deepseek.com",
        native_sdk=False,
        api_styles=_BOTH_API_STYLES,
        # This OpenAI-compatible endpoint documents only the legacy spelling.
        chat_completions_max_output_param="max_tokens",
    ),
    "glm-openai": ProviderSpec(
        label="GLM",
        sdk="openai",
        api_key_env="ZAI_API_KEY",
        base_url_env="ZAI_BASE_URL",
        default_base_url="https://open.bigmodel.cn/api/paas/v4",
        native_sdk=False,
        api_styles=None,
        # This OpenAI-compatible endpoint documents only the legacy spelling.
        chat_completions_max_output_param="max_tokens",
    ),
}

# Built once at import; the Chat Completions classes bind their parameter from
# it when they are defined, so a later change to PROVIDERS does not reach them.
CHAT_COMPLETIONS_TOKEN_LIMIT_PARAMS: Final[Mapping[str, str]] = MappingProxyType(
    {
        name: spec.chat_completions_max_output_param
        for name, spec in PROVIDERS.items()
        if spec.chat_completions_max_output_param is not None
    }
)

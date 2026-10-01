# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The ``AsyncAnthropic`` client Chrys builds for the Anthropic provider.

Importing this module loads the Anthropic SDK; the client factory imports it
only while it builds such a client.
"""

from __future__ import annotations

from anthropic import AsyncAnthropic

from chrys.service.llm.sdk_retry import DeterministicConnectionRetryGuard


class RetryGuardedAsyncAnthropic(DeterministicConnectionRetryGuard, AsyncAnthropic):
    """``AsyncAnthropic`` that does not retry a connection error that cannot self-heal."""

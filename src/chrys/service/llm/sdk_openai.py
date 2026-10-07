# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The ``AsyncOpenAI`` client Chrys builds for every OpenAI-SDK provider.

Importing this module loads the OpenAI SDK; the client factory imports it only
while it builds such a client.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from chrys.service.llm.sdk_headers import CaseFoldedDefaultHeaders
from chrys.service.llm.sdk_retry import DeterministicConnectionSleepGuard


class RetryGuardedAsyncOpenAI(DeterministicConnectionSleepGuard, CaseFoldedDefaultHeaders, AsyncOpenAI):
    """``AsyncOpenAI`` that does not retry a connection error that cannot self-heal, and sends each header name once."""

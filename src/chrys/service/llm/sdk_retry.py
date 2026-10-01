# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Retry policy shared by the provider SDK client subclasses.

It imports no provider SDK, so each SDK's subclass module loads only its own.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Protocol

from chrys.foundation.errors import is_deterministic_connection_error

if TYPE_CHECKING:

    class _ConnectionRetryBase(Protocol):
        async def _sleep_for_retry(self, **kwargs: Any) -> None: ...

else:
    _ConnectionRetryBase = object


class DeterministicConnectionRetryGuard(_ConnectionRetryBase):
    """Stop provider retries when the active request error cannot self-heal.

    The pinned OpenAI and Anthropic SDKs call their async
    ``_sleep_for_retry`` hook from the ``except`` block that caught the
    request exception. Python preserves that handled exception across the
    awaited call, so :func:`sys.exception` exposes its transport cause chain.
    """

    async def _sleep_for_retry(self, **kwargs: Any) -> None:
        active_exception = sys.exception()
        if active_exception is not None and is_deterministic_connection_error(active_exception):
            raise active_exception
        await super()._sleep_for_retry(**kwargs)

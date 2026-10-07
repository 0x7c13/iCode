# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Retry policy for the provider SDK client subclasses.

It imports no provider SDK, so each SDK's subclass module loads only its own.
The two pinned SDKs expose different retry hooks, so each has its own guard.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Protocol

from chrys.foundation.errors import is_deterministic_connection_error

if TYPE_CHECKING:
    import httpx

    class _RetrySleepBase(Protocol):
        async def _sleep_for_retry(self, **kwargs: Any) -> None: ...

    class _RetryDecisionBase(Protocol):
        def _should_retry_exception(self, err: BaseException) -> tuple[bool, httpx.Response | None]: ...

else:
    _RetrySleepBase = object
    _RetryDecisionBase = object


class DeterministicConnectionSleepGuard(_RetrySleepBase):
    """Stop OpenAI SDK retries when the active request error cannot self-heal.

    The pinned OpenAI SDK calls its async ``_sleep_for_retry`` hook from the
    ``except`` block that caught the request exception. Python preserves that
    handled exception across the awaited call, so :func:`sys.exception`
    exposes its transport cause chain.
    """

    async def _sleep_for_retry(self, **kwargs: Any) -> None:
        active_exception = sys.exception()
        if active_exception is not None and is_deterministic_connection_error(active_exception):
            raise active_exception
        await super()._sleep_for_retry(**kwargs)


class DeterministicConnectionDecisionGuard(_RetryDecisionBase):
    """Stop Anthropic SDK retries when a request attempt's error cannot self-heal.

    The pinned Anthropic SDK passes every exception a request attempt raised to
    ``_should_retry_exception``; a status response it retries never reaches it,
    so an exception the caller is handling cannot veto that retry. Declining
    here makes the SDK re-raise its own error, an ``APIConnectionError`` whose
    ``__cause__`` is the transport failure.
    """

    def _should_retry_exception(self, err: BaseException) -> tuple[bool, httpx.Response | None]:
        if is_deterministic_connection_error(err):
            return False, None
        return super()._should_retry_exception(err)

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Offline clients with scripted responses and non-retryable failures."""

from __future__ import annotations

from typing import Any

from chrys.service.llm.mock import MockChatClient, MockResponse


class ErrorMockChatClient(MockChatClient):
    """``MockChatClient`` extended with per-call exception scripting.

    ``outcomes`` is a sequence of ``MockResponse`` or ``BaseException``.
    Each call pops the next entry; responses are returned normally,
    exceptions raise inside the client's awaitable (mirrors how a real
    SDK surfaces transport/rate-limit failures).
    """

    def __init__(
        self,
        outcomes: list[MockResponse | BaseException] | None = None,
        *,
        default_model_id: str = "mock-model",
    ) -> None:
        placeholder_responses: list[MockResponse] = []
        errors: dict[int, BaseException] = {}
        for i, o in enumerate(outcomes or []):
            if isinstance(o, BaseException):
                # Fill the responses slot with a placeholder so
                # call_index bookkeeping stays aligned with the real
                # parent implementation.
                placeholder_responses.append(MockResponse())
                errors[i] = o
            else:
                placeholder_responses.append(o)
        super().__init__(responses=placeholder_responses, default_model_id=default_model_id)
        self._errors_by_index = errors
        self.stream_flags: list[bool] = []

    def _inner_get_response(self, *, messages, stream, options, **kwargs):  # type: ignore[override]
        self.stream_flags.append(stream)
        idx = self._call_index
        err = self._errors_by_index.get(idx)
        if err is None:
            return super()._inner_get_response(messages=messages, stream=stream, options=options, **kwargs)

        # Scripted error path — still advance the call counter / history.
        self._call_history.append((list(messages), dict(options)))
        self._call_index += 1

        async def _raise() -> Any:
            raise err

        return _raise()


# A non-retryable framework-style exception.  Plain ``RuntimeError``
# doesn't match ``errors.RETRYABLE_TYPE_NAMES``, so it pauses on the
# first failure — perfect for exercising the pause path deterministically.
class FrameworkBoom(Exception):
    pass

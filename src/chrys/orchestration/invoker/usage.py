# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Pass-local usage observations beneath validation and wire retry boundaries."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from chrys.kernel import ChatContext, ChatMiddleware, ChatResponse, ChatResponseUpdate, ResponseStream
from chrys.kernel.types import normalize_stream_usage

from .contracts import UsageDelta


@dataclass
class _WireUsage:
    """A started request remains unreported until its response has finalized."""

    details: Mapping[str, Any] | None = None
    updates: list[Mapping[str, Any]] = field(default_factory=list)
    reported: bool = False

    def observe_update(self, update: ChatResponseUpdate) -> ChatResponseUpdate:
        for content in update.contents:
            if content.type == "usage" and content.usage_details is not None:
                self.updates.append(dict(content.usage_details))
        return update

    def finish(self, response: ChatResponse) -> None:
        self.details = response.usage_details
        self.reported = self.details is not None

    def observed_details(self) -> Mapping[str, Any]:
        return self.details if self.details is not None else normalize_stream_usage(self.updates) or {}


class PassUsageProbe(ChatMiddleware):
    """Observe each raw response, including rejected validation attempts.

    Install last in the per-call chat chain. Each call_next acquisition creates
    one record, so exceptions and abandoned streams retain a missing report.
    A stream finalizer observes the original response without rebuilding it;
    result hooks alone would be discarded by response validation on rejection.
    This probe is allocated per pass and never participates in retry rollback.
    """

    def __init__(self) -> None:
        self._requests: list[_WireUsage] = []

    async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
        request = _WireUsage()
        self._requests.append(request)
        await call_next()
        result = context.result
        if isinstance(result, ChatResponse):
            request.finish(result)
        elif isinstance(result, ResponseStream):
            result.with_update_filter(request.observe_update)

            async def finalize(_updates: Sequence[ChatResponseUpdate]) -> ChatResponse:
                response = await result.get_final_response()
                request.finish(response)
                return response

            context.result = result.with_finalizer(finalize)

    def snapshot(self, *, response_received: bool) -> UsageDelta:
        details = [request.observed_details() for request in self._requests]
        # No observations cannot establish complete zero usage (e.g. a hook
        # failed before the request, or an alternate agent bypassed the chain).
        missing = sum(not request.reported for request in self._requests) if self._requests else 1
        return UsageDelta(
            input_tokens=sum(item.get("input_token_count") or 0 for item in details),
            output_tokens=sum(item.get("output_token_count") or 0 for item in details),
            total_tokens=sum(item.get("total_token_count") or 0 for item in details),
            complete=missing == 0 and response_received,
            unreported=missing,
        )

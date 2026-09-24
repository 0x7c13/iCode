# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Scripted workflow agent used by persistence and historical TUI replay tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from chrys.kernel import ChatResponse, ChatResponseUpdate, Content, FinishReason, FinishReasonLiteral, Message
from chrys.service.llm.mock import MockChatClient, MockResponse
from tests.orchestration.workflows._hosting import PROFILE


class ArchiveClient(MockChatClient):
    def __init__(self, target: Path, outcome: str, *, pause_before_final: bool = False) -> None:
        call = ("read_file", "reused-call", {"path": str(target)})
        super().__init__(
            responses=[
                MockResponse(text="First inspection.", tool_calls=[call]),
                MockResponse(text="Second inspection.", tool_calls=[call]),
                MockResponse(text="Done"),
            ]
        )
        self.outcome = outcome
        self.pause_before_final = pause_before_final
        self.waiting = asyncio.Event()
        self.release_final = asyncio.Event()

    def _next_response(self) -> MockResponse:
        if self.call_count == 2 and self.outcome == "failed":
            raise RuntimeError("Archive failure [literal]")
        return super()._next_response()

    def _build_messages(self, resp: MockResponse) -> list[Message]:
        # Model the provider's prose-before-call sequence, which replay must
        # retain exactly (the generic mock emits calls before its text).
        contents = [Content.from_text(resp.text)] if resp.text else []
        contents.extend(
            Content.from_function_call(call_id, name, arguments=args) for name, call_id, args in resp.tool_calls
        )
        return [Message("assistant", contents)]

    async def _mock_response(
        self,
        resp: MockResponse,
        model_id: str,
        finish_reason: FinishReasonLiteral | FinishReason,
        async_cb: Callable[[str], Awaitable[None]] | None,
    ) -> ChatResponse[Any]:
        if resp.text == "Done" and (self.outcome == "cancelled" or self.pause_before_final):
            self.waiting.set()
            await self.release_final.wait()
        return await super()._mock_response(resp, model_id, finish_reason, async_cb)

    async def _stream_updates(
        self, resp: MockResponse, model_id: str, finish_reason: FinishReasonLiteral | FinishReason
    ) -> AsyncIterable[ChatResponseUpdate]:
        if resp.text == "Done" and (self.outcome == "cancelled" or self.pause_before_final):
            self.waiting.set()
            await self.release_final.wait()
        if resp.text:
            yield ChatResponseUpdate(contents=[Content.from_text(resp.text)], role="assistant", model=model_id)
        async for update in super()._stream_updates(replace(resp, text=""), model_id, finish_reason):
            yield update


def source() -> bytes:
    return (
        "from chrys.workflows import WorkflowBuilder\n"
        "wf = WorkflowBuilder('archive')\n"
        f"node = wf.agent('node', profile={PROFILE!r})\n"
        "wf.start(node)\nwf.output(node)\nworkflow = wf.build()\n"
    ).encode()

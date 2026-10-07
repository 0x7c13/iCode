# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A streaming attempt that stops reading before the final response closes its stream.

The stream's cleanup hooks (usage, the OTel ContextVar resets) run only when it
is closed, and a reset only succeeds in the context the stream was opened in.
A pull that fails runs those hooks and one that is cancelled closes the
stream; an attempt that ends between two pulls has to close it, in its own
task, before it ends.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import AsyncIterator, Callable
from unittest.mock import create_autospec

import pytest

from chrys.kernel import AgentResponse, AgentResponseUpdate, Content, Message, ResponseStream
from chrys.orchestration.engine.run.bindings import TurnBindings

_OPENED = contextvars.ContextVar[bool]("_OPENED", default=False)


class _Provider:
    """Agent stream that opens like the telemetry layer: a ContextVar set on open, reset on cleanup."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = texts
        self.ends: list[str] = []
        self.pulls_after_first_update = 0

    def run(self, messages: list[Message], *, stream: bool) -> ResponseStream[AgentResponseUpdate, AgentResponse]:
        del messages
        assert stream is True
        token = _OPENED.set(True)

        async def _updates() -> AsyncIterator[AgentResponseUpdate]:
            try:
                for index, text in enumerate(self._texts):
                    yield AgentResponseUpdate(contents=[Content.from_text(text)], role="assistant")
                    if index == 0:
                        self.pulls_after_first_update += 1
            finally:
                # A real provider's teardown suspends: it closes a connection.
                await asyncio.sleep(0)
                self.ends.append("provider closed")

        def _reset() -> None:
            # Raises ValueError outside the context the stream was opened in.
            _OPENED.reset(token)
            self.ends.append("cleanup")

        return ResponseStream(_updates(), finalizer=AgentResponse.from_updates).with_cleanup_hook(_reset)


class _Observer:
    """Stream observer acting on the first update it sees."""

    def __init__(self, act: Callable[[], None]) -> None:
        self._act = act

    def on_update(self, update: AgentResponseUpdate) -> None:
        del update
        self._act()

    async def on_retry_boundary(self) -> None:
        pass

    async def before_finalize(self) -> None:
        pass


def _fail() -> None:
    raise RuntimeError("observer failed")


@pytest.mark.parametrize(
    ("leave", "raised"),
    [("interrupted", asyncio.CancelledError), ("observer_failed", RuntimeError)],
)
async def test_an_attempt_left_between_chunks_closes_its_stream_in_its_own_task(
    executor: TurnBindings, monkeypatch: pytest.MonkeyPatch, leave: str, raised: type[BaseException]
) -> None:
    provider = _Provider(["one", "two"])
    monkeypatch.setattr(executor._agent, "run", create_autospec(executor._agent.run, side_effect=provider.run))
    # An interrupt cancels the attempt task through its handle; requested
    # while the update is observed, it lands at the yield between chunks.
    act = executor._attempt_handle.cancel if leave == "interrupted" else _fail
    monkeypatch.setattr(executor._attempts, "_stream_observer", lambda: _Observer(act))

    with pytest.raises(raised):
        await executor._attempts._stream_single_attempt([], {}, watchdog=False)

    # The attempt left between chunks: the provider was never pulled again,
    # so only the attempt's own close ended it, and in the attempt's context.
    assert provider.pulls_after_first_update == 0
    assert provider.ends == ["provider closed", "cleanup"]
    assert executor._attempt_handle.task is None

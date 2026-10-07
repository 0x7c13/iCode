# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The streaming validation proxy closes the inner stream it was reading.

An explicit close of the proxy mid-response reaches the provider stream before
it returns. Finalization, which closes each abandoned generator in a task of
its own, must not have the proxy's generator close the provider's.
"""

from __future__ import annotations

import asyncio
from typing import Any

from chrys.kernel import ChatMiddlewareLayer, Message
from chrys.service.agent_middleware.response_validation import ResponseValidationMiddleware
from tests.kernel.test_loop_stream_close import _StreamEndsClient, _text
from tests.support.transcript_invariants import InvariantCheckedToolLoopLayer


def _validated_layer(client: _StreamEndsClient) -> InvariantCheckedToolLoopLayer:
    return InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(client, middleware=[ResponseValidationMiddleware()]))


async def test_closing_mid_response_closes_the_provider_stream_under_the_validation_proxy() -> None:
    client = _StreamEndsClient([[_text("one"), _text("two")]])

    stream = _validated_layer(client).get_response([Message("user", ["hi"])], stream=True)
    async for _update in stream:
        break
    await stream.aclose()

    assert client.ends == [["provider closed", "cleanup"]]


def test_a_validated_stream_left_open_at_loop_shutdown_closes_each_generator_on_its_own() -> None:
    # ``loop.shutdown_asyncgens`` (like garbage collection) closes every
    # abandoned generator in a task of its own. A proxy generator whose exit
    # closed the provider's would collide with that one's own close once its
    # teardown suspends ("aclose(): asynchronous generator is already
    # running").
    client = _StreamEndsClient([[_text("one"), _text("two")]])
    layer = _validated_layer(client)
    reported: list[dict[str, Any]] = []
    left_open: list[object] = []

    async def _read_one_update_and_leave() -> None:
        asyncio.get_running_loop().set_exception_handler(lambda _loop, context: reported.append(context))
        stream = layer.get_response([Message("user", ["hi"])], stream=True)
        # Kept alive, so the shutdown closes its generators, not the GC.
        left_open.append(stream)
        async for _update in stream:
            break

    asyncio.run(_read_one_update_and_leave())

    assert reported == []
    assert client.ends == [["provider closed"]]

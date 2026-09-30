# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Drive a reminder-middleware call the way the kernel establishes its request.

Production runs a call's request observers at two points: the pipeline's
final-handler boundary with ``context.messages``, then the kernel client with
the provider-request views it builds (fresh wrappers and contents lists
sharing the content objects and ``additional_properties``).  In between, the
client runs compaction over ``context.messages`` (``_prepare_wire_call``), so
a fold's re-sent reminders reach only the second delivery.  Direct middleware
tests establish requests through here so both delivery points, the step
between them and the views' fresh contents are exercised.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from chrys.kernel.client import _prepare_provider_request_messages
from chrys.kernel.middleware import ChatContext

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from chrys.kernel import Message
    from chrys.service.agent_middleware.system_reminder import SystemReminderMiddleware


# The client's work between the deliveries: compaction over the call's list,
# returning the messages it sends.
type PrepareStep = Callable[[list[Message]], Awaitable[list[Message]]]


async def establish_request(context: ChatContext, *, prepare: PrepareStep | None = None) -> list[Message]:
    """Run *context*'s request observers at both delivery points and return the provider-request views."""
    for observer in context.request_message_observers:
        observer(context.messages)
    messages = cast("list[Message]", context.messages)
    prepared = await prepare(messages) if prepare is not None else list(messages)

    def _observe_prepared(views: Sequence[Message]) -> None:
        for observer in context.request_message_observers:
            observer(views)

    return _prepare_provider_request_messages(prepared, _observe_prepared)


async def _run_call(
    middleware: SystemReminderMiddleware,
    messages: list[Message],
    options: dict[str, Any] | None,
    requests: int,
    prepare: PrepareStep | None = None,
) -> tuple[list[Message], list[Message]]:
    context = ChatContext(client=None, messages=list(messages), options=options)
    sent: list[Message] = []

    async def _requests() -> None:
        for _ in range(requests):
            sent[:] = await establish_request(context, prepare=prepare)

    await middleware.process(context, _requests)
    return cast("list[Message]", context.messages), sent


async def enrich_call(
    middleware: SystemReminderMiddleware,
    messages: list[Message],
    *,
    options: dict[str, Any] | None = None,
    requests: int = 1,
) -> list[Message]:
    """Run one call through *middleware* that establishes *requests* provider requests.

    ``requests=0`` is a call whose request never went out; more than one is a
    validation retry re-sending from the same pass.  Returns the per-call list
    the middleware produced (an untouched message keeps its identity).
    """
    enriched, _sent = await _run_call(middleware, messages, options, requests)
    return enriched


async def request_views(
    middleware: SystemReminderMiddleware,
    messages: list[Message],
    *,
    options: dict[str, Any] | None = None,
    prepare: PrepareStep | None = None,
) -> list[Message]:
    """Run one established call through *middleware* and return its provider-request views.

    *prepare* runs where the client compacts, between the two deliveries;
    the views are built from what it returns.
    """
    _enriched, sent = await _run_call(middleware, messages, options, 1, prepare)
    return sent

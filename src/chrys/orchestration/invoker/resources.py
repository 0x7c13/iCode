# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Prepared, conversation and pass resources with a single close authority.

Close tasks stay reachable through their owner until every release has run.
Cancellation of a waiter cannot orphan the displaced build or cancel another
close waiter. A release's own cancellation is remembered and re-raised after
the remaining releases, matching the build transaction's former cleanup chain.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any

from .attempts import AttemptTaskHandle
from .contracts import AbortCause, OperationBinding, PreparedClosed, Unbind

logger = logging.getLogger(__name__)

type Release = Callable[[], Awaitable[object]]


async def finish_close(task: asyncio.Task[None]) -> None:
    """Drain an owned cleanup task even after repeated waiter cancellation."""
    cancelled: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancelled = exc
    task.result()
    if cancelled is not None:
        raise cancelled


async def rollback_resources(owner: ResourceScope) -> None:
    """Complete rollback, preserving any new cancellation of its caller.

    aclose/finish_close owns cancellation drainage. Only ordinary cleanup
    failures are logged here so the caller can re-raise its acquisition error.
    """
    try:
        await owner.aclose()
    except Exception:
        logger.exception("Error rolling back invocation resources")


class ResourceScope:
    """Reverse-acquisition release stack; concurrent closes share one task."""

    def __init__(self) -> None:
        self._releases: list[Release] = []
        self._retained: list[object] = []
        self._releasing: set[asyncio.Task[None]] = set()
        self._close_task: asyncio.Task[None] | None = None

    @property
    def closing(self) -> bool:
        return self._close_task is not None

    def own(self, release: Release) -> None:
        """Register the one release authority immediately after acquisition."""
        if self.closing:
            raise PreparedClosed("Resource owner is closing")
        self._releases.append(release)

    def retain[T](self, resource: T) -> T:
        """Keep private in-memory state reachable until this scope drains."""
        if self.closing:
            raise PreparedClosed("Resource owner is closing")
        self._retained.append(resource)
        return resource

    async def release(self, release: Release) -> None:
        """Release an operation-bound resource at its established shell boundary."""
        self._releases.remove(release)

        async def close_one() -> None:
            await release()

        task = asyncio.create_task(close_one())
        self._releasing.add(task)
        try:
            await finish_close(task)
        finally:
            self._releasing.discard(task)

    async def _close(self) -> None:
        cancelled: asyncio.CancelledError | None = None
        for task in tuple(self._releasing):
            try:
                await finish_close(task)
            except asyncio.CancelledError as exc:
                cancelled = exc
            except Exception:
                logger.exception("Error releasing invocation resource")
        while self._releases:
            release = self._releases.pop()
            try:
                await release()
            except asyncio.CancelledError as exc:
                cancelled = exc
            except Exception:
                logger.exception("Error releasing invocation resource")
        self._retained.clear()
        if cancelled is not None:
            raise cancelled

    async def aclose(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await finish_close(self._close_task)


class PassResources(ResourceScope):
    """A pass's mutable presentation and validation reset/cleanup lifetime.

    This is only a resource scope, not an invocation handle or a backend run
    contract. Validation instances belong to the conversation; their reset
    hooks still execute inside the caller's existing pass try boundary.
    """

    def __init__(self, handle: AttemptTaskHandle, start_hooks: tuple[Callable[[], None], ...]) -> None:
        super().__init__()
        self.handle = handle
        self._start_hooks = start_hooks

    def begin(self) -> None:
        for hook in self._start_hooks:
            hook()

    def request_close(self, cause: AbortCause) -> None:
        self.handle.cancel()

    @property
    def drained(self) -> Awaitable[None]:
        return self._drain()

    async def _drain(self) -> None:
        task = self.handle.task
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)


class Conversation(ResourceScope):
    """Private runtime resources, plus the operation that currently uses them."""

    def __init__(self, *, on_closed: Callable[[Conversation], None] | None = None) -> None:
        super().__init__()
        self._on_closed = on_closed
        self._operation: OperationBinding | None = None
        self._pass: OperationBinding | None = None

    def bind_operation(self, binding: OperationBinding) -> Unbind:
        if self.closing:
            raise PreparedClosed("Conversation is closing")
        if self._operation is not None:
            raise RuntimeError("Conversation already has a bound operation")
        self._operation = binding

        def unbind() -> None:
            if self._operation is binding:
                self._operation = None

        return unbind

    def bind_pass(self, binding: OperationBinding) -> Unbind:
        """Retain pass convergence when there is no caller operation binding."""
        if self.closing:
            raise PreparedClosed("Conversation is closing")
        if self._pass is not None:
            raise RuntimeError("Conversation already has a bound pass")
        self._pass = binding

        def unbind() -> None:
            if self._pass is binding:
                self._pass = None

        return unbind

    async def _close(self) -> None:
        binding = self._operation or self._pass
        cancelled: asyncio.CancelledError | None = None
        try:
            if binding is not None:
                binding.request_close(AbortCause.OWNER_CLOSE)
                await binding.drained
        except asyncio.CancelledError as exc:
            cancelled = exc
        except Exception:
            logger.exception("Error draining invocation operation")
        finally:
            try:
                await super()._close()
            finally:
                if self._on_closed is not None:
                    self._on_closed(self)
        if cancelled is not None:
            raise cancelled


class PreparedAgent(ResourceScope):
    """Shared resources, and the private conversations opened against them."""

    def __init__(self) -> None:
        super().__init__()
        self._conversations: list[Conversation] = []
        self._opening: list[tuple[asyncio.Task[Any], asyncio.Event]] = []

    async def open[T](self, factory: Callable[[Conversation], Awaitable[T]]) -> T:
        """Acquire a private runtime, rolling back partial opens in reverse."""
        if self.closing:
            raise PreparedClosed("Prepared agent is closing")
        conversation = Conversation(on_closed=self._forget_conversation)
        task = asyncio.current_task()
        assert task is not None
        opening = (task, asyncio.Event())
        self._opening.append(opening)
        try:
            value = await factory(conversation)
            if self.closing:
                raise PreparedClosed("Prepared agent closed during open")
            self._conversations.append(conversation)
            return value
        except BaseException:
            await rollback_resources(conversation)
            raise
        finally:
            self._opening.remove(opening)
            opening[1].set()

    def _forget_conversation(self, conversation: Conversation) -> None:
        if conversation in self._conversations:
            self._conversations.remove(conversation)

    async def _close(self) -> None:
        opening = tuple(self._opening)
        for task, _drained in opening:
            task.cancel()
        if opening:
            # Await this acquire's rollback, not its caller task: the caller
            # may itself join aclose() in its finally after open raises.
            await asyncio.gather(*(drained.wait() for _task, drained in opening))
        cancelled: asyncio.CancelledError | None = None
        for conversation in reversed(tuple(self._conversations)):
            try:
                await conversation.aclose()
            except asyncio.CancelledError as exc:
                cancelled = exc
        self._conversations.clear()
        try:
            await super()._close()
        except asyncio.CancelledError as exc:
            cancelled = exc
        if cancelled is not None:
            raise cancelled


class OperationLifetime:
    """Shell-owned completion barrier, including cleanup after the pass returns.

    The common child shell owns this completion barrier from preparation through
    its backend policy and terminal cleanup. Once cleanup
    starts, owner-close latches but lets that cleanup finish in its original task.
    """

    def __init__(self) -> None:
        self._task = asyncio.current_task()
        self._done = asyncio.Event()
        self._cleaning = False
        self._requested = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def drained(self) -> Awaitable[None]:
        return self._drain()

    async def _drain(self) -> None:
        await self._done.wait()
        if self._close_task is not None:
            await finish_close(self._close_task)

    def cancel_preparation(self) -> None:
        self._requested = True
        self._cancel_operation()

    def _cancel_operation(self) -> None:
        if not self._cleaning and self._task is not None and not self._task.done():
            self._task.cancel()

    def close_with(self, close: Callable[[], Coroutine[Any, Any, None]]) -> None:
        """Run existing controller convergence, then wake a late pause callback."""
        if self._requested:
            return
        self._requested = True

        async def converge() -> None:
            try:
                await close()
            finally:
                self._cancel_operation()

        self._close_task = asyncio.create_task(converge())

    def begin_cleanup(self) -> None:
        self._cleaning = True

    def finish(self) -> None:
        self._done.set()


class TurnTaskBinding:
    """A Turn shell's drain includes finalization and save on its owned task."""

    def __init__(self, task: asyncio.Task[None], on_close: Callable[[AbortCause], None] | None) -> None:
        self._task = task
        self._on_close = on_close

    def request_close(self, cause: AbortCause) -> None:
        if self._on_close is not None:
            self._on_close(cause)
        if not self._task.done():
            self._task.cancel()

    @property
    def drained(self) -> Awaitable[None]:
        return self._drain()

    async def _drain(self) -> None:
        await asyncio.gather(self._task, return_exceptions=True)

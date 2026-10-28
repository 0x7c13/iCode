# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A frontend double for hook confirmations: claims what it is told to, answers on demand."""

from __future__ import annotations

from typing import TYPE_CHECKING

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import HookConfirmationRequest, HookConfirmationResponse, HookConfirmationWithdrawn
from chrys.foundation.models.hook_confirmations import HookConfirmationAnswer

if TYPE_CHECKING:
    from chrys.service.hooks.confirmation import HookConfirmationService


def confirmations_settled(service: HookConfirmationService) -> bool:
    """Whether no ask waits and none is still winding down (telling its frontend, closing its span)."""
    return not service._pending and not service._lifecycles


class FakeConfirmationFrontend:
    """Claims every request of *session_id* (every session's when None; unless ``claim`` is False), and
    answers inline with ``answer_inline`` or later through :meth:`answer`; every request and withdrawal is kept."""

    def __init__(
        self,
        bus: EventBus,
        *,
        session_id: str | None,
        claim: bool = True,
        answer_inline: HookConfirmationAnswer | None = None,
    ) -> None:
        self.bus = bus
        self.session_id = session_id
        self.claim = claim
        self.answer_inline = answer_inline
        self.requests: list[HookConfirmationRequest] = []
        self._sessions: dict[str, str | None] = {}
        self.claimed: list[str] = []
        self.withdrawn: list[HookConfirmationWithdrawn] = []

    async def attach(self) -> None:
        await self.bus.subscribe(HookConfirmationRequest, self._on_request)
        await self.bus.subscribe(HookConfirmationWithdrawn, self._on_withdrawn)

    async def detach(self) -> None:
        await self.bus.unsubscribe(HookConfirmationRequest, self._on_request)
        await self.bus.unsubscribe(HookConfirmationWithdrawn, self._on_withdrawn)

    async def _on_request(self, event: HookConfirmationRequest) -> None:
        if self.session_id is not None and event.session_id != self.session_id:
            return
        self.requests.append(event)
        self._sessions[event.request_id] = event.session_id
        if self.claim and event.claim.try_claim():
            self.claimed.append(event.request_id)
            if self.answer_inline is not None:
                await self.answer(event.request_id, self.answer_inline)

    async def _on_withdrawn(self, event: HookConfirmationWithdrawn) -> None:
        if event.request_id in self._sessions:
            self.withdrawn.append(event)

    async def answer(self, request_id: str, answer: HookConfirmationAnswer, *, session_id: str | None = None) -> None:
        own = self._sessions.get(request_id, self.session_id)
        await self.bus.publish(
            HookConfirmationResponse(
                session_id=own if session_id is None else session_id, request_id=request_id, answer=answer
            )
        )

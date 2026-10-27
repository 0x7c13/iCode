# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""What a frontend needs to ask the user about a hook's ``ask``, and how it answers.

A blocking ``before_tool_call`` or ``user_prompt_submit`` hook can return
``action: ask``. The session's confirmation service publishes one
``HookConfirmationRequest`` per ask; the one frontend that claims it shows it
and publishes the ``HookConfirmationResponse``; ``HookConfirmationWithdrawn``
tells it the request ended without that answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

HookConfirmationAnswer = Literal["approved", "declined", "dismissed", "frontend_unavailable"]
"""What a frontend answers: the user continued, cancelled, or closed the question, or the
frontend could not show it (a failed display, a disconnected client)."""

HookConfirmationOutcome = Literal["approved", "declined", "timed_out", "unavailable", "invalidated", "interrupted"]
"""How one ask ended. ``unavailable``: nobody could be asked (the session cannot ask the user,
no frontend claimed the request, or the frontend failed); ``invalidated``: the operation it asked
about ended or changed first; ``interrupted``: the waiting task was cancelled."""

RefusedInput = Literal["prompt", "retry_note", "injection"]
"""Which user input a gate refused before it ran (see ``Error.refused_input``): a new turn's
message, the note a retry adds, or a message sent into a running turn."""

HookConfirmationSubjectKind = Literal["tool", "input"]
"""What the hook asks about: a tool call (``before_tool_call``) or the user's input (``user_prompt_submit``)."""


@dataclass(frozen=True, slots=True)
class HookArgumentChange:
    """One tool argument that differs from the version the user last continued, as JSON text."""

    name: str
    before: str | None
    """None: the argument was not there."""
    after: str | None
    """None: the argument is gone."""


class HookConfirmationClaim:
    """Which frontend shows a request: the first ``try_claim`` wins, and no claim follows the request's end.

    Claiming is synchronous, so the handlers ``EventBus.publish`` runs in turn
    settle it before the publish returns; the service then ends an unclaimed
    request at once instead of waiting for nobody.
    """

    __slots__ = ("_claimed", "_ended")

    def __init__(self) -> None:
        self._claimed = False
        self._ended = False

    @property
    def claimed(self) -> bool:
        return self._claimed

    def try_claim(self) -> bool:
        """Take the request for this frontend; False when another did or the request already ended."""
        if self._claimed or self._ended:
            return False
        self._claimed = True
        return True

    def end(self) -> None:
        """The request ended: nothing claims it any more."""
        self._ended = True

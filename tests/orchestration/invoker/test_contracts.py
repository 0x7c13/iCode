# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Intent/capability admission is independent of backend side effects."""

from __future__ import annotations

import pytest

from chrys.foundation.models.invocations import InvocationOrigin
from chrys.kernel import Content, Message
from chrys.orchestration.invoker.contracts import (
    ContinuationCapability,
    ContinuationTicket,
    RunIntent,
    RunRequest,
    StaleContinuation,
    UnsupportedRequest,
    validate_request,
)


@pytest.mark.parametrize("capability", list(ContinuationCapability))
@pytest.mark.parametrize("intent", list(RunIntent))
@pytest.mark.parametrize("input_kind", ["empty_sequence", "empty_text", "text", "image"])
@pytest.mark.parametrize("ticket_kind", ["none", "matching", "other"])
def test_intent_capability_admission_table(
    capability: ContinuationCapability, intent: RunIntent, input_kind: str, ticket_kind: str
) -> None:
    messages = {
        "empty_sequence": [],
        "empty_text": [Message("user", [""])],
        "text": [Message("user", ["prompt"])],
        "image": [Message("user", [Content.from_uri("https://example.invalid/image.png", media_type="image/png")])],
    }[input_kind]
    other = next(item for item in ContinuationCapability if item is not capability)
    ticket = (
        None
        if ticket_kind == "none"
        else ContinuationTicket("conversation", "pass", 0, capability if ticket_kind == "matching" else other)
    )
    request = RunRequest(messages, intent, InvocationOrigin("turn", "session", "inv", None), ticket)
    accepted = (
        (intent is RunIntent.FRESH and ticket_kind == "none" and input_kind != "empty_sequence")
        or (
            intent is RunIntent.CONTINUE
            and capability is ContinuationCapability.CONTINUE_HISTORY
            and ticket_kind == "none"
        )
        or (intent is RunIntent.RETRY and ticket_kind == "matching")
    )
    if capability is ContinuationCapability.FRESH_SESSION and input_kind in ("image", "empty_sequence"):
        accepted = False
    if accepted:
        validate_request(request, capability)
    else:
        with pytest.raises((UnsupportedRequest, StaleContinuation)):
            validate_request(request, capability)

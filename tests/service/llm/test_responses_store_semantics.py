# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""OpenAI Responses adapter derives conversation-id learning from the effective store.

A present ``extra_body.store`` overrides the named ``store`` on the wire
(request-body merge is extra_body-last), so the adapter must apply the same
veto when deciding whether a response id may be recorded as a continuation
handle — recording an id the service never persisted makes the next
``previous_response_id`` request fail.  This covers raw request options too
(``Agent.run(options=...)`` and direct client calls bypass profile-level
``effective_chat_options`` normalization).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from chrys.kernel import ChatResponseUpdate
from chrys.service.llm.openai_responses.client import OPENAI_RESPONSES
from chrys.service.llm.openai_responses.decode import decode_response
from chrys.service.llm.openai_responses.stream import StreamState


def _decoded_update(event: Any, options: dict[str, Any]) -> ChatResponseUpdate:
    return StreamState(options, model="gpt-test", variant=OPENAI_RESPONSES).update_for(event)


def _response() -> SimpleNamespace:
    return SimpleNamespace(
        id="resp_1",
        created_at=0,
        model="gpt-test",
        metadata={},
        output=[],
        usage=None,
        conversation=None,
        status="completed",
        incomplete_details=None,
    )


def _conversation_id(options: dict[str, Any]) -> str | None:
    parsed = decode_response(_response(), options, variant=OPENAI_RESPONSES)
    return parsed.conversation_id


def _streaming_conversation_id(options: dict[str, Any], event_type: str = "response.completed") -> str | None:
    event = SimpleNamespace(type=event_type, response=_response())
    update = _decoded_update(event, options)
    return update.conversation_id


def test_extra_body_store_false_blocks_conversation_id_learning() -> None:
    # The wire body carries store:false (extra_body wins), so the service
    # never persists resp_1 — recording it would poison the next request's
    # previous_response_id.
    assert _conversation_id({"store": True, "extra_body": {"store": False}}) is None


def test_extra_body_store_null_keeps_conversation_id_learning() -> None:
    # A present null reaches the wire as JSON null and selects the provider
    # default — Responses stores — so the id stays referenceable.
    assert _conversation_id({"store": True, "extra_body": {"store": None}}) == "resp_1"


def test_plain_store_values_are_unchanged() -> None:
    assert _conversation_id({"store": True}) == "resp_1"
    assert _conversation_id({"store": False}) is None
    # Absent store: the provider default stores, and the adapter has always
    # recorded the id in that case.
    assert _conversation_id({}) == "resp_1"
    # extra_body store:true never PROMOTES a top-level false — client-side
    # history over a storing wire stays fully local.
    assert _conversation_id({"store": False, "extra_body": {"store": True}}) is None


@pytest.mark.parametrize("event_type", ["response.created", "response.in_progress", "response.completed"])
def test_streaming_events_apply_the_same_store_veto(event_type: str) -> None:
    # All three streaming call sites must share the blocking path's rule —
    # a streamed run otherwise learns an unstored resp_1 and the next
    # request's previous_response_id 400s.
    assert _streaming_conversation_id({"store": True, "extra_body": {"store": False}}, event_type) is None
    assert _streaming_conversation_id({"store": True, "extra_body": {"store": None}}, event_type) == "resp_1"
    assert _streaming_conversation_id({"store": True}, event_type) == "resp_1"
    assert _streaming_conversation_id({"store": False, "extra_body": {"store": True}}, event_type) is None


def _continuation_token(options: dict[str, Any]) -> Any:
    response = _response()
    response.status = "in_progress"
    return decode_response(response, options, variant=OPENAI_RESPONSES).continuation_token


def _streaming_continuation_token(options: dict[str, Any], event_type: str) -> Any:
    response = _response()
    response.status = "in_progress"
    event = SimpleNamespace(type=event_type, response=response)
    return _decoded_update(event, options).continuation_token


def test_unstored_responses_never_mint_continuation_tokens() -> None:
    # A store:false response cannot be retrieved later — a minted token turns
    # a disconnect-retry into a guaranteed 404 retrieval instead of a clean
    # fresh create from local history.
    assert _continuation_token({"store": False}) is None
    assert _continuation_token({"store": True, "extra_body": {"store": False}}) is None
    assert _continuation_token({"store": True}) is not None
    assert _continuation_token({}) is not None


@pytest.mark.parametrize("event_type", ["response.created", "response.in_progress"])
def test_streaming_mint_sites_apply_the_same_store_gate(event_type: str) -> None:
    assert _streaming_continuation_token({"store": False}, event_type) is None
    assert _streaming_continuation_token({"store": True, "extra_body": {"store": False}}, event_type) is None
    assert _streaming_continuation_token({"store": True}, event_type) is not None

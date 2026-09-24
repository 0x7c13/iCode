# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Instrumented OpenAI client builders shared by the instrumented-client test modules."""

from __future__ import annotations

from typing import Any

from chrys.service.llm.instrumented import (
    create_instrumented_openai_client,
    create_instrumented_openai_responses_client,
)


def make_chat_client(
    session_id: str | None = None,
    chat_client_cls: type[Any] | None = None,
    parent_session_id: str | None = None,
    use_route_session_context: bool = False,
) -> object:
    """Construct the real instrumented OpenAI client with a fake API key."""
    from openai import AsyncOpenAI

    return create_instrumented_openai_client(
        model_id="gpt-test",
        session_id=session_id,
        parent_session_id=parent_session_id,
        use_route_session_context=use_route_session_context,
        client=AsyncOpenAI(api_key="sk-fake"),
        chat_client_cls=chat_client_cls,
    )


def make_responses_chat_client(
    session_id: str | None = None,
    parent_session_id: str | None = None,
    use_route_session_context: bool = False,
    chat_client_cls: type[Any] | None = None,
) -> object:
    """Construct the real instrumented OpenAI Responses client with a fake API key."""
    from openai import AsyncOpenAI

    return create_instrumented_openai_responses_client(
        model_id="gpt-test",
        session_id=session_id,
        parent_session_id=parent_session_id,
        use_route_session_context=use_route_session_context,
        client=AsyncOpenAI(api_key="sk-fake"),
        chat_client_cls=chat_client_cls,
    )

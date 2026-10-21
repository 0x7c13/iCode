# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The batch numbering of a main pass's tool-calling responses."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from chrys.kernel import Content, Message


def intermediate_text_contents(messages: Sequence[Message]) -> tuple[Content, ...]:
    """Return the text one tool-calling response wrote beside its local calls.

    The wire client hands it to the intermediate-text callback, which records
    it for the session's ``_intermediate_text`` sidecar. A response with
    provider-hosted content keeps its text in its own messages and records
    none.
    """
    text_contents: list[Content] = []
    has_function_calls = False
    for message in messages:
        for content in message.contents:
            if content.provider_hosted:
                return ()
            if content.type == "text":
                if content.text:
                    text_contents.append(content)
            elif content.type == "function_call" and not content.informational_only:
                has_function_calls = True
    return tuple(text_contents) if has_function_calls else ()


class ResponseBatchCounter:
    """Number each tool-calling response of a pass.

    The intermediate-text callback starts a batch for every response that
    calls a local tool; the tool calls it dispatches are stamped with that
    batch's id, which groups them in the session history.
    """

    def __init__(self) -> None:
        self.batch_id: int = 0

    def new_batch(self) -> None:
        """Start the batch of the next tool-calling response."""
        self.batch_id += 1

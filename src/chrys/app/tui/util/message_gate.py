# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Dropping the messages that given widgets post to themselves, for code that awaits.

``MessagePump.prevent()`` only guards synchronous code: every task in the App shares its stack, so
while a block inside it awaits, the prevented types are dropped for every widget. A tab bar the reader
clicks meanwhile then moves its selection while its content stays put.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from textual.message import Message
    from textual.message_pump import MessagePump


@contextmanager
def messages_disabled(message_type: type[Message], *pumps: MessagePump) -> Iterator[None]:
    """Drop *message_type* posted to *pumps* until the block ends, including while it awaits.

    Messages that other widgets post are unaffected. Blocks on one widget must not overlap: the first
    to end enables the type again.
    """
    for pump in pumps:
        pump.disable_messages(message_type)
    try:
        yield
    finally:
        for pump in pumps:
            pump.enable_messages(message_type)

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The error text a tool card shows for a tool call that raised."""

from __future__ import annotations

from chrys.foundation.errors import clean_error_message
from chrys.kernel.exceptions import ModelVisibleToolError, tool_error_result_text


def tool_card_error_text(exc: Exception) -> str:
    """Return the ``Error: …`` text a tool card shows for a call that raised *exc*.

    A ``ModelVisibleToolError`` shows exactly what the model read, blank-message
    fallback included; ``clean_error_message`` would prefer its cause, which that
    message was written to stand in for. Any other exception shows its cleaned
    message, which the user may see but the model never reads.
    """
    if isinstance(exc, ModelVisibleToolError):
        return tool_error_result_text(exc)
    message = clean_error_message(exc)
    return message if message.startswith("Error: ") else f"Error: {message}"

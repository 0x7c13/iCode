# Copyright (c) Microsoft. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Microsoft Agent Framework (MIT License; see NOTICE).

"""Chrys-owned exception hierarchy."""

import logging
from typing import Any, Literal

from chrys.foundation.platform.files import surrogate_safe_text

logger = logging.getLogger(__name__)


class ChrysException(Exception):
    """Base exception for the Chrys kernel.

    Automatically logs the message as debug.
    """

    def __init__(
        self,
        message: str,
        inner_exception: Exception | None = None,
        log_level: Literal[0] | Literal[10] | Literal[20] | Literal[30] | Literal[40] | Literal[50] | None = 10,
        *args: Any,
        **kwargs: Any,
    ):
        """Create a ChrysException.

        This emits a debug log (by default), with the inner_exception if provided.
        """
        if log_level is not None:
            logger.log(log_level, message, exc_info=inner_exception)
        if inner_exception:
            super().__init__(message, inner_exception, *args)
        else:
            super().__init__(message, *args)


# region Agent Exceptions


class AgentException(ChrysException):
    """Base class for all agent exceptions."""


class AgentInvalidAuthException(AgentException):
    """An authentication error occurred in an agent."""


class AgentInvalidRequestException(AgentException):
    """An invalid request was made to an agent."""


class AgentInvalidResponseException(AgentException):
    """An invalid or unexpected response was received from an agent."""


class AgentContentFilterException(AgentException):
    """A content filter was triggered by an agent."""


# endregion

# region Chat Client Exceptions


class ChatClientException(ChrysException):
    """Base class for all chat client exceptions."""


class ChatClientInvalidAuthException(ChatClientException):
    """An authentication error occurred in a chat client."""


class ChatClientInvalidRequestException(ChatClientException):
    """An invalid request was made to a chat client."""


class ChatClientInvalidResponseException(ChatClientException):
    """An invalid or unexpected response was received from a chat client."""


class ChatClientContentFilterException(ChatClientException):
    """A content filter was triggered by a chat client."""


# endregion

# region Integration Exceptions


class IntegrationException(ChrysException):
    """Base class for all external service/dependency integration exceptions."""


class IntegrationInitializationError(IntegrationException):
    """A wrapped dependency/service lifecycle failure occurred during setup."""


class IntegrationInvalidAuthException(IntegrationException):
    """An authentication error occurred in an external integration."""


class IntegrationInvalidRequestException(IntegrationException):
    """An invalid request was made to an external integration."""


class IntegrationInvalidResponseException(IntegrationException):
    """An invalid or unexpected response was received from an external integration."""


class IntegrationContentFilterException(IntegrationException):
    """A content filter was triggered by an external integration."""


# endregion

# region Content Exceptions


class ContentError(ChrysException):
    """An error occurred while processing content."""


class AdditionItemMismatch(ContentError):
    """A type mismatch occurred while merging content items."""


# endregion

# region Tool Exceptions


class ToolException(ChrysException):
    """Base class for all tool-related exceptions."""


class ToolExecutionException(ToolException):
    """A tool or prompt call failed at runtime."""


GENERIC_TOOL_ERROR_TEXT = "Error: Function failed."
"""The result the model reads for a failed tool call whose error text it may not see."""


def tool_error_result_text(exc: BaseException) -> str:
    """Return the result text the model reads for a tool call that raised *exc*.

    Exception text can carry local paths, stderr or credentials, so the model
    reads ``GENERIC_TOOL_ERROR_TEXT`` unless the tool raised
    ``ModelVisibleToolError`` with a non-blank message (see its ``result_text``).
    Tool cards showing a model-visible error use this same text.
    """
    text = exc.result_text if isinstance(exc, ModelVisibleToolError) else None
    return text or GENERIC_TOOL_ERROR_TEXT


class ModelVisibleToolError(ToolExecutionException):
    """A tool failure whose message the model reads to correct its next call.

    The tool loop answers any other exception with ``GENERIC_TOOL_ERROR_TEXT``,
    because arbitrary exception text can carry local paths, stderr or
    credentials. For this type it answers ``Error: <message>`` instead and
    still marks the result failed. Raise it only with text safe to show the
    model: an error a remote server sent back for this call, or text Chrys
    wrote for the model — never another exception's ``str()``.
    """

    @property
    def model_message(self) -> str:
        """The message passed to the constructor, without the ``inner_exception`` that ``str()`` also shows."""
        message = self.args[0] if self.args else ""
        return message if isinstance(message, str) else ""

    @property
    def result_text(self) -> str | None:
        """The failed result the model reads, ``Error: <message>``; None when the message is blank.

        The message is made surrogate-safe and stripped, and an ``Error:`` it
        already starts with gives way to the ``Error: `` prefix (space included)
        that renderers and the model key on; a message that is only that prefix
        is blank. Read it through ``tool_error_result_text``, which supplies
        the fallback for a blank message.
        """
        message = surrogate_safe_text(self.model_message).strip()
        if message.startswith("Error:"):
            message = message.removeprefix("Error:").lstrip()
        return f"Error: {message}" if message else None


# endregion

# region Middleware Exceptions


class MiddlewareException(ChrysException):
    """An error occurred during middleware execution."""


# endregion

# region Settings Exceptions


class SettingNotFoundError(ChrysException):
    """A required setting could not be resolved from any source."""


# endregion

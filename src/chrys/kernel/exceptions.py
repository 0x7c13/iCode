# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Chrys-owned exception hierarchy."""

import logging
from typing import Any, Literal

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


# endregion

# region Middleware Exceptions


class MiddlewareException(ChrysException):
    """An error occurred during middleware execution."""


# endregion

# region Settings Exceptions


class SettingNotFoundError(ChrysException):
    """A required setting could not be resolved from any source."""


# endregion

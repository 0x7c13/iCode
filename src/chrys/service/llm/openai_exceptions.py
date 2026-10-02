# Copyright (c) Microsoft. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Microsoft Agent Framework (MIT License; see NOTICE).

"""OpenAI wire exceptions owned by Chrys."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

from openai import BadRequestError

from chrys.kernel.exceptions import ChatClientContentFilterException


class ContentFilterResultSeverity(Enum):
    """The severity of the content filter result."""

    HIGH = "high"
    MEDIUM = "medium"
    SAFE = "safe"
    LOW = "low"
    # A severity this client does not know, or a null one.
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: object) -> ContentFilterResultSeverity:
        return cls.UNKNOWN


@dataclass
class ContentFilterResult:
    """The result of a content filter check."""

    filtered: bool = False
    detected: bool = False
    severity: ContentFilterResultSeverity = ContentFilterResultSeverity.SAFE

    @classmethod
    def from_inner_error_result(cls, inner_error_results: dict[str, Any]) -> ContentFilterResult:
        """Create a content-filter result from provider error details."""
        return cls(
            filtered=inner_error_results.get("filtered", False),
            detected=inner_error_results.get("detected", False),
            severity=ContentFilterResultSeverity(
                inner_error_results.get("severity", ContentFilterResultSeverity.SAFE.value)
            ),
        )


class ContentFilterCodes(Enum):
    """Content filter codes."""

    RESPONSIBLE_AI_POLICY_VIOLATION = "ResponsibleAIPolicyViolation"
    CONTENT_FILTERED = "ContentFiltered"
    # A code this client does not know, or a null one: the error is still a
    # content-filter error.
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: object) -> ContentFilterCodes:
        return cls.UNKNOWN


@dataclass
class OpenAIContentFilterException(ChatClientContentFilterException):
    """Exception for OpenAI-compatible content-filter errors."""

    param: str | None
    content_filter_code: ContentFilterCodes
    content_filter_result: dict[str, ContentFilterResult]

    def __init__(self, message: str, inner_exception: BadRequestError) -> None:
        super().__init__(message, inner_exception=inner_exception)
        self.param = inner_exception.param
        self.content_filter_code = ContentFilterCodes.RESPONSIBLE_AI_POLICY_VIOLATION
        self.content_filter_result = {}
        if inner_exception.body is not None and isinstance(inner_exception.body, dict):
            inner_error = inner_exception.body.get("innererror", {})
            if isinstance(inner_error, dict):
                self.content_filter_code = ContentFilterCodes(
                    inner_error.get("code", ContentFilterCodes.RESPONSIBLE_AI_POLICY_VIOLATION.value)
                )
                raw_result = inner_error.get("content_filter_result", {})
                if isinstance(raw_result, dict):
                    # OpenAI error bodies are decoded JSON objects, whose keys are strings.
                    self.content_filter_result = {
                        cast(str, key): ContentFilterResult.from_inner_error_result(cast(dict[str, Any], values))
                        for key, values in raw_result.items()
                        if isinstance(values, dict)
                    }

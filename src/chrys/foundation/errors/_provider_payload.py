# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Normalize provider error envelopes and recognize precise gateway verdicts.

Only documented fields in one envelope count. Never recursively search raw
upstream payloads, echoed prompts or arbitrary prose for error keywords.
HTTP status remains the fallback when no precise verdict is available.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .kinds import ErrorKind


def error_text(value: object) -> str | None:
    """An error identifier or message, accepting numeric codes but not booleans."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    return value if isinstance(value, str) and value else None


@dataclass(frozen=True, slots=True)
class ProviderErrorPayload:
    """Protocol-independent fields from a single provider failure."""

    code: str | None = None
    error_type: str | None = None
    message: str | None = None
    # OpenRouter preserves its precise type separately from lossy wire codes.
    canonical_code: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def read(
        cls, body: object, *, code: object = None, error_type: object = None, message: object = None
    ) -> ProviderErrorPayload:
        """Read nested/flat objects or JSON text, with SDK attributes as fallbacks."""
        if isinstance(body, str | bytes):
            try:
                body = json.loads(body)
            except ValueError:
                body = None
        envelope = body if isinstance(body, Mapping) else {}
        nested = envelope.get("error")
        error = nested if isinstance(nested, Mapping) else envelope
        metadata = error.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        return cls(
            code=error_text(error.get("code")) or error_text(code),
            error_type=error_text(error.get("type")) or error_text(error_type),
            message=error_text(error.get("message")) or error_text(nested) or error_text(message),
            canonical_code=(
                error_text(error.get("error_type"))
                or error_text(envelope.get("error_type"))
                or error_text(metadata.get("error_type"))
            ),
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class ProviderVerdict:
    """An explicit semantic verdict stronger than a gateway's HTTP status."""

    kind: ErrorKind
    retryable: bool
    evidence: str


# New API: relaykit/types/error.go and relay/request_billing.go.
# DashScope: https://www.alibabacloud.com/help/en/model-studio/error-code
# Ark: https://console.volcengine.com/ark/region%3Acn-beijing/docs/error-codes
# These are exact codes, including the documented Ark subcodes. A similarly
# named inspection service failure is not a content rejection.
CONTENT_FILTER_CODES = frozenset(
    {
        "content_filter",
        "sensitive_words_detected",
        "prompt_blocked",
        "DataInspectionFailed",
        "data_inspection_failed",
        "SensitiveContentDetected",
        "SensitiveContentDetected.SevereViolation",
        "SensitiveContentDetected.Violence",
        "InputTextSensitiveContentDetected",
        "InputImageSensitiveContentDetected",
        "InputVideoSensitiveContentDetected",
        "InputAudioSensitiveContentDetected",
        "OutputTextSensitiveContentDetected",
        "OutputImageSensitiveContentDetected",
        "OutputVideoSensitiveContentDetected",
        "OutputAudioSensitiveContentDetected",
        "InputTextSensitiveContentDetected.PolicyViolation",
        "InputImageSensitiveContentDetected.PolicyViolation",
        "InputVideoSensitiveContentDetected.PolicyViolation",
        "InputAudioSensitiveContentDetected.PolicyViolation",
        "OutputVideoSensitiveContentDetected.PolicyViolation",
        "OutputAudioSensitiveContentDetected.PolicyViolation",
        "InputImageSensitiveContentDetected.PrivacyInformation",
        "InputVideoSensitiveContentDetected.PrivacyInformation",
        "OutputImageSensitiveContentDetected.DeepFake",
        "InputTextRiskDetection",
        "InputImageRiskDetection",
        "OutputTextRiskDetection",
        "OutputImageRiskDetection",
    }
)
GATEWAY_QUOTA_CODES = frozenset({"insufficient_user_quota", "Arrearage"})

# https://openrouter.ai/docs/api_reference/errors-and-debugging
# These codes are authoritative only in the canonical error_type field;
# generic native types such as invalid_request_error remain status-dependent.
_CANONICAL_KINDS: Mapping[str, ErrorKind] = {
    "content_policy_violation": ErrorKind.CONTENT_FILTERED,
    "refusal": ErrorKind.CONTENT_FILTERED,
    "payment_required": ErrorKind.QUOTA_EXHAUSTED,
    "authentication": ErrorKind.AUTH_FAILED,
    "permission_denied": ErrorKind.AUTH_FAILED,
    "context_length_exceeded": ErrorKind.CONTEXT_OVERFLOW,
    "max_tokens_exceeded": ErrorKind.REQUEST_REJECTED,
    "token_limit_exceeded": ErrorKind.QUOTA_EXHAUSTED,
    "string_too_long": ErrorKind.REQUEST_REJECTED,
    "invalid_request": ErrorKind.REQUEST_REJECTED,
    "invalid_prompt": ErrorKind.REQUEST_REJECTED,
    "not_found": ErrorKind.REQUEST_REJECTED,
    "precondition_failed": ErrorKind.REQUEST_REJECTED,
    "payload_too_large": ErrorKind.PAYLOAD_TOO_LARGE,
    "unprocessable": ErrorKind.REQUEST_REJECTED,
    "invalid_image": ErrorKind.REQUEST_REJECTED,
    "image_too_large": ErrorKind.REQUEST_REJECTED,
    "image_too_small": ErrorKind.REQUEST_REJECTED,
    "unsupported_image_format": ErrorKind.REQUEST_REJECTED,
    "image_not_found": ErrorKind.REQUEST_REJECTED,
    "image_download_failed": ErrorKind.REQUEST_REJECTED,
    "rate_limit_exceeded": ErrorKind.RATE_LIMITED,
    "provider_overloaded": ErrorKind.OVERLOADED,
    "provider_unavailable": ErrorKind.SERVER_ERROR,
    "server": ErrorKind.SERVER_ERROR,
    "timeout": ErrorKind.READ_TIMEOUT,
}
_TRANSIENT_KINDS = frozenset(
    {ErrorKind.RATE_LIMITED, ErrorKind.OVERLOADED, ErrorKind.SERVER_ERROR, ErrorKind.READ_TIMEOUT}
)
_BUDGET_SOURCES = frozenset({"openrouter_guardrail_budget", "openrouter_workspace_budget", "openrouter_key_limit"})
_NEW_API_MESSAGE = re.compile(r"sensitive_words_detected(?: \(request id: [^\r\n()]+\))?")


def gateway_verdict(
    payload: ProviderErrorPayload, status: int | None, retry_after: float | None = None
) -> ProviderVerdict | None:
    """Interpret verified structured codes without treating every 500 as terminal."""
    code, error_type, metadata = payload.code, payload.error_type, payload.metadata
    in_band = status is None or 200 <= status < 300
    limit_source = error_text(metadata.get("limit_source"))
    if (status == 402 or (in_band and code == "402")) and (
        limit_source == "openrouter_in_flight_budget" and retry_after is not None and retry_after > 0
    ):
        return ProviderVerdict(ErrorKind.RATE_LIMITED, True, limit_source)
    if status == 403 or (in_band and code == "403"):
        if limit_source in _BUDGET_SOURCES:
            return ProviderVerdict(ErrorKind.QUOTA_EXHAUSTED, False, f"budget {limit_source}")
        reasons = metadata.get("reasons")
        if (
            isinstance(reasons, list)
            and reasons
            and all(isinstance(reason, str) for reason in reasons)
            and isinstance(metadata.get("flagged_input"), str)
            and isinstance(metadata.get("provider_name"), str)
        ):
            return ProviderVerdict(ErrorKind.CONTENT_FILTERED, False, "moderation metadata")
    # Preserve the overflow veto: a 5xx/429/413 may describe a temporary
    # limit or payload size, not a context window compaction can repair.
    if (
        payload.canonical_code is not None
        and (kind := _CANONICAL_KINDS.get(payload.canonical_code)) is not None
        and (
            kind is not ErrorKind.CONTEXT_OVERFLOW
            or (status not in (413, 429) and not (status is not None and 500 <= status < 600))
        )
    ):
        return ProviderVerdict(kind, kind in _TRANSIENT_KINDS, f"error_type {payload.canonical_code}")
    for value in (code, error_type):
        if value in CONTENT_FILTER_CODES:
            return ProviderVerdict(ErrorKind.CONTENT_FILTERED, False, f"content code {value}")
        if value in GATEWAY_QUOTA_CODES:
            return ProviderVerdict(ErrorKind.QUOTA_EXHAUSTED, False, f"quota code {value}")
    # The Anthropic envelope drops New API's code. Older versions put the
    # exact code in message; newer versions include the matched words.
    message = payload.message or ""
    if error_type == "new_api_error" and (
        _NEW_API_MESSAGE.fullmatch(message) or message.startswith("user sensitive words detected: ")
    ):
        return ProviderVerdict(ErrorKind.CONTENT_FILTERED, False, "new_api_error sensitive_words_detected")
    return None

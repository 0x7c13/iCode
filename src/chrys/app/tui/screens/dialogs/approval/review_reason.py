# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Localized approval concerns from structured judge results."""

from __future__ import annotations

from chrys.app.tui.i18n import render_str
from chrys.foundation.i18n import Localizer, msg
from chrys.service.approval.judge import JudgeVerdict

_EXTERNAL = msg("tui.approval.risk.external_action", fallback="External publication or transfer")
_DESTRUCTIVE = msg("tui.approval.risk.destructive_action", fallback="Deletion or destructive changes")
_SHARED = msg("tui.approval.risk.shared_mutation", fallback="Changes to shared or production resources")
_SENSITIVE = msg("tui.approval.risk.sensitive_data", fallback="Sensitive data or account operations")
_SECURITY = msg("tui.approval.risk.security_boundary", fallback="Security or permission changes")
_UNTRUSTED = msg("tui.approval.risk.untrusted_execution", fallback="Untrusted code or externally exposed services")
_SCOPE = msg("tui.approval.risk.scope_escalation", fallback="Actions beyond the requested scope")
_RISKS = {
    "external_action": _EXTERNAL,
    "destructive_action": _DESTRUCTIVE,
    "high_impact_shared_or_production_mutation": _SHARED,
    "sensitive_data_access_or_operation": _SENSITIVE,
    "security_boundary_change": _SECURITY,
    "untrusted_execution_or_exposure": _UNTRUSTED,
    "scope_escalation": _SCOPE,
}
_TRIGGERED = msg("tui.approval.review.triggered", fallback="Requires human confirmation: {conditions}.")
_FAILED = msg("tui.approval.review.failed", fallback="Automatic review failed. Please check the action.")
_INPUT = msg("tui.approval.review.invalid_input", fallback="There is not enough valid context to review this action.")
_ASSET = msg("tui.approval.review.asset_unavailable", fallback="The risk definitions could not be loaded.")
_TIMEOUT = msg("tui.approval.review.timeout", fallback="Automatic review timed out.")
_SHAPE = msg("tui.approval.review.invalid_shape", fallback="The risk check returned an invalid response.")
_VERDICT = msg("tui.approval.review.invalid_verdict", fallback="The approval model returned an invalid response.")
_AUTH = msg(
    "tui.approval.review.auth_failed",
    fallback="The approval model rejected authentication or access. Check the API key and model permissions.",
)
_RATE_LIMITED = msg(
    "tui.approval.review.rate_limited",
    fallback="The approval model is rate limiting requests. Please try again later.",
)
_SERVER = msg(
    "tui.approval.review.server_error",
    fallback="The approval model service returned an internal error. Please try again later.",
)
_FAILURES = {
    "invalid_input": _INPUT,
    "asset_unavailable": _ASSET,
    "timeout": _TIMEOUT,
    "invalid_shape": _SHAPE,
    "invalid_verdict": _VERDICT,
    "auth_failed": _AUTH,
    "rate_limited": _RATE_LIMITED,
    "server_error": _SERVER,
}


def review_reason(verdict: JudgeVerdict, localizer: Localizer) -> str:
    """Render program-generated concerns; model-written reasons remain verbatim."""
    if verdict.failure_reason:
        return render_str(localizer, _FAILURES.get(verdict.failure_reason, _FAILED).bind())
    if verdict.triggered:
        conditions = ", ".join(render_str(localizer, _RISKS[key].bind()) for key in verdict.triggered if key in _RISKS)
        return render_str(localizer, _TRIGGERED.bind(conditions=conditions)) if conditions else verdict.reason
    return verdict.reason

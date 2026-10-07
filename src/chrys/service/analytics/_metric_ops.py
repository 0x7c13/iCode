# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Metric arithmetic that carries precision and reasons along with the value."""

from __future__ import annotations

from typing import cast

from chrys.service.analytics.model import Metric, Precision, least_precision


def _sum_metrics(
    metrics: list[Metric],
    *,
    missing_is_missing: bool = False,
    metric_subject: str = "selected turns",
) -> Metric:
    if any(metric.value is None for metric in metrics):
        missing = any(metric.precision is Precision.MISSING for metric in metrics)
        precision = Precision.MISSING if missing_is_missing and missing else Precision.UNRESOLVED
        reason = next((metric.reason for metric in metrics if metric.value is None and metric.reason is not None), None)
        return Metric(
            None,
            precision,
            reason or f"one or more {metric_subject} are unresolved",
        )
    values = [metric.value for metric in metrics if metric.value is not None]
    precision = least_precision(metric.precision for metric in metrics)
    return Metric(
        sum(values),
        precision,
        None if precision is Precision.EXACT else f"one or more {metric_subject} are not exact",
    )


def _sum_optional_bucket_metrics(metrics: list[Metric], *, metric_subject: str = "selected turns") -> Metric:
    """Sum an optional bucket across turns without letting absence poison it."""
    reported = [metric for metric in metrics if metric.value is not None]
    if not reported:
        reason = next((metric.reason for metric in metrics if metric.reason is not None), None)
        return Metric(None, Precision.MISSING, reason or "no exchange reported this bucket")
    value = sum(cast("int | float", metric.value) for metric in reported)
    if any(metric.precision is Precision.UNRESOLVED for metric in metrics):
        return Metric(value, Precision.UNRESOLVED, f"one or more {metric_subject} are unresolved")
    if len(reported) < len(metrics) or any(metric.precision is Precision.ESTIMATED for metric in reported):
        return Metric(value, Precision.ESTIMATED, "not every exchange reported this bucket")
    return Metric(value, Precision.EXACT)


def _percentile_metric(values: list[int], quantile: float) -> Metric:
    if not values:
        return Metric(None, Precision.MISSING, "no resolved samples")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) * quantile) + 0.999999999) - 1))
    return Metric(ordered[index], Precision.EXACT)


def _with_unresolved_precision(metric: Metric, reason: str) -> Metric:
    return Metric(metric.value, Precision.UNRESOLVED, reason)


def _cap_session_metric(metric: Metric, reason: str) -> Metric:
    if metric.precision is Precision.MISSING:
        return metric
    if metric.precision is Precision.UNRESOLVED and metric.reason is not None:
        return metric
    return Metric(metric.value, Precision.UNRESOLVED, _merge_reasons(metric.reason, reason))


def _cap_session_precision(
    precision: Precision,
    existing_reason: str | None,
    integrity_reason: str,
) -> tuple[Precision, str | None]:
    if precision is Precision.MISSING:
        return precision, existing_reason
    if precision is Precision.UNRESOLVED and existing_reason is not None:
        return precision, existing_reason
    return Precision.UNRESOLVED, _merge_reasons(existing_reason, integrity_reason)


def _merge_reasons(*reasons: str | None) -> str:
    return "; ".join(dict.fromkeys(reason for reason in reasons if reason))

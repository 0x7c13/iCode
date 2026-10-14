# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Metric arithmetic that carries precision and reasons along with the value."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from chrys.service.analytics.model import Grade, Metric, Precision, least_precision

if TYPE_CHECKING:
    from collections.abc import Iterable


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
    # The parts' own reasons let the total name its root cause rather than
    # only pointing back at the parts.
    return _graded_metric(
        sum(values),
        (metric.grade for metric in metrics),
        fallback=f"one or more {metric_subject} are not exact",
    )


def _graded_metric(
    value: int | float,
    grades: Iterable[Grade],
    *,
    fallback: str | None = None,
) -> Metric:
    """*value* at the least precision among *grades*, with the reasons of the grades that set it."""
    grades = tuple(grades)
    precision = least_precision(grade.precision for grade in grades)
    if precision is Precision.EXACT:
        return Metric(value, precision)
    # A part at a better grade has another cause; naming it here would split
    # this metric's diagnostics group from the parts that share its cause.
    return Metric(
        value,
        precision,
        _merge_reasons(*(grade.reason for grade in grades if grade.precision is precision)) or fallback,
    )


def _sum_optional_bucket_metrics(metrics: list[Metric], *, metric_subject: str = "selected turns") -> Metric:
    """Sum an optional bucket across turns without letting absence poison it."""
    reported = [metric for metric in metrics if metric.value is not None]
    if not reported:
        reason = next((metric.reason for metric in metrics if metric.reason is not None), None)
        return Metric(None, Precision.MISSING, reason or "no exchange reported this bucket")
    value = sum(cast("int | float", metric.value) for metric in reported)
    if any(metric.precision is Precision.UNRESOLVED for metric in metrics):
        reason = _merge_reasons(*(metric.reason for metric in metrics if metric.precision is Precision.UNRESOLVED))
        return Metric(value, Precision.UNRESOLVED, reason or f"one or more {metric_subject} are unresolved")
    if len(reported) < len(metrics) or any(metric.precision is Precision.ESTIMATED for metric in reported):
        reasons = (
            *(metric.reason for metric in reported if metric.precision is Precision.ESTIMATED),
            "not every exchange reported this bucket" if len(reported) < len(metrics) else None,
        )
        return Metric(value, Precision.ESTIMATED, _merge_reasons(*reasons) or "not every exchange reported this bucket")
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


def _cap_session_grade(grade: Grade, integrity_reason: str) -> Grade:
    if grade.precision is Precision.MISSING:
        return grade
    if grade.precision is Precision.UNRESOLVED and grade.reason is not None:
        return grade
    return Grade(Precision.UNRESOLVED, _merge_reasons(grade.reason, integrity_reason))


def _merge_reasons(*reasons: str | None) -> str:
    # A reason may already be a merge of others; splitting it keeps one cause
    # from being listed twice, which would split its diagnostics group.
    return "; ".join(dict.fromkeys(part for reason in reasons if reason for part in reason.split("; ")))

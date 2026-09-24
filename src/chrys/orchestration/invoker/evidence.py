# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Scope-bound observations; wire snapshots are never invocation totals."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Completeness(Enum):
    EXACT = "exact"
    LOWER_BOUND = "lower_bound"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Count:
    observed: int
    completeness: Completeness

    def __post_init__(self) -> None:
        if self.observed < 0 or (self.completeness is Completeness.UNKNOWN and self.observed != 0):
            raise ValueError("Counts must be nonnegative; UNKNOWN has no observed value")

    def __add__(self, other: Count) -> Count:
        observed = self.observed + other.observed
        if self.completeness is other.completeness is Completeness.EXACT:
            completeness = Completeness.EXACT
        elif Completeness.LOWER_BOUND in (self.completeness, other.completeness) or observed:
            completeness = Completeness.LOWER_BOUND
        else:
            completeness = Completeness.UNKNOWN
        return Count(observed, completeness)


UNKNOWN_COUNT = Count(0, Completeness.UNKNOWN)
ZERO_COUNT = Count(0, Completeness.EXACT)


def hosted_count(labels: tuple[str, ...] | None) -> Count:
    """One hosted operation may have multiple labels (shell + shell result)."""
    return UNKNOWN_COUNT if labels is None else Count(int(bool(labels)), Completeness.LOWER_BOUND)


@dataclass(frozen=True, slots=True)
class PassEvidence:
    invocation_id: str
    pass_id: str
    local_dispatched: Count
    local_answered: Count
    hosted_observed: Count
    external_stateful: bool | None


@dataclass(frozen=True, slots=True)
class WireSnapshot:
    invocation_id: str
    pass_id: str
    wire_attempt: int
    hosted_in_flight: Count


@dataclass(frozen=True, slots=True)
class InvocationEvidence:
    """Caller-owned accumulation; each converged pass is accepted exactly once."""

    invocation_id: str
    passes: tuple[str, ...] = ()
    local_dispatched: Count = ZERO_COUNT
    local_answered: Count = ZERO_COUNT
    hosted_observed: Count = ZERO_COUNT
    external_stateful: bool | None = False

    def add(self, evidence: PassEvidence) -> InvocationEvidence:
        if not isinstance(evidence, PassEvidence):
            raise TypeError("Only pass evidence may be accumulated")
        if evidence.invocation_id != self.invocation_id or evidence.pass_id in self.passes:
            raise ValueError("Foreign or already accumulated pass")
        stateful = (
            True
            if self.external_stateful is True or evidence.external_stateful is True
            else False
            if self.external_stateful is False and evidence.external_stateful is False
            else None
        )
        return InvocationEvidence(
            self.invocation_id,
            (*self.passes, evidence.pass_id),
            self.local_dispatched + evidence.local_dispatched,
            self.local_answered + evidence.local_answered,
            self.hosted_observed + evidence.hosted_observed,
            stateful,
        )

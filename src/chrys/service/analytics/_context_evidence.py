# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Verified context evidence: replayed revision memberships and compaction consumption.

Revision membership is rebuilt from checkpoint/delta records and accepted only
when the replay is consistent (valid occurrences, matching item count) and any
declared membership hash verifies. Phase-4 compaction consumption names the
items a compaction run folded and the already-reserved successor exchange whose
request that run compacted. This module proves membership only;
``_turn_graph`` decides which causal edges that evidence supports.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import cast

from chrys.foundation.trajectory.ids import is_valid_analytics_id
from chrys.foundation.trajectory.revisions import MembershipRef, membership_hash
from chrys.service.analytics._facts import (
    _active,
    _Endpoint,
    _Intermediate,
    _payload_bool,
    _payload_int,
    _payload_str,
    _RevisionEntry,
    _Segment,
)
from chrys.service.analytics._timeline import _endpoint_in_coverage_runtime, _ResolvedNode
from chrys.service.analytics.reader import raise_if_cancelled as _check_cancelled


@dataclass(frozen=True, slots=True)
class _RevisionResolution:
    memberships: dict[str, tuple[str, ...]]
    endpoints: dict[str, _Endpoint]
    errors: dict[str, tuple[str, ...]]
    side_call_empty_shell_revisions: tuple[str, ...]
    # Replayed and hash-verified, but the producer reported items it could not
    # name: membership is a lower bound, good for positive claims only.
    unidentified_membership_revision_count: int = 0


@dataclass(frozen=True, slots=True)
class _Phase4Consumption:
    item_ids: frozenset[str]
    next_exchange_operation_id: str


@dataclass(frozen=True, slots=True)
class _CompactionConsumptionResolution:
    by_run: dict[str, _Phase4Consumption]
    exact: bool


def _revision_memberships(
    intermediate: _Intermediate,
    inactive_ranges: tuple[tuple[int, int], ...],
    *,
    fingerprint_key: bytes | None,
    cancel_event: Event | None,
) -> _RevisionResolution:
    ordered_memberships: dict[str, tuple[MembershipRef, ...]] = {}
    endpoints: dict[str, _Endpoint] = {}
    errors: dict[str, tuple[str, ...]] = {}
    side_call_empty_shell_revisions: list[str] = []
    unidentified_membership_revisions = 0
    definitions = {
        revision_id: [endpoint for endpoint in candidates if _active(endpoint.sequence, inactive_ranges)]
        for revision_id, candidates in intermediate.context_revisions.items()
    }
    for revision_id, candidates in definitions.items():
        _check_cancelled(cancel_event)
        if len(candidates) > 1:
            errors[revision_id] = (f"context revision {revision_id} is defined more than once",)
        elif candidates:
            endpoints[revision_id] = candidates[0]
    for revision_id, endpoint in sorted(endpoints.items(), key=lambda item: item[1].sequence):
        _check_cancelled(cancel_event)
        if revision_id in errors:
            continue
        revision_errors: list[str] = []
        membership: tuple[MembershipRef, ...] | None = None
        entries = None
        if not _endpoint_in_coverage_runtime(intermediate, endpoint, inactive_ranges):
            revision_errors.append(f"context revision {revision_id} falls outside coverage or after runtime closure")
        else:
            entries = _segmented_revision_entries(intermediate, endpoint, inactive_ranges, cancel_event=cancel_event)
        if entries is None:
            revision_errors.append(f"context revision {revision_id} has invalid segmented membership")
        elif _payload_bool(endpoint.payload, "is_checkpoint"):
            membership = _replay_checkpoint(entries)
            if membership is None:
                revision_errors.append(f"context revision {revision_id} has invalid checkpoint membership")
        else:
            parent_id = _payload_str(endpoint.payload, "parent_revision_id")
            parent = endpoints.get(parent_id or "")
            parent_membership = ordered_memberships.get(parent_id or "")
            if (
                parent_id is None
                or parent is None
                or parent_membership is None
                or parent.sequence >= endpoint.sequence
                or parent.actor_id is None
                or parent.actor_id != endpoint.actor_id
                or parent.runtime_id != endpoint.runtime_id
                or parent.branch_id != endpoint.branch_id
            ):
                revision_errors.append(f"context revision {revision_id} has no valid same-actor/runtime parent")
                membership = None
            else:
                membership = _replay_delta(parent_membership, entries)
                if membership is None:
                    revision_errors.append(f"context revision {revision_id} has an invalid delta replay")
        if entries is not None and membership is not None:
            item_count = _payload_int(endpoint.payload, "item_count")
            unidentified = _payload_int(endpoint.payload, "unidentified_item_count")
            if item_count is None or item_count != len(membership):
                revision_errors.append(f"context revision {revision_id} item_count does not match replayed membership")
            if unidentified is None or unidentified < 0:
                revision_errors.append(f"context revision {revision_id} has an invalid unidentified item count")
            elif unidentified and endpoint.side_call and item_count == 0:
                # The known empty-shell shape of a side call: nothing named,
                # nothing to replay; its volume travels through exchange usage.
                side_call_empty_shell_revisions.append(revision_id)
            elif unidentified:
                # The named items replay and verify; the request also carried
                # items nothing can name, so the membership only supports
                # positive claims and the count is surfaced as information.
                unidentified_membership_revisions += 1
            expected_hash = _payload_str(endpoint.payload, "membership_hash")
            if expected_hash is not None:
                actual_hash = membership_hash(fingerprint_key, membership)
                if actual_hash is None:
                    revision_errors.append(f"context revision {revision_id} membership_hash cannot be verified")
                elif actual_hash != expected_hash:
                    revision_errors.append(f"context revision {revision_id} membership_hash does not match replay")
        if revision_errors:
            errors[revision_id] = tuple(revision_errors)
            continue
        if membership is None:
            raise RuntimeError("A validated context revision has no resolved membership.")
        ordered_memberships[revision_id] = membership
    return _RevisionResolution(
        memberships={
            revision_id: tuple(item_id for item_id, _ in membership)
            for revision_id, membership in ordered_memberships.items()
        },
        endpoints=endpoints,
        errors=errors,
        side_call_empty_shell_revisions=tuple(side_call_empty_shell_revisions),
        unidentified_membership_revision_count=unidentified_membership_revisions,
    )


def _segmented_revision_entries(
    intermediate: _Intermediate,
    endpoint: _Endpoint,
    inactive_ranges: tuple[tuple[int, int], ...],
    *,
    cancel_event: Event | None,
) -> tuple[_RevisionEntry, ...] | None:
    field_pointer = "/payload/refs"
    declarations = [item for item in endpoint.segmented_fields if item.field_pointer == field_pointer]
    if len(declarations) != 1 or endpoint.event_id is None:
        return None
    declaration = declarations[0]
    segments = [
        segment
        for segment in intermediate.segments.get(endpoint.event_id, ())
        if _active(segment.sequence, inactive_ranges) and segment.field_pointer == field_pointer
    ]
    if len(segments) != declaration.segment_count:
        return None
    indexed_segments: list[tuple[int, _Segment]] = []
    for segment in segments:
        _check_cancelled(cancel_event)
        segment_index = segment.segment_index
        if segment_index is None:
            return None
        indexed_segments.append((segment_index, segment))
    if sorted(segment_index for segment_index, _ in indexed_segments) != list(range(declaration.segment_count)):
        return None
    entries: list[_RevisionEntry] = []
    for _, segment in sorted(indexed_segments):
        _check_cancelled(cancel_event)
        if (
            segment.segment_count != declaration.segment_count
            or segment.segment_group_id != declaration.segment_group_id
            or segment.encoding != "array_slice"
            or segment.entry_oversized
            or segment.runtime_id != endpoint.runtime_id
            or segment.branch_id != endpoint.branch_id
            or segment.coverage_id != endpoint.coverage_id
            or segment.turn_id != endpoint.turn_id
            or segment.sequence <= endpoint.sequence
            or not _endpoint_in_coverage_runtime(intermediate, segment, inactive_ranges)
        ):
            return None
        values = segment.entries
        if values is None or not all(isinstance(value, _RevisionEntry) for value in values):
            return None
        entries.extend(value for value in values if isinstance(value, _RevisionEntry))
    return tuple(entries)


def _replay_checkpoint(entries: tuple[_RevisionEntry, ...]) -> tuple[MembershipRef, ...] | None:
    if any(entry.action != "add" for entry in entries):
        return None
    ordered = sorted(entries, key=lambda entry: entry.position)
    if [entry.position for entry in ordered] != list(range(len(ordered))):
        return None
    membership = tuple((entry.item_id, entry.occurrence) for entry in ordered)
    return membership if _valid_membership_occurrences(membership) else None


def _replay_delta(
    parent: tuple[MembershipRef, ...],
    entries: tuple[_RevisionEntry, ...],
) -> tuple[MembershipRef, ...] | None:
    removals = [entry for entry in entries if entry.action == "remove"]
    additions = [entry for entry in entries if entry.action == "add"]
    if len(removals) + len(additions) != len(entries):
        return None
    removal_positions = [entry.position for entry in removals]
    if len(removal_positions) != len(set(removal_positions)):
        return None
    for entry in removals:
        if entry.position >= len(parent) or parent[entry.position] != (entry.item_id, entry.occurrence):
            return None
    replayed = list(parent)
    for entry in sorted(removals, key=lambda item: item.position, reverse=True):
        del replayed[entry.position]
    addition_positions = [entry.position for entry in additions]
    if len(addition_positions) != len(set(addition_positions)):
        return None
    for entry in sorted(additions, key=lambda item: item.position):
        ref = (entry.item_id, entry.occurrence)
        if entry.position > len(replayed) or ref in replayed:
            return None
        replayed.insert(entry.position, ref)
    membership = tuple(replayed)
    return membership if _valid_membership_occurrences(membership) else None


def _valid_membership_occurrences(membership: tuple[MembershipRef, ...]) -> bool:
    seen: dict[str, int] = {}
    for item_id, occurrence in membership:
        if occurrence != seen.get(item_id, 0):
            return False
        seen[item_id] = occurrence + 1
    return True


def _compaction_consumption_resolution(
    intermediate: _Intermediate,
    nodes: list[_ResolvedNode],
    inactive_ranges: tuple[tuple[int, int], ...],
    *,
    cancel_event: Event | None,
    diagnostics: list[str],
) -> _CompactionConsumptionResolution:
    by_run: dict[str, _Phase4Consumption] = {}
    seen_runs: set[str] = set()
    exact = True
    for phase in (node for node in nodes if node.family == "compaction.phase"):
        _check_cancelled(cancel_event)
        if _payload_str(phase.finish.payload, "phase") != "phase4":
            continue
        endpoint = phase.finish
        run_id = endpoint.parent_operation_id
        if run_id is None:
            diagnostics.append("Phase-4 consumed item declaration lacks a compaction run")
            exact = False
            continue
        if run_id in seen_runs:
            diagnostics.append("compaction run declares more than one Phase-4 consumption event")
            by_run.pop(run_id, None)
            exact = False
            continue
        seen_runs.add(run_id)
        entries = _segmented_string_entries(
            intermediate,
            endpoint,
            inactive_ranges,
            field_pointer="/payload/consumed_item_ids",
            cancel_event=cancel_event,
        )
        if entries is None:
            diagnostics.append("Phase-4 compaction has invalid segmented consumed item ids")
            exact = False
            continue
        next_exchange = _payload_str(endpoint.payload, "next_exchange_operation_id")
        if next_exchange is None:
            diagnostics.append("Phase-4 compaction lacks next_exchange_operation_id")
            exact = False
            continue
        by_run[run_id] = _Phase4Consumption(
            item_ids=frozenset(entries),
            next_exchange_operation_id=next_exchange,
        )
    return _CompactionConsumptionResolution(
        by_run=by_run,
        exact=exact,
    )


def _segmented_string_entries(
    intermediate: _Intermediate,
    endpoint: _Endpoint,
    inactive_ranges: tuple[tuple[int, int], ...],
    *,
    field_pointer: str,
    cancel_event: Event | None,
) -> tuple[str, ...] | None:
    declarations = [item for item in endpoint.segmented_fields if item.field_pointer == field_pointer]
    if len(declarations) != 1 or endpoint.event_id is None:
        return None
    declaration = declarations[0]
    segments = [
        segment
        for segment in intermediate.segments.get(endpoint.event_id, ())
        if _active(segment.sequence, inactive_ranges) and segment.field_pointer == field_pointer
    ]
    if len(segments) != declaration.segment_count:
        return None
    indexed_segments: list[tuple[int, _Segment]] = []
    for segment in segments:
        _check_cancelled(cancel_event)
        if segment.segment_index is None:
            return None
        indexed_segments.append((segment.segment_index, segment))
    if sorted(index for index, _ in indexed_segments) != list(range(declaration.segment_count)):
        return None
    entries: list[str] = []
    for _, segment in sorted(indexed_segments):
        _check_cancelled(cancel_event)
        if (
            segment.segment_count != declaration.segment_count
            or segment.segment_group_id != declaration.segment_group_id
            or segment.encoding != "array_slice"
            or segment.entry_oversized
            or segment.runtime_id != endpoint.runtime_id
            or segment.branch_id != endpoint.branch_id
            or segment.coverage_id != endpoint.coverage_id
            or segment.turn_id != endpoint.turn_id
            or segment.sequence <= endpoint.sequence
            or not _endpoint_in_coverage_runtime(intermediate, segment, inactive_ranges)
        ):
            return None
        values = segment.entries
        if values is None or not all(isinstance(value, str) and is_valid_analytics_id(value) for value in values):
            return None
        entries.extend(cast("str", value) for value in values)
    return tuple(entries)

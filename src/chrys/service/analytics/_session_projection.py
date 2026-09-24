# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Lazy projection of persisted session state into analytics joins."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any, Final

from chrys.foundation.trajectory.metadata import read_analytics_item_id
from chrys.kernel.exchanges import (
    LEGACY_CALL_CONTENT_TYPE,
    TOOL_CALL_CONTENT_TYPES,
    TOOL_RESULT_CONTENT_TYPES,
    DictAccessor,
    EmptyIdPolicy,
    NoneIdPolicy,
    PairingPolicy,
    iter_exchanges,
    pair_results,
)
from chrys.service.analytics.reader import plain_int_value as _plain_int_value
from chrys.service.analytics.reader import raise_if_cancelled as _check_cancelled
from chrys.service.mutations.types import parse_skip_reason

_PROJECTION_PAIRING_POLICY: Final = PairingPolicy(
    call_types=TOOL_CALL_CONTENT_TYPES | {LEGACY_CALL_CONTENT_TYPE},
    include_informational_calls=True,
    result_types=TOOL_RESULT_CONTENT_TYPES,
    none_id=NoneIdPolicy.POSITIONAL,
    empty_id=EmptyIdPolicy.POSITIONAL,
    malformed_id="treat_as_none",
)
"""Labels tool results in the session projection by the call they answered."""

_CONTENT_HASH_RE: Final = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class _SessionMutation:
    turn_number: int
    path: str
    old_path: str | None
    before_hash: str | None
    after_hash: str | None
    before_skip: str | None
    after_skip: str | None
    provenance: str | None
    contested: bool
    # A move's before_hash describes the destination; the source's pre-state
    # comes from its own turn snapshot.
    old_before_hash: str | None
    old_before_skip: str | None


@dataclass(frozen=True, slots=True)
class _SessionProjection:
    available: bool
    # A parseable session document does not imply its mutation detail is
    # present; change verification must not read an absent or malformed
    # ``chrys_mutations`` as an exact zero-mutation session.
    mutation_detail_available: bool = False
    carriers: dict[str, str] = field(default_factory=dict)
    commands: dict[str, str] = field(default_factory=dict)
    skill_names: dict[str, str] = field(default_factory=dict)
    item_tokens: dict[str, int] = field(default_factory=dict)
    item_roles: dict[str, str] = field(default_factory=dict)
    item_tool_names: dict[str, tuple[str, ...]] = field(default_factory=dict)
    mutations: tuple[_SessionMutation, ...] = ()
    detection_truncated_turns: frozenset[int] = frozenset()


def _lazy_session_projection(
    path: Path,
    *,
    cancel_event: Event | None = None,
) -> Callable[[], _SessionProjection]:
    loaded: _SessionProjection | None = None

    def resolve() -> _SessionProjection:
        nonlocal loaded
        if loaded is None:
            loaded = _read_session_projection(path, cancel_event=cancel_event)
        return loaded

    return resolve


def _session_document_path(path: Path) -> Path | None:
    """Return the session document beside *path*, or None for a loose log."""
    if (
        len(path.parents) < 4
        or path.name != "events.jsonl"
        or path.parent.name != "trajectory"
        or path.parents[2].name != "sessions"
    ):
        return None
    return path.parents[1] / "session.json"


class _SessionProjectionCache:
    """Reuse the parsed session document across analysis generations.

    The live dashboard refreshes twice a second; the events scan is
    incremental, but the companion ``session.json`` parse is not, so an
    unchanged document must not be reparsed on every append batch. The
    store replaces the document whole, so one (device, inode, size,
    mtime) signature identifies one written state; a stat failure or a
    loose log disables reuse instead of trusting a stale parse. The stat
    is taken before the read: a document replaced between the two pairs
    the old signature with the newer parse, which the next generation's
    mismatching stat corrects.
    """

    def __init__(self) -> None:
        self._signature: tuple[int, int, int, int] | None = None
        self._projection: _SessionProjection | None = None

    def lazy(self, path: Path, *, cancel_event: Event | None = None) -> Callable[[], _SessionProjection]:
        loaded: _SessionProjection | None = None

        def resolve() -> _SessionProjection:
            nonlocal loaded
            if loaded is None:
                loaded = self._resolve(path, cancel_event=cancel_event)
            return loaded

        return resolve

    def changed(self, path: Path) -> bool:
        """Whether the on-disk document no longer matches the last parse.

        The analyzer's unchanged-log fast path may return a cached
        analysis only while the companion document it folded is still the
        written state; one that appeared, vanished, or was replaced since
        invalidates it. Unreadable on both sides projects the same
        unavailable state and keeps the fast path.
        """
        return self._stat_signature(path) != self._signature

    def _resolve(self, path: Path, *, cancel_event: Event | None) -> _SessionProjection:
        signature = self._stat_signature(path)
        if signature is not None and signature == self._signature and self._projection is not None:
            return self._projection
        projection = _read_session_projection(path, cancel_event=cancel_event)
        self._signature = signature
        self._projection = projection if signature is not None else None
        return projection

    @staticmethod
    def _stat_signature(path: Path) -> tuple[int, int, int, int] | None:
        document = _session_document_path(path)
        if document is None:
            return None
        try:
            stat = document.stat()
        except OSError:
            return None
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _read_session_projection(path: Path, *, cancel_event: Event | None = None) -> _SessionProjection:
    _check_cancelled(cancel_event)
    document = _session_document_path(path)
    if document is None:
        return _SessionProjection(available=False)
    try:
        with document.open("rb") as handle:
            envelope = json.load(handle)
    except OSError, ValueError, RecursionError:
        # ValueError is the decoder's whole failure family: JSONDecodeError,
        # UnicodeDecodeError (json.load decodes the raw bytes itself), and the
        # bare ValueError an integer token past the digit limit raises;
        # recursion-deep nesting is the one failure outside it. None is a
        # document the session store writes, and all mean the session document
        # cannot back the projection.
        return _SessionProjection(available=False)
    if not isinstance(envelope, dict) or not isinstance((state := envelope.get("state")), dict):
        return _SessionProjection(available=False)

    def transcript(value: object) -> list[dict[Any, Any]]:
        return [message for message in value if isinstance(message, dict)] if isinstance(value, list) else []

    # Compaction moves original messages into ``compressed_msgs[*].messages``
    # and leaves only a summary in the live list; each block is scanned as a
    # transcript of its own so compacted turns keep their carrier, argument,
    # and token joins.
    raw_blocks = state.get("compressed_msgs")
    transcripts = [
        transcript(block.get("messages"))
        for block in (raw_blocks if isinstance(raw_blocks, list) else [])
        if isinstance(block, dict)
    ]
    transcripts.append(transcript(state.get("messages")))
    carriers: dict[str, str] = {}
    ambiguous_carriers: set[str] = set()
    commands: dict[str, str] = {}
    ambiguous_commands: set[str] = set()
    skill_names: dict[str, str] = {}
    ambiguous_skill_names: set[str] = set()
    item_tokens: dict[str, int] = {}
    ambiguous_item_tokens: set[str] = set()
    item_roles: dict[str, str] = {}
    ambiguous_item_roles: set[str] = set()
    tool_names: dict[str, tuple[str, ...]] = {}
    ambiguous_tool_names: set[str] = set()
    for messages in transcripts:
        carrier_by_index: dict[int, str] = {}
        transcript_tool_names: dict[str, list[str]] = {}
        for message_index, message in enumerate(messages):
            _check_cancelled(cancel_event)
            properties = message.get("additional_properties")
            carrier_id = read_analytics_item_id(properties)
            if carrier_id is not None and isinstance(properties, dict):
                carrier_by_index[message_index] = carrier_id
                group = properties.get("_group")
                token_count = group.get("token_count") if isinstance(group, dict) else None
                if (
                    isinstance(token_count, int)
                    and not isinstance(token_count, bool)
                    and token_count >= 0
                    and carrier_id not in ambiguous_item_tokens
                ):
                    existing_tokens = item_tokens.get(carrier_id)
                    if existing_tokens is not None and existing_tokens != token_count:
                        item_tokens.pop(carrier_id)
                        ambiguous_item_tokens.add(carrier_id)
                    else:
                        item_tokens[carrier_id] = token_count
                role = message.get("role")
                if isinstance(role, str) and carrier_id not in ambiguous_item_roles:
                    existing_role = item_roles.get(carrier_id)
                    if existing_role is not None and existing_role != role:
                        item_roles.pop(carrier_id)
                        ambiguous_item_roles.add(carrier_id)
                    else:
                        item_roles[carrier_id] = role
            contents = message.get("contents")
            if not isinstance(contents, list):
                continue
            for content in contents:
                _check_cancelled(cancel_event)
                if not isinstance(content, dict):
                    continue
                content_id = read_analytics_item_id(content.get("additional_properties"))
                if content_id is not None and carrier_id is not None and content_id not in ambiguous_carriers:
                    existing_carrier = carriers.get(content_id)
                    if existing_carrier is not None and existing_carrier != carrier_id:
                        carriers.pop(content_id)
                        ambiguous_carriers.add(content_id)
                    else:
                        carriers[content_id] = carrier_id
                if content.get("type") != "function_call":
                    continue
                name = content.get("name")
                if isinstance(name, str) and carrier_id is not None:
                    transcript_tool_names.setdefault(carrier_id, []).append(name)
                if content_id is None:
                    continue
                arguments = content.get("arguments")
                if isinstance(arguments, dict):
                    # Anthropic blocking responses persist the tool input as
                    # the decoded mapping rather than a JSON string.
                    decoded: object = arguments
                else:
                    try:
                        decoded = json.loads(arguments) if isinstance(arguments, str) else None
                    except ValueError, RecursionError:
                        # Model-authored arguments persist verbatim, so a
                        # decoder limit — nesting past its recursion budget,
                        # an integer token past the digit limit — is reachable
                        # even when the outer document decoded fine.
                        decoded = None
                command = decoded.get("command") if isinstance(decoded, dict) else None
                if isinstance(command, str) and content_id not in ambiguous_commands:
                    existing_command = commands.get(content_id)
                    if existing_command is not None and existing_command != command:
                        commands.pop(content_id)
                        ambiguous_commands.add(content_id)
                    else:
                        commands[content_id] = command
                skill_name = decoded.get("skill_name") if isinstance(decoded, dict) else None
                if isinstance(skill_name, str) and content_id not in ambiguous_skill_names:
                    existing_skill_name = skill_names.get(content_id)
                    if existing_skill_name is not None and existing_skill_name != skill_name:
                        skill_names.pop(content_id)
                        ambiguous_skill_names.add(content_id)
                    else:
                        skill_names[content_id] = skill_name
        # Tool results carry no name of their own; the exchange grammar pairs
        # each one with the call it answered so the result's carrier is
        # labelled by tool. Calls never pair across a compaction boundary: a
        # call and its result share a turn, and compaction moves whole turns.
        accessor = DictAccessor()
        for exchange in iter_exchanges(messages, accessor):
            _check_cancelled(cancel_event)
            pairing = pair_results(messages, exchange, accessor, _PROJECTION_PAIRING_POLICY)
            for assignments in (*pairing.truthy_assignments.values(), *pairing.falsy_assignments.values()):
                for call, result in assignments:
                    if result is None or (carrier_id := carrier_by_index.get(result.message_index)) is None:
                        continue
                    name = accessor.contents(messages[call.message_index])[call.content_index].get("name")
                    if isinstance(name, str):
                        transcript_tool_names.setdefault(carrier_id, []).append(name)
        # A message can sit in both a compressed block and the live list, so
        # equal per-transcript name tuples describe one message and collapse;
        # transcripts that disagree mean the carrier identity is unreliable
        # and it keeps no names at all.
        for carrier_id in set(carrier_by_index.values()) | set(transcript_tool_names):
            if carrier_id in ambiguous_tool_names:
                continue
            merged = tuple(transcript_tool_names.get(carrier_id, ()))
            existing_names = tool_names.get(carrier_id)
            if existing_names is not None and existing_names != merged:
                tool_names.pop(carrier_id)
                ambiguous_tool_names.add(carrier_id)
            else:
                tool_names[carrier_id] = merged
    mutations: list[_SessionMutation] = []
    detection_truncated: set[int] = set()
    mutation_state = state.get("chrys_mutations")
    turns = mutation_state.get("turns") if isinstance(mutation_state, dict) else None
    raw_snapshots = mutation_state.get("snapshots") if isinstance(mutation_state, dict) else None
    snapshots = raw_snapshots if isinstance(raw_snapshots, dict) else {}
    # Every structural record the walk skips is a detail row silently lost;
    # the survivors must not pass for the complete detail, or a damaged
    # document reads as an exact zero- or under-count.
    detail_complete = True
    if isinstance(turns, list):
        for turn in turns:
            _check_cancelled(cancel_event)
            if not isinstance(turn, dict):
                detail_complete = False
                continue
            turn_number = _plain_int_value(turn.get("turn_id"))
            if turn_number is None:
                detail_complete = False
                continue
            raw_detection_truncated = turn.get("detection_truncated")
            # The boolean leaves are omitted or bools, never anything else;
            # coercing a damaged value silently drops a truncation notice or
            # a contested marker, so they gate completeness like the strings.
            if raw_detection_truncated is not None and not isinstance(raw_detection_truncated, bool):
                detail_complete = False
            if raw_detection_truncated is True:
                detection_truncated.add(turn_number)
            raw_mutations = turn.get("mutations")
            if not isinstance(raw_mutations, list):
                detail_complete = False
                continue
            for mutation in raw_mutations:
                _check_cancelled(cancel_event)
                if not isinstance(mutation, dict) or not isinstance((mutation_path := mutation.get("path")), str):
                    detail_complete = False
                    continue
                old_path = mutation.get("old_path")
                before_hash = mutation.get("before_hash")
                after_hash = mutation.get("after_hash")
                before_skip = mutation.get("before_skip")
                after_skip = mutation.get("after_skip")
                provenance = mutation.get("provenance")
                # These fields are omitted or strings, never anything else;
                # coercing a damaged value to None silently reclassifies the
                # row (a lost before side turns a modify into a create), so
                # the survivors cannot pass for the complete detail either.
                for leaf in (old_path, before_hash, after_hash, before_skip, after_skip, provenance):
                    if leaf is not None and not isinstance(leaf, str):
                        detail_complete = False
                # A skip label this build cannot parse (a newer writer)
                # would fold as though the content were never withheld,
                # silently upgrading an unprovable net-zero to exact.
                for leaf in (before_skip, after_skip):
                    if isinstance(leaf, str) and parse_skip_reason(leaf) is None:
                        detail_complete = False
                # Content identity is a sha-256 hexdigest; a string of any
                # other shape (damage, or a newer writer's representation)
                # cannot carry the equality the fold reads off it.
                for leaf in (before_hash, after_hash):
                    if isinstance(leaf, str) and _CONTENT_HASH_RE.fullmatch(leaf) is None:
                        detail_complete = False
                raw_contested = mutation.get("contested")
                if raw_contested is not None and not isinstance(raw_contested, bool):
                    detail_complete = False
                old_before_hash: str | None = None
                old_before_skip: str | None = None
                if isinstance(old_path, str):
                    # Keys carry the path exactly as the row does; joining
                    # instead of re-normalizing keeps a session analyzable
                    # away from the machine that recorded it.
                    snapshot = snapshots.get(f"{old_path}::{turn_number}")
                    # A rename records its source snapshot in the same turn
                    # and removal drops the two together, so a missing or
                    # damaged entry is loss — reading it as "no pre-state"
                    # folds the source as a file that never existed.
                    if not isinstance(snapshot, dict):
                        detail_complete = False
                    else:
                        raw_old_hash = snapshot.get("content_hash")
                        raw_old_skip = snapshot.get("skip_reason")
                        raw_existed = snapshot.get("existed")
                        for leaf in (raw_old_hash, raw_old_skip):
                            if leaf is not None and not isinstance(leaf, str):
                                detail_complete = False
                        if isinstance(raw_old_skip, str) and parse_skip_reason(raw_old_skip) is None:
                            detail_complete = False
                        if isinstance(raw_old_hash, str) and _CONTENT_HASH_RE.fullmatch(raw_old_hash) is None:
                            detail_complete = False
                        # The writer keys each snapshot by its own path and
                        # turn and records existence as a boolean whose value
                        # fixes the hash/skip combination (existed: exactly
                        # one of them; gone: neither) — an entry that breaks
                        # any of that is damage, and its hash and skip cannot
                        # pass for the source's pre-state.
                        existence_consistent = (
                            (raw_old_hash is None) != (raw_old_skip is None)
                            if raw_existed is True
                            else raw_old_hash is None and raw_old_skip is None
                            if raw_existed is False
                            else False
                        )
                        if (
                            snapshot.get("path") != old_path
                            or _plain_int_value(snapshot.get("period_index", snapshot.get("turn_id"))) != turn_number
                            or not existence_consistent
                        ):
                            detail_complete = False
                        old_before_hash = raw_old_hash if isinstance(raw_old_hash, str) else None
                        old_before_skip = raw_old_skip if isinstance(raw_old_skip, str) else None
                mutations.append(
                    _SessionMutation(
                        turn_number=turn_number,
                        path=mutation_path,
                        old_path=old_path if isinstance(old_path, str) else None,
                        before_hash=before_hash if isinstance(before_hash, str) else None,
                        after_hash=after_hash if isinstance(after_hash, str) else None,
                        before_skip=before_skip if isinstance(before_skip, str) else None,
                        after_skip=after_skip if isinstance(after_skip, str) else None,
                        provenance=provenance if isinstance(provenance, str) else None,
                        contested=raw_contested is True,
                        old_before_hash=old_before_hash,
                        old_before_skip=old_before_skip,
                    )
                )
    return _SessionProjection(
        available=True,
        mutation_detail_available=isinstance(turns, list) and detail_complete,
        carriers=carriers,
        commands=commands,
        skill_names=skill_names,
        item_tokens=item_tokens,
        item_roles=item_roles,
        item_tool_names=tool_names,
        mutations=tuple(mutations),
        detection_truncated_turns=frozenset(detection_truncated),
    )

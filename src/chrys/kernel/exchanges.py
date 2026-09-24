# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Unified transcript exchange grammar.

One implementation of the exchange walk every transcript consumer shares,
over an accessor protocol so live ``Message`` objects and their ``to_dict``
serializations run the same code. The grammar:

- an EXCHANGE is a maximal run of consecutive assistant siblings containing
  at least one tool call (text-only and reasoning-only siblings ride the same
  run), followed by the tool output answering it — tool-role messages and
  result-only assistant messages (hosted MCP reports results assistant-role);
- a message carrying a history marker is a hard boundary in BOTH walks: never
  a response sibling, never part of the output run;
- a result-only assistant message belongs to the output block, never to the
  response run; a message with empty contents is NOT result-only;
- a call-carrying assistant after tool output starts a new exchange;
- a call-carrying assistant's own result-type contents are the exchange's
  EMBEDDED results: pairing follows the message-by-message rule — message
  N's calls register before message N's embedded results consume, so an
  embedded result may answer its own message's calls even when it physically
  precedes them, but never a call introduced by a later sibling;
- ``legacy``-typed contents count as calls in the dict accessor only: the
  type exists solely in serialized transcripts from older sessions.

Pairing is policy-driven: each consumer declares a :class:`PairingPolicy`
row (call/result type scope, informational gating, the None-id and
empty-id policies, and how malformed non-string ids normalize). The
accessor reports raw facts only — every pairing decision lives in
:func:`pair_results`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Protocol

from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.tool_result_metadata import TOOL_ERROR_KIND_METADATA_KEY

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

# ---------------------------------------------------------------------------
# Content-type sets (moved here from kernel.compaction so the iterator does
# not import the compaction module; compaction re-exports them unchanged)
# ---------------------------------------------------------------------------

# All content types that represent a tool invocation requested by the model.
# The Phase 4 last-words side call must reject responses carrying any of
# these in ANY message; note that compaction's ``_is_tool_call_assistant``
# helper restricts itself to assistant messages and must not be reused for
# that guard.
TOOL_CALL_CONTENT_TYPES = frozenset(
    {
        "function_call",
        "hosted_tool_call",
        "code_interpreter_tool_call",
        "image_generation_tool_call",
        "mcp_server_tool_call",
        "search_tool_call",
        "shell_tool_call",
    }
)

# Content types carrying the result half of a tool invocation; results can
# ride in assistant-role messages (hosted MCP), not only tool-role ones.
TOOL_RESULT_CONTENT_TYPES = frozenset(
    {
        "function_result",
        "hosted_tool_result",
        "code_interpreter_tool_result",
        "image_generation_tool_result",
        "mcp_server_tool_result",
        "search_tool_result",
        "shell_tool_result",
    }
)

# Serialized transcripts from older sessions carry ``legacy``-typed call
# contents; the type never appears on live kernel objects, so only the dict
# accessor treats it as a call.
LEGACY_CALL_CONTENT_TYPE = "legacy"

# The two content types that pair on ``image_id`` (their ``call_id`` is
# always None); every other call/result type pairs on ``call_id``.
_IMAGE_CONTENT_TYPES = frozenset({"image_generation_tool_call", "image_generation_tool_result"})

# Error kinds a tool result may carry when the kernel rejected the call
# before dispatch; the POSITIONAL_FAILURES policy consumes id-less failure
# results only when they carry one of these.
PREVALIDATION_ERROR_KINDS = frozenset({"argument_parsing", "argument_truncated", "tool_not_found"})

PairingKey = tuple[Literal["call_id", "image_id"], str]
"""Namespaced pairing id: image generation pairs on ``image_id`` in its own
namespace, so an equal string in the other namespace never matches."""

FalsyStream = Literal["none", "empty"]


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


class MessageAccessor(Protocol):
    """Raw-fact reads over one transcript representation.

    The accessor never derives pairing keys and never applies policy: it
    reports the verbatim identifier in the content's own namespace, and
    :func:`pair_results` normalizes it per the consumer's policy row.
    """

    def role(self, message: Any) -> str: ...

    def contents(self, message: Any) -> Sequence[Any]: ...

    def content_type(self, content: Any) -> str | None: ...

    def raw_id(self, content: Any) -> object: ...

    def is_informational(self, content: Any) -> bool: ...

    def error_kind(self, content: Any) -> str | None: ...

    def has_marker(self, message: Any) -> bool: ...

    def call_types(self) -> frozenset[str]: ...

    def result_types(self) -> frozenset[str]: ...


class LiveAccessor:
    """Accessor over live kernel ``Message``/``Content`` objects."""

    def role(self, message: Any) -> str:
        return message.role

    def contents(self, message: Any) -> Sequence[Any]:
        return message.contents

    def content_type(self, content: Any) -> str | None:
        return content.type

    def raw_id(self, content: Any) -> object:
        if content.type in _IMAGE_CONTENT_TYPES:
            return content.image_id
        return content.call_id

    def is_informational(self, content: Any) -> bool:
        return bool(content.informational_only)

    def error_kind(self, content: Any) -> str | None:
        kind = content.additional_properties.get(TOOL_ERROR_KIND_METADATA_KEY)
        return kind if isinstance(kind, str) else None

    def has_marker(self, message: Any) -> bool:
        # KEY presence, not value truthiness: the value is the marker kind,
        # and a falsy kind must still bound the exchange (fail-closed).
        return HistoryMarkerKind.KEY in message.additional_properties

    def call_types(self) -> frozenset[str]:
        return TOOL_CALL_CONTENT_TYPES

    def result_types(self) -> frozenset[str]:
        return TOOL_RESULT_CONTENT_TYPES


_DICT_CALL_CONTENT_TYPES = TOOL_CALL_CONTENT_TYPES | {LEGACY_CALL_CONTENT_TYPE}


class DictAccessor:
    """Accessor over ``Message.to_dict()`` transcript serializations."""

    def role(self, message: Mapping[str, Any]) -> str:
        role = message.get("role")
        return role if isinstance(role, str) else ""

    def contents(self, message: Mapping[str, Any]) -> Sequence[Any]:
        contents = message.get("contents")
        return contents if isinstance(contents, list) else []

    def content_type(self, content: Any) -> str | None:
        if not isinstance(content, Mapping):
            return None
        content_type = content.get("type")
        return content_type if isinstance(content_type, str) else None

    def raw_id(self, content: Any) -> object:
        if not isinstance(content, Mapping):
            return None
        if content.get("type") in _IMAGE_CONTENT_TYPES:
            return content.get("image_id")
        return content.get("call_id")

    def is_informational(self, content: Any) -> bool:
        if not isinstance(content, Mapping):
            return False
        # Mirror of the live constructor rule: mcp_server_tool_call is forced
        # informational at construction (whether or not the flag serialized).
        return bool(content.get("informational_only")) or content.get("type") == "mcp_server_tool_call"

    def error_kind(self, content: Any) -> str | None:
        if not isinstance(content, Mapping):
            return None
        properties = content.get("additional_properties")
        if not isinstance(properties, Mapping):
            return None
        kind = properties.get(TOOL_ERROR_KIND_METADATA_KEY)
        return kind if isinstance(kind, str) else None

    def has_marker(self, message: Mapping[str, Any]) -> bool:
        properties = message.get("additional_properties")
        return isinstance(properties, Mapping) and HistoryMarkerKind.KEY in properties

    def call_types(self) -> frozenset[str]:
        return _DICT_CALL_CONTENT_TYPES

    def result_types(self) -> frozenset[str]:
        return TOOL_RESULT_CONTENT_TYPES


# ---------------------------------------------------------------------------
# Grammar dataclasses and policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Occurrence:
    """One call or result CONTENT occurrence, addressed by coordinates.

    ``raw_id`` is the normalized tri-state (``None`` / ``""`` / non-empty
    string): a non-string verbatim value has already passed through the
    consuming policy's ``malformed_id`` rule. Consumers needing the original
    value (external correlation, persisted/emitted ids) dereference the
    content at these coordinates and apply their own transform — the
    normalized value drives pairing-derived decisions only.
    """

    message_index: int
    content_index: int
    raw_id: str | None
    content_type: str
    error_kind: str | None


@dataclass(frozen=True)
class Exchange:
    """One response run plus the tool output answering it.

    ``response_indices`` and ``output_indices`` never share a message, and
    embedded results (result contents riding the response messages
    themselves) are listed separately so nothing double-scans. Embedded
    occurrences here carry the policy-free structural view of the id
    (strings and ``None`` verbatim; non-string values stringified) —
    pairing consumers read :func:`pair_results` views, whose occurrences
    are normalized per their own policy row.
    """

    response_indices: tuple[int, ...]
    output_indices: tuple[int, ...]
    embedded_results: tuple[Occurrence, ...]


class NoneIdPolicy(Enum):
    """How a consumer treats ``None``-id call/result occurrences."""

    IGNORE = "ignore"
    UNPAIRABLE_OCCURRENCE = "unpairable_occurrence"
    POSITIONAL_FAILURES = "positional_failures"
    POSITIONAL = "positional"


class EmptyIdPolicy(Enum):
    """How a consumer treats ``""``-id call/result occurrences."""

    IGNORE = "ignore"
    COERCED_SINGLE_KEY = "coerced_single_key"
    POSITIONAL = "positional"
    UNPAIRABLE_OCCURRENCE = "unpairable_occurrence"


@dataclass(frozen=True)
class PairingPolicy:
    """One consumer's declared pairing row (the §3.1 policy table).

    ``malformed_id`` is required on every row: restored sessions build live
    contents from serialized dicts with no validation, so non-string ids
    reach live consumers too. ``stringify`` maps a truthy non-string to
    ``str(value)`` and a falsy non-string to ``""``; ``treat_as_none`` maps
    any non-string to ``None``. Strings always pass through verbatim.
    """

    call_types: frozenset[str]
    include_informational_calls: bool
    result_types: frozenset[str]
    none_id: NoneIdPolicy
    empty_id: EmptyIdPolicy
    malformed_id: Literal["stringify", "treat_as_none"]


@dataclass(frozen=True)
class EligibleResults:
    """Inventory minus forward orphans, keyed for post-pairing consumers.

    A truthy result is eligible when it registered at-or-after the first
    same-key call registration; a falsy result when ANY falsy call — either
    stream — registered before it (combined-falsy: the block-wide falsy
    boolean consumers derive from this view is stream-blind today).
    """

    truthy: dict[PairingKey, list[Occurrence]] = field(default_factory=dict)
    falsy: dict[FalsyStream, list[Occurrence]] = field(default_factory=dict)


@dataclass(frozen=True)
class ExchangePairing:
    """Pairing views for one exchange under one policy row.

    - ``truthy_assignments``: per key, call occurrences in registration
      order, each mapped to the result occurrence that consumed it or
      ``None`` — one result consumes one call occurrence (shared cursor);
    - ``unconsumed_results``: truthy results no call consumed (forward
      orphans included — a result registering before any same-key call is
      an orphan by the no-forward-pairing rule);
    - ``falsy_assignments`` / ``falsy_unconsumed_results``: the separated
      None/empty two-phase queues (occurrence order);
    - ``unpairable_calls`` / ``unpairable_results``: occurrences surfaced by
      an UNPAIRABLE_OCCURRENCE policy — visible, never paired;
    - ``eligible_results``: see :class:`EligibleResults`.
    """

    truthy_assignments: dict[PairingKey, list[tuple[Occurrence, Occurrence | None]]]
    unconsumed_results: dict[PairingKey, list[Occurrence]]
    falsy_assignments: dict[FalsyStream, list[tuple[Occurrence, Occurrence | None]]]
    falsy_unconsumed_results: dict[FalsyStream, list[Occurrence]]
    unpairable_calls: tuple[Occurrence, ...]
    unpairable_results: tuple[Occurrence, ...]
    eligible_results: EligibleResults

    @property
    def result_keys(self) -> set[PairingKey]:
        """Inventory: every truthy key with at least one result occurrence."""
        return set(self.results_by_key)

    @property
    def results_by_key(self) -> dict[PairingKey, list[Occurrence]]:
        """Inventory: every truthy result occurrence by key, orphans included."""
        inventory: dict[PairingKey, list[Occurrence]] = {}
        for key, assignments in self.truthy_assignments.items():
            for _call, result in assignments:
                if result is not None:
                    inventory.setdefault(key, []).append(result)
        for key, orphans in self.unconsumed_results.items():
            inventory.setdefault(key, []).extend(orphans)
        return inventory

    @property
    def answered_keys(self) -> set[PairingKey]:
        """Consumed-only view: keys holding at least one non-None assignment.

        Completeness consumers read this, never ``result_keys`` — a
        forward-reference embedded result is an orphan and must not make its
        call look answered.
        """
        return {
            key
            for key, assignments in self.truthy_assignments.items()
            if any(result is not None for _call, result in assignments)
        }


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


def is_result_only_message(message: Any, accessor: MessageAccessor) -> bool:
    """Role-gated result-only predicate: an assistant message carrying tool
    results for an earlier call and no new calls. Results ride tool-role or
    assistant-role messages only, so any other role is never result-only."""
    if accessor.role(message) != "assistant":
        return False
    call_types = accessor.call_types()
    result_types = accessor.result_types()
    saw_result = False
    for content in accessor.contents(message):
        content_type = accessor.content_type(content)
        if content_type in call_types:
            return False
        if content_type in result_types:
            saw_result = True
    return saw_result


def _carries_call(message: Any, accessor: MessageAccessor) -> bool:
    call_types = accessor.call_types()
    return any(accessor.content_type(content) in call_types for content in accessor.contents(message))


def _structural_raw_id(verbatim: object) -> str | None:
    """Policy-free id view for :class:`Exchange` embedded occurrences."""
    if verbatim is None or isinstance(verbatim, str):
        return verbatim
    return str(verbatim) if verbatim else ""


# ---------------------------------------------------------------------------
# Iterator
# ---------------------------------------------------------------------------


def iter_exchanges(messages: Sequence[Any], accessor: MessageAccessor) -> Iterator[Exchange]:
    """Yield every exchange of *messages* in transcript order.

    Only call-bearing response runs form exchanges; assistant runs without
    a single call content, orphan output, markers, and other-role messages
    are skipped (they belong to no exchange).
    """
    result_types = accessor.result_types()
    total = len(messages)
    index = 0
    while index < total:
        message = messages[index]
        if (
            accessor.has_marker(message)
            or accessor.role(message) != "assistant"
            or is_result_only_message(message, accessor)
        ):
            index += 1
            continue

        run_start = index
        while (
            index < total
            and accessor.role(messages[index]) == "assistant"
            and not accessor.has_marker(messages[index])
            and not is_result_only_message(messages[index], accessor)
        ):
            index += 1
        response_indices = tuple(range(run_start, index))

        if not any(_carries_call(messages[i], accessor) for i in response_indices):
            continue

        output_start = index
        while index < total and not accessor.has_marker(messages[index]):
            candidate = messages[index]
            if accessor.role(candidate) == "tool" or is_result_only_message(candidate, accessor):
                index += 1
                continue
            break
        output_indices = tuple(range(output_start, index))

        embedded: list[Occurrence] = []
        for message_index in response_indices:
            for content_index, content in enumerate(accessor.contents(messages[message_index])):
                content_type = accessor.content_type(content)
                if content_type in result_types:
                    embedded.append(
                        Occurrence(
                            message_index=message_index,
                            content_index=content_index,
                            raw_id=_structural_raw_id(accessor.raw_id(content)),
                            content_type=content_type,
                            error_kind=accessor.error_kind(content),
                        )
                    )

        yield Exchange(
            response_indices=response_indices,
            output_indices=output_indices,
            embedded_results=tuple(embedded),
        )


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def _normalize_raw_id(verbatim: object, policy: PairingPolicy) -> str | None:
    if verbatim is None or isinstance(verbatim, str):
        return verbatim
    if policy.malformed_id == "stringify":
        return str(verbatim) if verbatim else ""
    return None


def _namespace(content_type: str) -> Literal["call_id", "image_id"]:
    return "image_id" if content_type in _IMAGE_CONTENT_TYPES else "call_id"


def namespaced_pairing_key(content_type: str, raw_id: object) -> PairingKey | None:
    """Return the canonical truthy pairing key for one call/result occurrence."""
    if not isinstance(raw_id, str) or not raw_id:
        return None
    return (_namespace(content_type), raw_id)


class _PairingState:
    """Mutable two-phase pairing walk state for one exchange."""

    def __init__(self, policy: PairingPolicy) -> None:
        self.policy = policy
        self.sequence = 0
        self.truthy_assignments: dict[PairingKey, list[list[Occurrence | None]]] = {}
        self.unconsumed_results: dict[PairingKey, list[Occurrence]] = {}
        self.falsy_assignments: dict[FalsyStream, list[list[Occurrence | None]]] = {"none": [], "empty": []}
        self.falsy_unconsumed: dict[FalsyStream, list[Occurrence]] = {"none": [], "empty": []}
        self.unpairable_calls: list[Occurrence] = []
        self.unpairable_results: list[Occurrence] = []
        self.first_call_seq: dict[PairingKey, int] = {}
        self.first_falsy_call_seq: int | None = None
        self.eligible_truthy: dict[PairingKey, list[Occurrence]] = {}
        self.eligible_falsy: dict[FalsyStream, list[Occurrence]] = {"none": [], "empty": []}

    def _next_seq(self) -> int:
        self.sequence += 1
        return self.sequence

    def _falsy_stream(self, raw_id: str | None) -> FalsyStream:
        return "none" if raw_id is None else "empty"

    def _falsy_call_policy(self, raw_id: str | None) -> NoneIdPolicy | EmptyIdPolicy:
        return self.policy.none_id if raw_id is None else self.policy.empty_id

    def register_call(self, occurrence: Occurrence) -> None:
        seq = self._next_seq()
        raw_id = occurrence.raw_id
        if raw_id:
            key = namespaced_pairing_key(occurrence.content_type, raw_id)
            if key is None:
                raise RuntimeError("An occurrence with a non-empty id must have a pairing key.")
            self.first_call_seq.setdefault(key, seq)
            self.truthy_assignments.setdefault(key, []).append([occurrence, None])
            return

        policy = self._falsy_call_policy(raw_id)
        if policy in (NoneIdPolicy.IGNORE, EmptyIdPolicy.IGNORE):
            return
        if self.first_falsy_call_seq is None:
            self.first_falsy_call_seq = seq
        if policy in (NoneIdPolicy.UNPAIRABLE_OCCURRENCE, EmptyIdPolicy.UNPAIRABLE_OCCURRENCE):
            self.unpairable_calls.append(occurrence)
            return
        if policy is EmptyIdPolicy.COERCED_SINGLE_KEY:
            # Legacy conflated queue: empty-id occurrences ride the None
            # stream's queue, mirroring the single coerced key they shared.
            self.falsy_assignments["none"].append([occurrence, None])
            return
        self.falsy_assignments[self._falsy_stream(raw_id)].append([occurrence, None])

    def consume_result(self, occurrence: Occurrence) -> None:
        seq = self._next_seq()
        raw_id = occurrence.raw_id
        if raw_id:
            key = namespaced_pairing_key(occurrence.content_type, raw_id)
            if key is None:
                raise RuntimeError("An occurrence with a non-empty id must have a pairing key.")
            first_seen = self.first_call_seq.get(key)
            if first_seen is not None and first_seen <= seq:
                self.eligible_truthy.setdefault(key, []).append(occurrence)
            for slot in self.truthy_assignments.get(key, []):
                if slot[1] is None:
                    slot[1] = occurrence
                    return
            self.unconsumed_results.setdefault(key, []).append(occurrence)
            return

        policy = self._falsy_call_policy(raw_id)
        if policy in (NoneIdPolicy.IGNORE, EmptyIdPolicy.IGNORE):
            return
        stream = self._falsy_stream(raw_id)
        if self.first_falsy_call_seq is not None and self.first_falsy_call_seq <= seq:
            self.eligible_falsy[stream].append(occurrence)
        if policy in (NoneIdPolicy.UNPAIRABLE_OCCURRENCE, EmptyIdPolicy.UNPAIRABLE_OCCURRENCE):
            self.unpairable_results.append(occurrence)
            return
        if policy is NoneIdPolicy.POSITIONAL_FAILURES and occurrence.error_kind not in PREVALIDATION_ERROR_KINDS:
            # An arbitrary id-less result must not eat the slot ahead of a
            # later qualifying failure; it stays surfaced, never consuming.
            self.falsy_unconsumed[stream].append(occurrence)
            return
        queue_stream: FalsyStream = "none" if policy is EmptyIdPolicy.COERCED_SINGLE_KEY else stream
        for slot in self.falsy_assignments[queue_stream]:
            if slot[1] is None:
                slot[1] = occurrence
                return
        self.falsy_unconsumed[stream].append(occurrence)

    def finish(self) -> ExchangePairing:
        return ExchangePairing(
            truthy_assignments={
                # Registration always puts the non-optional call in slot 0.
                key: [(slot[0], slot[1]) for slot in slots if slot[0] is not None]
                for key, slots in self.truthy_assignments.items()
            },
            unconsumed_results=self.unconsumed_results,
            falsy_assignments={
                # Registration always puts the non-optional call in slot 0.
                stream: [(slot[0], slot[1]) for slot in slots if slot[0] is not None]
                for stream, slots in self.falsy_assignments.items()
            },
            falsy_unconsumed_results=self.falsy_unconsumed,
            unpairable_calls=tuple(self.unpairable_calls),
            unpairable_results=tuple(self.unpairable_results),
            eligible_results=EligibleResults(truthy=self.eligible_truthy, falsy=self.eligible_falsy),
        )


def pair_results(
    messages: Sequence[Any],
    exchange: Exchange,
    accessor: MessageAccessor,
    policy: PairingPolicy,
) -> ExchangePairing:
    """Pair *exchange*'s results with its calls under one policy row.

    The walk is message-phased: each response sibling registers its call
    occurrences before its embedded results consume (so a within-message
    reversed pair still pairs, and nothing forward-pairs with a later
    sibling's call); external output consumes against the full accumulated
    exchange in transcript order.
    """
    state = _PairingState(policy)

    def occurrence_of(message_index: int, content_index: int, content: Any, content_type: str) -> Occurrence:
        return Occurrence(
            message_index=message_index,
            content_index=content_index,
            raw_id=_normalize_raw_id(accessor.raw_id(content), policy),
            content_type=content_type,
            error_kind=accessor.error_kind(content),
        )

    for message_index in exchange.response_indices:
        contents = accessor.contents(messages[message_index])
        for content_index, content in enumerate(contents):
            content_type = accessor.content_type(content)
            if content_type not in policy.call_types:
                continue
            if not policy.include_informational_calls and accessor.is_informational(content):
                continue
            state.register_call(occurrence_of(message_index, content_index, content, content_type))
        for content_index, content in enumerate(contents):
            content_type = accessor.content_type(content)
            if content_type in policy.result_types:
                state.consume_result(occurrence_of(message_index, content_index, content, content_type))

    for message_index in exchange.output_indices:
        for content_index, content in enumerate(accessor.contents(messages[message_index])):
            content_type = accessor.content_type(content)
            if content_type in policy.result_types:
                state.consume_result(occurrence_of(message_index, content_index, content, content_type))

    return state.finish()


__all__ = [
    "LEGACY_CALL_CONTENT_TYPE",
    "PREVALIDATION_ERROR_KINDS",
    "TOOL_CALL_CONTENT_TYPES",
    "TOOL_RESULT_CONTENT_TYPES",
    "DictAccessor",
    "EligibleResults",
    "EmptyIdPolicy",
    "Exchange",
    "ExchangePairing",
    "FalsyStream",
    "LiveAccessor",
    "MessageAccessor",
    "NoneIdPolicy",
    "Occurrence",
    "PairingKey",
    "PairingPolicy",
    "is_result_only_message",
    "iter_exchanges",
    "namespaced_pairing_key",
    "pair_results",
]

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Turn resolution over live wire messages and provider state."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.models.turns import opens_turn, turn_slices
from chrys.kernel import EXCLUDED_KEY, Message
from chrys.service.context.providers.history import TURN_INDEX_KEY

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Turn resolution (§3.2 of the turn-boundary unification plan)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnSpan:
    """One resolved turn over the wire list.  Snapshot-scoped indices."""

    start: int
    """Wire index of the span's first message."""

    end: int
    """Wire index one past the span's last message."""

    absolute_number: int
    """State-backed absolute turn number (event cosmetics only)."""

    state_boundary: int | None
    """Provider-state index of the span's start-boundary message.

    ``None`` when the boundary message is absent from state — for the
    current span that is the Phase 3 clamp's protect-nothing signal: the
    input is stored only after the run, and inputs are stored before
    outputs, so state then holds no current-turn work at all.
    """


@dataclass(frozen=True)
class ResolvedTurns:
    """Snapshot-scoped turn resolution over the live wire + provider state.

    Valid only until the next wire/state mutation: Phase 1 inserts summary
    messages into the wire list and every Phase 3 fold replaces
    ``state["messages"]`` outright, shifting all later indices.  Re-resolve
    after a mutation; never cache a result across one.
    """

    spans: list[TurnSpan]
    """Real turns, oldest first; the last span is the current turn."""

    degraded: bool
    """True when the current marker region holds no real opener and the
    current span was synthesized from the region (§3.2 rule 4)."""

    synthetic_opener_idx: int | None
    """Wire index of the first user message inside a degraded region (the
    prompt-visible "opener" is synthetic); ``None`` for an assistant/tool-
    only degraded region, and always ``None`` in strict mode."""

    region_start: int
    """Wire index where the current marker region starts."""

    @property
    def current(self) -> TurnSpan | None:
        """The current turn's span (the last one), or ``None``."""
        return self.spans[-1] if self.spans else None

    @property
    def previous(self) -> list[TurnSpan]:
        """All completed turns' spans, oldest first."""
        return self.spans[:-1]


def _state_correspondence(messages: list[Message], state_messages: list[Message]) -> dict[int, int]:
    """Map wire index → provider-state index.

    Membership is by identity first, in two tiers.  Wrapper identity hits
    when the wire list holds the state objects themselves (legacy callers,
    state-side projections).  The kernel wire, however, hands every layer
    per-call message VIEWS — fresh wrapper + fresh contents list — so the
    second tier matches on the tuple of shared content-object ids, which is
    exact to the same degree: content objects live in exactly one message
    (the transcript's no-duplicate invariant), and a view carries the state
    original's objects.  Empty-contents messages carry no identity evidence
    and skip the tier.  The third tier matches wrappers REBUILT with fresh
    contents but the original's ``additional_properties`` dict — the
    sanctioned write-through channel, kept by the reminder middleware's
    enriched user messages and the LAST_WORDS refresh.  Without it a
    stored opener that is also the last user message (a child continuing
    from its own history) drops out of the history segment and the turn
    resolves to nothing.  The dict must belong to exactly one state message
    and the roles must agree.  The ``message_id`` fallback serves only wire
    messages with no identity hit anywhere in state (streaming finalizers
    rebuild current-turn messages into fresh objects, contents included)
    and is guarded: built EXCLUDING excluded and marker state messages and
    requiring role agreement — a bare id match is not safe because
    response aggregation reuses sequential ids across runs and
    ``_dedup_message_ids`` runs only over the wire list, so a pre-marker
    EXCLUDED state message can hold the same id as a visible current-turn
    message.  Later state copies win: rebuilt wire objects twin with the
    newest state occurrence of their id.
    """
    by_identity = {id(m): i for i, m in enumerate(state_messages)}
    by_content_identity: dict[tuple[int, ...], int] = {}
    by_properties_identity: dict[int, int] = {}
    shared_properties: set[int] = set()
    for s, m in enumerate(state_messages):
        if m.contents:
            by_content_identity[tuple(id(c) for c in m.contents)] = s
        properties_key = id(m.additional_properties)
        if properties_key in by_properties_identity:
            shared_properties.add(properties_key)
        by_properties_identity[properties_key] = s
    for properties_key in shared_properties:
        del by_properties_identity[properties_key]
    mapping: dict[int, int] = {}
    unmatched: list[int] = []
    for w, msg in enumerate(messages):
        s = by_identity.get(id(msg))
        if s is None and msg.contents:
            s = by_content_identity.get(tuple(id(c) for c in msg.contents))
        if s is None:
            s = by_properties_identity.get(id(msg.additional_properties))
            if s is not None and state_messages[s].role != msg.role:
                s = None
        if s is None:
            unmatched.append(w)
        else:
            mapping[w] = s
    if not unmatched:
        return mapping

    fallback: dict[tuple[str, str], int] = {}
    for s, msg in enumerate(state_messages):
        mid = msg.message_id
        if not mid:
            continue
        props = msg.additional_properties
        if props.get(EXCLUDED_KEY, False) or HistoryMarkerKind.KEY in props:
            continue
        fallback[(mid, msg.role)] = s
    engaged = False
    for w in unmatched:
        mid = messages[w].message_id
        if not mid:
            continue
        s = fallback.get((mid, messages[w].role))
        if s is not None:
            mapping[w] = s
            engaged = True
    if engaged:
        _log.debug("turn resolver: message_id fallback engaged for wire-state correspondence")
    return mapping


def _last_healthy_turn_marker(state_messages: list[Message]) -> tuple[int | None, int]:
    """Return ``(state index, _turn int)`` of the last HEALTHY turn marker.

    A marker cluster (the contiguous run of ``_chrys_kind`` messages ending
    at a TURN marker) containing ANY ``HistoryMarkerKind.STATUS_MARKERS``
    member does not close a completed turn: a crash-terminated turn is an
    interrupted CURRENT task, not a completed exchange.  Walk back to the
    newest turn marker whose cluster carries no status marker.  Returns
    ``(None, 0)`` when no healthy marker exists; the int is ``0`` when the
    healthy marker carries no usable ``_turn``.
    """
    i = len(state_messages) - 1
    while i >= 0:
        props = state_messages[i].additional_properties
        if props.get(HistoryMarkerKind.KEY) != HistoryMarkerKind.TURN:
            i -= 1
            continue
        healthy = True
        j = i
        while j >= 0:
            kind = state_messages[j].additional_properties.get(HistoryMarkerKind.KEY)
            if not kind:
                break
            if kind in HistoryMarkerKind.STATUS_MARKERS:
                healthy = False
            j -= 1
        if healthy:
            turn_int = props.get(TURN_INDEX_KEY)
            return i, turn_int if isinstance(turn_int, int) else 0
        i = j
    return None, 0


def _folded_turn_seed(state: dict[str, Any] | None) -> int:
    """Return the highest turn number preserved only in compressed blocks.

    Phase 3 / force compression folds up to a marker INCLUSIVE, so folded
    turns' markers no longer live in active state; their numbers survive in
    ``compressed_msgs[*].turn_range``.  Skips the ``(0, 0)`` no-marker
    sentinel.
    """
    if state is None:
        return 0
    blocks = state.get("compressed_msgs")
    if not isinstance(blocks, list):
        return 0
    seed = 0
    for block in blocks:
        turn_range = getattr(block, "turn_range", None)
        if turn_range is None and isinstance(block, dict):
            turn_range = block.get("turn_range")
        if not isinstance(turn_range, tuple | list) or len(turn_range) != 2:
            continue
        low, high = turn_range
        if (low, high) == (0, 0) or not isinstance(high, int):
            continue
        seed = max(seed, high)
    return seed


def _resolve_turns(messages: list[Message], state: dict[str, Any] | None) -> ResolvedTurns:
    """Resolve real turns over the live wire list + provider state (§3.2).

    Strict when the current marker region holds a real opener; degraded
    (current span = the whole region) when the region is non-empty with no
    opener but holds a user message or follows completed strict turns; the
    empty result only for histories that never held user input.  The marker
    boundary comes from the provider state (where markers live) and is
    PROJECTED into the wire list via the identity-first correspondence —
    compaction receives the marker-less wire view, so a wire-side marker
    scan can never see one.  EXCLUDED wire messages are invisible to the
    scan: a mid-call fold (Step 0 / P3) excludes a completed turn in place,
    and its opener must not keep opening a turn the state no longer holds.
    """
    state_messages: list[Message] = []
    if state is not None:
        raw = state.get("messages")
        if isinstance(raw, list):
            state_messages = raw

    correspondence = _state_correspondence(messages, state_messages) if state_messages else {}

    # Rule 1a — history segment.  Spans are confined to it in both strict
    # and degraded modes: non-history providers can emit user-role messages
    # into the request prefix, and unclipped slicing would let such a prefix
    # message open a "turn", handing provider context to P1/P2 and
    # mis-anchoring P4's prior-turn boundary.
    if correspondence:
        history_start = min(correspondence)
    elif state_messages:
        # State exists but none of it is on the wire (service-side context
        # skips local history): the only session content is the in-flight
        # input tail — anchor at its opener (or first user message for a
        # flagged-only tail) so a provider prefix can never open a turn.
        last_opener = None
        first_user = None
        for i, msg in enumerate(messages):
            if opens_turn(msg):
                last_opener = i
            if first_user is None and msg.role == "user":
                first_user = i
        if last_opener is not None:
            history_start = last_opener
        elif first_user is not None:
            history_start = first_user
        else:
            history_start = len(messages)
    else:
        # No provider state (fresh session or unbound strategy): the whole
        # wire is the segment, matching pre-resolver behavior.
        history_start = 0

    # Rule 1b — marker region, from the state-side scan projected into the
    # wire: just after the last wire message belonging to the completed-
    # turns set (state messages at/before the last HEALTHY turn marker).
    marker_idx, healthy_turn_int = _last_healthy_turn_marker(state_messages)
    if marker_idx is None:
        region_start = history_start
    else:
        completed_wire = [w for w, s in correspondence.items() if s <= marker_idx]
        region_start = max(completed_wire) + 1 if completed_wire else history_start

    # Slice over VISIBLE messages only: Step-0 compressions and P3 folds
    # exclude wire messages in place mid-call, and a folded opener must not
    # reopen a turn that no longer exists in provider state (the provider
    # filters excluded messages from the wire before the next request, so
    # mid-call exclusion is the only way one reaches the resolver).  Spans
    # keep raw wire indices — each span runs to the next span's start, so
    # interleaved excluded messages stay inside a span and the phases'
    # exclusion-aware group collection sees contiguous ranges.
    segment = messages[history_start:]
    visible_idx = [i for i, m in enumerate(segment) if not m.additional_properties.get(EXCLUDED_KEY, False)]
    visible_slices = turn_slices([segment[i] for i in visible_idx])
    slices: list[tuple[int, int]] = []
    for k, (s, _e) in enumerate(visible_slices):
        start = visible_idx[s] + history_start
        end = visible_idx[visible_slices[k + 1][0]] + history_start if k + 1 < len(visible_slices) else len(messages)
        slices.append((start, end))

    region = messages[region_start:]
    region_visible = [m for m in region if not m.additional_properties.get(EXCLUDED_KEY, False)]
    degraded = False
    synthetic_opener_idx: int | None = None
    if not region_visible or any(opens_turn(m) for m in region_visible):
        # Rules 2 + 3 — strict: current turn = the last slice.
        final_slices = slices
    elif any(m.role == "user" for m in region_visible) or slices:
        # Rule 4 — degraded: the current span is the whole region (its
        # assistant/tool work precedes any synthetic opener and P4 only
        # collects groups from the span start); previous turns are the
        # strict slices clipped at the region boundary (only the last one
        # can straddle it, and clipping is correct precisely because the
        # boundary cluster is healthy).
        degraded = True
        final_slices = [(s, min(e, region_start)) for s, e in slices if s < region_start]
        final_slices.append((region_start, len(messages)))
        synthetic_opener_idx = next(
            (
                region_start + i
                for i, m in enumerate(region)
                if m.role == "user" and not m.additional_properties.get(EXCLUDED_KEY, False)
            ),
            None,
        )
        _log.debug(
            "turn resolver: degraded region at wire index %d (synthetic opener: %s)",
            region_start,
            synthetic_opener_idx,
        )
    else:
        # Rule 5 — no user message anywhere: today's no-op early return.
        return ResolvedTurns(spans=[], degraded=False, synthetic_opener_idx=None, region_start=region_start)

    # Absolute numbering.  Completed spans read the first state turn marker
    # after their opener; the current span is max(last healthy active
    # marker, folded seed) + 1 — NOT turn_counter + 1, which counts an
    # unstripped crash cluster's aborted turn while the resolver refuses to
    # close it.  The resolver wins on divergence.  The previous-span count
    # keeps enumeration coherent when state carries no numbering at all
    # (unbound strategy) or a healthy marker lost its ``_turn`` int.
    seed = _folded_turn_seed(state)
    current_number = max(healthy_turn_int, seed, len(final_slices) - 1) + 1
    state_turn_markers = [
        (i, m.additional_properties.get(TURN_INDEX_KEY))
        for i, m in enumerate(state_messages)
        if m.additional_properties.get(HistoryMarkerKind.KEY) == HistoryMarkerKind.TURN
    ]

    spans: list[TurnSpan] = []
    last = len(final_slices) - 1
    for k, (start, end) in enumerate(final_slices):
        state_boundary = correspondence.get(start)
        if k == last:
            number = current_number
        else:
            number = None
            if state_boundary is not None:
                for m_idx, turn_int in state_turn_markers:
                    if m_idx > state_boundary and isinstance(turn_int, int):
                        number = turn_int
                        break
            if number is None:
                number = seed + k + 1
                _log.debug("turn resolver: enumerated fallback number %d for span %d", number, k)
        spans.append(TurnSpan(start=start, end=end, absolute_number=number, state_boundary=state_boundary))

    return ResolvedTurns(
        spans=spans,
        degraded=degraded,
        synthetic_opener_idx=synthetic_opener_idx,
        region_start=region_start,
    )

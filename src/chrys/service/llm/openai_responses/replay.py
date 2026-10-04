# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Replaying reasoning from local history.

Under local storage the request carries reasoning items the client rebuilds
from history, and the API accepts one only next to the calls it led to. So
history is cut into positional groups (:func:`partition_groups`): a call
message with the reasoning around it and the outputs that answer it, or a
message standing alone. :func:`plan_groups` rebuilds each group's reasoning
items and degrades a group that cannot replay them safely, encrypted
reasoning another endpoint issued included: it then goes out without
reasoning or hosted MCP calls, its function calls without item ids.
:func:`encode_input` encodes history with those plans.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import groupby
from typing import TYPE_CHECKING, Any

from chrys.foundation.reasoning_origin import replays_to
from chrys.kernel.compaction import (
    GROUP_ANNOTATION_KEY,
    GROUP_HAS_REASONING_KEY,
    GROUP_KIND_KEY,
    group_annotation_signature,
)
from chrys.kernel.exchanges import (
    TOOL_CALL_CONTENT_TYPES,
    TOOL_RESULT_CONTENT_TYPES,
    EmptyIdPolicy,
    LiveAccessor,
    NoneIdPolicy,
    PairingPolicy,
    iter_exchanges,
    namespaced_pairing_key,
    pair_results,
)
from chrys.service.llm.chat_completions.reasoning import (
    REASONING_CONTENT_FIELD,
    REASONING_DETAILS_FIELD,
    REASONING_FIELD,
    REASONING_FORMAT_KEY,
)

from .history import (
    OPENAI_SHELL_OUTPUT_TYPE_KEY,
    OPENAI_SHELL_OUTPUT_TYPE_LOCAL_SHELL_CALL,
    OPENAI_SHELL_OUTPUT_TYPE_SHELL_CALL,
    answer_apply_patch_calls,
    encode_content,
    encode_message,
    hosted_degradations,
    marked_reasoning_text,
)
from .hosted import coalesce_pending_results

if TYPE_CHECKING:
    from chrys.foundation.reasoning_origin import ReasoningOrigin
    from chrys.kernel import Content, Message
    from chrys.kernel.exchanges import PairingKey

    from .client import ResponsesVariant

logger = logging.getLogger(__name__)

_TOOL_TYPES = TOOL_CALL_CONTENT_TYPES | TOOL_RESULT_CONTENT_TYPES
_SHELL_OUTPUT_MARKERS = frozenset({OPENAI_SHELL_OUTPUT_TYPE_SHELL_CALL, OPENAI_SHELL_OUTPUT_TYPE_LOCAL_SHELL_CALL})
# Reasoning captured from Chat Completions; it never replays as an item.
_CHAT_COMPLETIONS_FORMATS = (REASONING_CONTENT_FIELD, REASONING_DETAILS_FIELD, REASONING_FIELD)
_LIVE = LiveAccessor()
_PAIRING = PairingPolicy(
    call_types=TOOL_CALL_CONTENT_TYPES,
    include_informational_calls=True,
    result_types=TOOL_RESULT_CONTENT_TYPES,
    none_id=NoneIdPolicy.POSITIONAL,
    empty_id=EmptyIdPolicy.POSITIONAL,
    malformed_id="stringify",
)

type _Signature = tuple[str, str, int, bool]


@dataclass(slots=True, frozen=True)
class ReplayGroup:
    """Messages that replay, or degrade, together."""

    position: int
    messages: tuple[Message, ...]
    annotated_reasoning_tool_call: bool
    # The exchange the group belongs to; None when it has none or several.
    exchange_ordinal: int | None


@dataclass(slots=True)
class ReplayPlan:
    """How one group is encoded; contents are named by identity."""

    group: ReplayGroup
    # Rebuilt items keyed by the first content of their occurrence.
    reasoning_items: dict[int, dict[str, Any]] = field(default_factory=dict)
    dropped: set[int] = field(default_factory=set)
    calls_without_id: set[int] = field(default_factory=set)
    reasoning_ids: list[str] = field(default_factory=list)
    call_ids: list[str] = field(default_factory=list)
    # Top-level item ids the group sends; ids must not repeat in a request.
    wire_ids: list[str] = field(default_factory=list)
    mcp_wire_ids: list[str] = field(default_factory=list)
    mcp_result_ids: list[str] = field(default_factory=list)
    degraded: bool = False
    # Degraded only because another endpoint issued some of the reasoning:
    # expected after a switch, not a fault.
    foreign_reasoning: bool = False

    def degrade(self) -> None:
        """Switch the group to the form that always replays."""
        self.degraded = True
        self.reasoning_items.clear()
        for message in self.group.messages:
            for content in message.contents:
                key = id(content)
                if content.type in {"text_reasoning", "mcp_server_tool_call"}:
                    self.dropped.add(key)
                elif content.type == "function_call" and key not in self.dropped:
                    self.calls_without_id.add(key)


def encode_input(
    messages: Sequence[Message],
    *,
    service_side: bool,
    variant: ResponsesVariant,
    origin: ReasoningOrigin | None = None,
) -> list[dict[str, Any]]:
    """Encode history as input items for the endpoint *origin*.

    Pending hosted results join their calls within one exchange only: call
    ids may repeat in a later one.
    """
    groups = partition_groups(messages)
    provider = variant.hosted_provider
    degradations = hosted_degradations(messages, provider=provider, service_side=service_side)
    plans = (
        [ReplayPlan(group) for group in groups] if service_side else plan_groups(groups, variant=variant, origin=origin)
    )
    items: list[dict[str, Any]] = []
    for _, batch in groupby(plans, key=_batch_key):
        batch_items = [
            item
            for plan in batch
            for message in plan.group.messages
            for item in encode_message(
                message,
                provider=provider,
                service_side=service_side,
                reasoning_items=plan.reasoning_items,
                dropped=plan.dropped,
                calls_without_id=plan.calls_without_id,
                degradations=degradations,
            )
        ]
        items.extend(answer_apply_patch_calls(coalesce_pending_results(batch_items)))
    return items


def _batch_key(plan: ReplayPlan) -> tuple[str, int]:
    ordinal = plan.group.exchange_ordinal
    return ("alone", plan.group.position) if ordinal is None else ("exchange", ordinal)


@dataclass(slots=True)
class _Run:
    """The group being cut: its annotation signature and its exchange."""

    signature: _Signature | None
    call_keys: set[PairingKey] = field(default_factory=set)
    owner: int | None = None

    def admits(self, signature: _Signature | None) -> bool:
        """Compaction annotations only split groups: a message whose
        signature conflicts with the group's stays out."""
        if self.signature is None:
            self.signature = signature
            return True
        return signature is None or signature == self.signature

    def admits_result(self, message: Message, signature: _Signature | None) -> bool:
        """An assistant message answering only this group's calls joins
        whatever its signature (the annotator gives it a span of its own);
        results for calls made elsewhere follow the signature rule, so a
        stray orphan cannot join and degrade a valid group."""
        if message.role == "assistant":
            keys = _pairing_keys(message, TOOL_RESULT_CONTENT_TYPES)
            if keys and keys <= self.call_keys:
                return True
        return self.admits(signature)


def partition_groups(messages: Sequence[Message]) -> list[ReplayGroup]:
    """Cut history into replay groups.

    The exchange grammar decides which outputs answer which calls: a group
    takes only outputs its own exchange owns, and a history marker always
    stands alone. Within an exchange every call message opens its own group.
    A group without a call message has no exchange; its orphan outputs join
    structurally and degrade with it. Annotations only split, so histories
    without them cut the same either way.
    """
    signatures = [group_annotation_signature(message) for message in messages]
    call_owner: dict[int, int] = {}
    output_owner: dict[int, int] = {}
    for ordinal, exchange in enumerate(iter_exchanges(messages, _LIVE)):
        call_owner.update(dict.fromkeys(exchange.response_indices, ordinal))
        output_owner.update(dict.fromkeys(exchange.output_indices, ordinal))
    total = len(messages)

    def skip_reasoning(index: int, run: _Run) -> int:
        while index < total and _is_reasoning_only(messages[index]) and run.admits(signatures[index]):
            index += 1
        return index

    def open_exchange(index: int, run: _Run) -> int:
        run.call_keys |= _pairing_keys(messages[index], TOOL_CALL_CONTENT_TYPES)
        run.owner = call_owner.get(index)
        return skip_reasoning(index + 1, run)

    def owned_output(index: int, owner: int | None) -> bool:
        if owner is not None:
            return output_owner.get(index) == owner
        return _is_output_message(messages[index]) and not _LIVE.has_marker(messages[index])

    groups: list[ReplayGroup] = []
    start = 0
    while start < total:
        first = messages[start]
        run = _Run(signatures[start])
        end = start + 1
        if _LIVE.has_marker(first):
            pass
        elif _is_call_message(first) or _is_reasoning_only(first):
            if _is_call_message(first):
                end = open_exchange(start, run)
            else:
                end = skip_reasoning(end, run)
                if end < total and _is_call_message(messages[end]) and run.admits(signatures[end]):
                    end = open_exchange(end, run)
            while end < total and owned_output(end, run.owner) and run.admits_result(messages[end], signatures[end]):
                end += 1
        elif _is_output_message(first):
            while (
                end < total
                and _is_output_message(messages[end])
                and not _LIVE.has_marker(messages[end])
                and run.admits_result(messages[end], signatures[end])
            ):
                end += 1
        members = tuple(messages[start:end])
        exchanges = {
            owner
            for index in range(start, end)
            if (owner := call_owner.get(index, output_owner.get(index))) is not None
        }
        groups.append(
            ReplayGroup(
                position=len(groups),
                messages=members,
                annotated_reasoning_tool_call=any(map(_annotated_reasoning_tool_call, members)),
                # A group spans at most one exchange; malformed history that
                # breaks this fails closed instead of coalescing across them.
                exchange_ordinal=next(iter(exchanges)) if len(exchanges) == 1 else None,
            )
        )
        start = end
    return groups


def _is_reasoning_only(message: Message) -> bool:
    return (
        message.role == "assistant"
        and bool(message.contents)
        and all(content.type == "text_reasoning" for content in message.contents)
    )


def _is_call_message(message: Message) -> bool:
    return message.role == "assistant" and any(content.type in TOOL_CALL_CONTENT_TYPES for content in message.contents)


def _is_output_message(message: Message) -> bool:
    """A message that only answers calls made before it."""
    if message.role == "tool":
        return True
    return (
        message.role == "assistant"
        and not _is_call_message(message)
        and any(content.type in TOOL_RESULT_CONTENT_TYPES for content in message.contents)
    )


def _annotated_reasoning_tool_call(message: Message) -> bool:
    annotation = message.additional_properties.get(GROUP_ANNOTATION_KEY)
    return (
        isinstance(annotation, Mapping)
        and annotation.get(GROUP_KIND_KEY) == "tool_call"
        and annotation.get(GROUP_HAS_REASONING_KEY) is True
    )


def _pairing_keys(message: Message, types: frozenset[str]) -> set[PairingKey]:
    return {
        key
        for content in message.contents
        if content.type in types and (key := namespaced_pairing_key(content.type, _LIVE.raw_id(content))) is not None
    }


def plan_groups(
    groups: Sequence[ReplayGroup], *, variant: ResponsesVariant, origin: ReasoningOrigin | None = None
) -> list[ReplayPlan]:
    """Plan every group of a request to the endpoint *origin* that replays local history."""
    result_owners = _result_owners(groups)
    plans = [_plan_group(group, result_owners, variant=variant, origin=origin) for group in groups]
    _degrade_reused_ids(plans)
    # A result whose call was degraded cannot replay without it. Results
    # without an owner stay until coalescing drops them: their position
    # still splits the message runs around them.
    degraded_positions = {plan.group.position for plan in plans if plan.degraded}
    for plan in plans:
        for message in plan.group.messages:
            for content in message.contents:
                if content.type == "mcp_server_tool_result" and result_owners.get(id(content)) in degraded_positions:
                    plan.dropped.add(id(content))
    degraded = [plan for plan in plans if plan.degraded]
    if faulty := [plan for plan in degraded if not plan.foreign_reasoning]:
        logger.warning(
            "Degraded stateless reasoning replay: groups=%s reasoning_ids=%s call_ids=%s",
            [plan.group.position for plan in faulty],
            list(dict.fromkeys(name for plan in faulty for name in plan.reasoning_ids)),
            list(dict.fromkeys(name for plan in faulty for name in plan.call_ids)),
        )
    if foreign := [plan for plan in degraded if plan.foreign_reasoning]:
        logger.debug("Left out reasoning another endpoint issued: groups=%s", [plan.group.position for plan in foreign])
    return plans


def _plan_group(
    group: ReplayGroup,
    result_owners: Mapping[int, int],
    *,
    variant: ResponsesVariant,
    origin: ReasoningOrigin | None,
) -> ReplayPlan:
    occurrences = _reasoning_occurrences(group)
    reasoning_items, replayable, only_foreign = _rebuilt_reasoning(
        occurrences, encrypted=variant.encrypted_reasoning, origin=origin
    )
    plan = ReplayPlan(group, reasoning_items)
    plan.reasoning_ids = [occurrence[0].id or "<missing>" for occurrence in occurrences]
    tools = [
        (message, content) for message in group.messages for content in message.contents if content.type in _TOOL_TYPES
    ]
    plan.call_ids = [
        name for _, content in tools if isinstance(name := content.call_id or content.image_id, str) and name
    ]

    invalid = bool(occurrences) and not replayable
    faulty = invalid and not only_foreign
    if group.annotated_reasoning_tool_call and tools and not occurrences:
        # Compaction saw reasoning here that history no longer has.
        faulty = True
        plan.reasoning_ids.append("<missing>")
    if group.annotated_reasoning_tool_call and occurrences and not tools:
        faulty = True

    def encode(message: Message, content: Content) -> dict[str, Any]:
        return encode_content(
            message.role,
            content,
            provider=variant.hosted_provider,
            replays_local_storage="_attribution" in message.additional_properties,
        )

    unsafe: set[int] = set()
    if occurrences or group.annotated_reasoning_tool_call:
        # Reasoning replays only with calls the API pairs with it again.
        for message, content in tools:
            shell_output = (
                content.type in {"function_call", "function_result"}
                and content.additional_properties.get(OPENAI_SHELL_OUTPUT_TYPE_KEY) in _SHELL_OUTPUT_MARKERS
            )
            informational = content.type == "function_call" and content.informational_only
            encoded = encode(message, content)
            if shell_output or informational or not encoded:
                unsafe.add(id(content))
            elif content.type == "function_call" and (wire_id := _item_id(encoded)):
                plan.wire_ids.append(wire_id)
        calls = {
            content_type: {
                content.call_id
                for _, content in tools
                if content.type == content_type and id(content) not in unsafe and content.call_id
            }
            for content_type in ("function_call", "mcp_server_tool_call")
        }
        for _, content in tools:
            if content.type not in {"function_result", "mcp_server_tool_result"}:
                continue
            owner = result_owners.get(id(content))
            if owner is not None and owner != group.position:
                # Carried by this group, but the exchange paired it with a
                # sibling group's call.
                continue
            call_type = "function_call" if content.type == "function_result" else "mcp_server_tool_call"
            if owner is None or not content.call_id or content.call_id not in calls[call_type]:
                unsafe.add(id(content))
    else:
        for message, content in tools:
            if content.type == "function_call" and content.call_id and (wire_id := _item_id(encode(message, content))):
                plan.wire_ids.append(wire_id)

    plan.wire_ids.extend(item["id"] for item in reasoning_items.values() if _item_id(item))
    plan.mcp_wire_ids = [
        content.call_id for _, content in tools if content.type == "mcp_server_tool_call" and content.call_id
    ]
    plan.mcp_result_ids = [
        content.call_id for _, content in tools if content.type == "mcp_server_tool_result" and content.call_id
    ]
    plan.wire_ids.extend(plan.mcp_wire_ids)
    if invalid or faulty or unsafe:
        plan.dropped.update(unsafe)
        plan.degrade()
        plan.foreign_reasoning = not (faulty or unsafe)
    return plan


def _item_id(item: Mapping[str, Any] | None) -> str | None:
    item_id = item.get("id") if item else None
    return item_id if isinstance(item_id, str) and item_id else None


def _degrade_reused_ids(plans: Sequence[ReplayPlan]) -> None:
    """Degrade a reasoning or MCP group whose item ids repeat.

    Groups with function calls only never degrade here: their ids are
    reserved against every other group wherever they sit. Repeated MCP ids
    whose every call is answered inside the group are separate calls that
    coalesce in order; any other repeat, inside the group or with an
    earlier one, degrades.
    """
    seen: set[str] = set()
    reserved: set[str] = set()
    candidates: list[ReplayPlan] = []
    for plan in plans:
        if plan.degraded:
            continue
        sends_mcp_calls = any(
            content.type == "mcp_server_tool_call" and content.call_id
            for message in plan.group.messages
            for content in message.contents
        )
        if plan.reasoning_items or sends_mcp_calls:
            candidates.append(plan)
        else:
            reserved.update(plan.wire_ids)
    for plan in candidates:
        mcp_calls = Counter(plan.mcp_wire_ids)
        mcp_results = Counter(plan.mcp_result_ids)
        repeats = any(
            count > 1 and not (mcp_calls[wire_id] == count and mcp_results[wire_id] == count)
            for wire_id, count in Counter(plan.wire_ids).items()
        )
        if repeats or any(wire_id in seen or wire_id in reserved for wire_id in plan.wire_ids):
            plan.degrade()
        else:
            seen.update(plan.wire_ids)


def _result_owners(groups: Sequence[ReplayGroup]) -> dict[int, int]:
    """The group whose call each paired result answers, by result identity.

    ``pair_results`` decides which occurrence answers which call, even with
    repeated ids, so a result carried by one group can belong to a sibling.
    A result object paired into two groups gets no owner.
    """
    messages = [message for group in groups for message in group.messages]
    positions = [group.position for group in groups for _ in group.messages]
    owners: dict[int, int] = {}
    ambiguous: set[int] = set()
    for exchange in iter_exchanges(messages, _LIVE):
        pairing = pair_results(messages, exchange, _LIVE, _PAIRING)
        for assignments in (*pairing.truthy_assignments.values(), *pairing.falsy_assignments.values()):
            for call, result in assignments:
                if result is None:
                    continue
                key = id(messages[result.message_index].contents[result.content_index])
                if key in ambiguous:
                    continue
                owner = positions[call.message_index]
                if owners.setdefault(key, owner) != owner:
                    del owners[key]
                    ambiguous.add(key)
    return owners


def _reasoning_occurrences(group: ReplayGroup) -> list[list[Content]]:
    """Runs of reasoning contents, one per provider reasoning item.

    A run ends at any other content or where the reasoning id changes; a
    content object repeated within a run counts once.
    """
    occurrences: list[list[Content]] = []
    current: list[Content] = []
    for message in group.messages:
        for content in message.contents:
            if content.type != "text_reasoning":
                if current:
                    occurrences.append(current)
                    current = []
                continue
            if current and current[-1].id != content.id:
                occurrences.append(current)
                current = []
            if not any(existing is content for existing in current):
                current.append(content)
    if current:
        occurrences.append(current)
    return occurrences


def _rebuilt_reasoning(
    occurrences: Sequence[Sequence[Content]], *, encrypted: bool, origin: ReasoningOrigin | None
) -> tuple[dict[int, dict[str, Any]], bool, bool]:
    """Rebuild each occurrence's reasoning item.

    Returns the items, whether every occurrence replays, and whether the
    ones that do not are all ones another endpoint issued (False when every
    one replays).

    OpenAI replays the encrypted payload; an occurrence without one, or with
    one an endpoint other than *origin* issued, cannot replay. Plaintext
    dialects replay the reasoning text; an occurrence captured from Chat
    Completions is left out without degrading the group.
    """
    if encrypted:
        foreign = {
            id(contents[0])
            for contents in occurrences
            if not all(replays_to(content.additional_properties, origin) for content in contents)
        }
        items = {
            id(contents[0]): item
            for contents in occurrences
            if id(contents[0]) not in foreign and (item := _encrypted_item(contents)) is not None
        }
        return items, len(items) == len(occurrences), bool(foreign) and len(items) + len(foreign) == len(occurrences)
    items = {id(contents[0]): item for contents in occurrences if (item := _plaintext_item(contents)) is not None}
    return items, all(map(_plaintext_replayable, occurrences)), False


def _encrypted_item(contents: Sequence[Content]) -> dict[str, Any] | None:
    reasoning_id = contents[0].id
    # Every streamed sibling carries the item's first payload; only the last
    # one carries the final payload, so the last one present wins.
    payload = next(
        (
            found
            for content in reversed(contents)
            if (found := content.protected_data or content.additional_properties.get("encrypted_content"))
        ),
        None,
    )
    if not reasoning_id or not payload:
        return None
    item: dict[str, Any] = {"type": "reasoning", "id": reasoning_id, "summary": [], "encrypted_content": payload}
    texts: list[dict[str, str]] = []
    for content in contents:
        properties = content.additional_properties
        if status := properties.get("status"):
            item["status"] = status
        if properties.get("reasoning_text"):
            if text := marked_reasoning_text(content):
                texts.append({"type": "reasoning_text", "text": text})
        elif content.text:
            item["summary"].append({"type": "summary_text", "text": content.text})
    if texts:
        item["content"] = texts
    return item


def _plaintext_item(contents: Sequence[Content]) -> dict[str, Any] | None:
    if not _plaintext_replayable(contents) or _chat_completions_formats(contents):
        return None
    status: Any = None
    texts: list[dict[str, str]] = []
    for content in contents:
        if (value := content.additional_properties.get("status")) is not None:
            status = value
        if text := marked_reasoning_text(content):
            texts.append({"type": "reasoning_text", "text": text})
    if not texts:
        return None
    item: dict[str, Any] = {"type": "reasoning", "content": texts}
    if contents[0].id:
        item["id"] = contents[0].id
    if status is not None:
        item["status"] = status
    return item


def _plaintext_replayable(contents: Sequence[Content]) -> bool:
    if formats := _chat_completions_formats(contents):
        return all(name in _CHAT_COMPLETIONS_FORMATS for name in formats)
    # An encrypted payload without its item id is broken history.
    payload = any(
        content.protected_data or content.additional_properties.get("encrypted_content") for content in contents
    )
    return not (payload and not contents[0].id)


def _chat_completions_formats(contents: Sequence[Content]) -> list[Any]:
    return [
        content.additional_properties[REASONING_FORMAT_KEY]
        for content in contents
        if REASONING_FORMAT_KEY in content.additional_properties
    ]

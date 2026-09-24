# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Physically scoped Phase-4 timeline and completer-slice construction."""

from __future__ import annotations

from collections import Counter
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Any

from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.kernel import (
    EXCLUDED_KEY,
    GROUP_ANNOTATION_KEY,
    SUMMARY_OF_GROUP_IDS_KEY,
    SUMMARY_OF_MESSAGE_IDS_KEY,
    TOOL_CALL_CONTENT_TYPES,
    Content,
    GroupKind,
    Message,
    TokenizerProtocol,
    estimate_message_tokens,
)
from chrys.kernel.compaction import GROUP_TOKEN_COUNT_KEY, _group_id, _group_kind
from chrys.service.agent_middleware.system_reminder import REMINDER_TAG_CLOSE, REMINDER_TAG_OPEN

DEGRADED_SCOPED_PREAMBLE = (
    "(This task was resumed without its original prompt. The messages that follow are the in-progress work "
    "of the current task.)"
)
SLICE_TRUNCATION_MARKER = (
    "[…truncated for summarization; archived details may be available after compaction—"
    "consult the dropped-record manifest before relying on them]"
)

_TOOL_RESULT_FLOOR_TOKENS = 2_000
_ASSISTANT_TEXT_CAP_TOKENS = 2_000

_RESULT_CONTENT_TYPES = frozenset(
    {
        "code_interpreter_tool_result",
        "function_result",
        "hosted_tool_result",
        "image_generation_tool_result",
        "mcp_server_tool_result",
        "search_tool_result",
        "shell_tool_result",
    }
)
_RESULT_TYPE_BY_CALL_TYPE = {
    "code_interpreter_tool_call": "code_interpreter_tool_result",
    "function_call": "function_result",
    "hosted_tool_call": "hosted_tool_result",
    "image_generation_tool_call": "image_generation_tool_result",
    "mcp_server_tool_call": "mcp_server_tool_result",
    "search_tool_call": "search_tool_result",
    "shell_tool_call": "shell_tool_result",
}


@dataclass(frozen=True, slots=True)
class ScopedGroup:
    """One annotated group in the current Phase-4 span."""

    group_id: str
    kind: GroupKind
    messages: tuple[Message, ...]
    integrity: bool


@dataclass(frozen=True, slots=True)
class ScopedTimeline:
    """Ordered scoped groups plus the effective degraded-opener state."""

    groups: tuple[ScopedGroup, ...]
    degraded_opener: bool


@dataclass(frozen=True, slots=True)
class PreparedScopedSlice:
    """A validated, budgeted completer slice made entirely from clones."""

    messages: tuple[Message, ...]
    estimated_tokens: int
    clipped_results: int
    clipped_assistant_texts: int


@dataclass(frozen=True, slots=True)
class _TextSlot:
    owner: Any
    key: str | int

    def get(self) -> str:
        value = self.owner.__dict__[self.key] if isinstance(self.owner, Content) else self.owner[self.key]
        return value if isinstance(value, str) else ""

    def set(self, value: str) -> None:
        if isinstance(self.owner, Content):
            self.owner.__dict__[self.key] = value
        else:
            self.owner[self.key] = value


def clone_for_slice(message: Message) -> Message:
    """Clone a message for sanitization or clipping without losing wire fields."""
    cloned = copy(message)
    cloned.contents = [deepcopy(content) for content in message.contents]
    cloned.additional_properties = deepcopy(message.additional_properties)
    annotation = cloned.additional_properties.get(GROUP_ANNOTATION_KEY)
    if isinstance(annotation, dict):
        annotation.pop(GROUP_TOKEN_COUNT_KEY, None)
    return cloned


def is_structural_group(group_messages: list[Message] | tuple[Message, ...]) -> bool:
    """Whether a group represents compressed/history structure, not live work."""
    for message in group_messages:
        properties = message.additional_properties
        if HistoryMarkerKind.KEY in properties:
            # Key presence, matching the grammar's boundary rule: a falsy
            # kind is still chrome and never live work.
            return True
        annotation = properties.get(GROUP_ANNOTATION_KEY)
        if isinstance(annotation, dict) and (
            SUMMARY_OF_MESSAGE_IDS_KEY in annotation or SUMMARY_OF_GROUP_IDS_KEY in annotation
        ):
            return True
    return False


def build_scoped_group_timeline(
    messages: list[Message],
    *,
    span_start: int,
    span_end: int,
    degraded: bool,
) -> ScopedTimeline:
    """Build the current-span group timeline from the annotated wire list."""
    ordered_ids: list[str] = []
    grouped: dict[str, list[Message]] = {}
    starts: dict[str, int] = {}
    kinds: dict[str, GroupKind] = {}
    for index, message in enumerate(messages):
        group_id = _group_id(message)
        group_kind = _group_kind(message)
        if group_id is None or group_kind is None:
            continue
        if group_id not in grouped:
            ordered_ids.append(group_id)
            grouped[group_id] = []
            starts[group_id] = index
            kinds[group_id] = group_kind
        grouped[group_id].append(message)

    scoped: list[ScopedGroup] = []
    opener_removed = False
    for group_id in ordered_ids:
        start = starts[group_id]
        kind = kinds[group_id]
        group_messages = grouped[group_id]
        if not span_start <= start < span_end or kind == "system" or is_structural_group(group_messages):
            continue
        included = tuple(
            message for message in group_messages if not message.additional_properties.get(EXCLUDED_KEY, False)
        )
        if not included:
            continue
        if kind == "user":
            sanitized = tuple(message for message in (_sanitize_user_message(item) for item in included) if message)
            if not sanitized:
                if start == span_start:
                    opener_removed = True
                continue
            included = sanitized
        integrity = (
            _tool_group_integrity(included)
            if kind == "tool_call"
            else all(bool(message.contents) for message in included)
        )
        scoped.append(ScopedGroup(group_id=group_id, kind=kind, messages=included, integrity=integrity))

    effective_degraded = degraded or opener_removed
    if effective_degraded:
        scoped.insert(
            0,
            ScopedGroup(
                group_id="phase4_degraded_preamble",
                kind="user",
                messages=(Message("user", [DEGRADED_SCOPED_PREAMBLE]),),
                integrity=True,
            ),
        )
    return ScopedTimeline(groups=tuple(scoped), degraded_opener=effective_degraded)


def prepare_scoped_slice(
    scoped_groups: list[ScopedGroup] | tuple[ScopedGroup, ...],
    *,
    tokenizer: TokenizerProtocol,
    slice_budget: int,
) -> PreparedScopedSlice | None:
    """Clone, validate, and progressively clip a completer slice to budget."""
    cloned_groups: list[ScopedGroup] = []
    for group in scoped_groups:
        if group.kind == "tool_call" and not group.integrity:
            # Omitting an invalid group here is unsafe: Phase 4 still owns the
            # original group and may exclude it after note generation, even
            # though the completer never saw its call/result payload.  Some
            # provider-hosted tools also use call/result layouts that are not
            # portable across every serializer.  The text reconstruction path
            # can represent both cases without replaying tool-bearing messages,
            # so demote the entire slice instead of silently dropping a group.
            return None
        cloned_groups.append(
            ScopedGroup(
                group_id=group.group_id,
                kind=group.kind,
                messages=tuple(clone_for_slice(message) for message in group.messages),
                integrity=group.integrity,
            )
        )

    messages = [message for group in cloned_groups for message in group.messages]
    if not _valid_slice_shape(cloned_groups, messages):
        return None

    estimated = _estimate_messages(messages, tokenizer)
    clipped_results = 0
    while estimated > slice_budget:
        result_candidates = _result_candidates(cloned_groups, tokenizer)
        candidate = next(
            (
                (content, slots, tokens)
                for content, slots, tokens in result_candidates
                if tokens > _TOOL_RESULT_FLOOR_TOKENS
            ),
            None,
        )
        if candidate is None:
            break
        content, slots, payload_tokens = candidate
        target = max(_TOOL_RESULT_FLOOR_TOKENS, payload_tokens - (estimated - slice_budget))
        if not _truncate_slots(slots, tokenizer=tokenizer, target_tokens=target):
            break
        _sync_result_text_mirror(content)
        clipped_results += 1
        estimated = _estimate_messages(messages, tokenizer)

    clipped_assistant_texts = 0
    if estimated > slice_budget:
        for slot in _assistant_text_slots(cloned_groups):
            current = tokenizer.count_tokens(slot.get())
            if current <= _ASSISTANT_TEXT_CAP_TOKENS:
                continue
            slot.set(_middle_truncate(slot.get(), tokenizer, _ASSISTANT_TEXT_CAP_TOKENS))
            clipped_assistant_texts += 1
        estimated = _estimate_messages(messages, tokenizer)

    if estimated > slice_budget or not _valid_slice_shape(cloned_groups, messages):
        return None
    return PreparedScopedSlice(
        messages=tuple(messages),
        estimated_tokens=estimated,
        clipped_results=clipped_results,
        clipped_assistant_texts=clipped_assistant_texts,
    )


def _sanitize_user_message(message: Message) -> Message | None:
    cloned = clone_for_slice(message)
    cloned.contents = [
        content
        for content in cloned.contents
        if not _strip_middleware_reminder(content) and _content_is_provider_visible(content)
    ]
    return cloned if cloned.contents else None


def _strip_middleware_reminder(content: Content) -> bool:
    if content.type != "text" or not isinstance(content.text, str):
        return False
    text = content.text.strip()
    if not text.startswith(f"{REMINDER_TAG_OPEN}\n") or not text.endswith(REMINDER_TAG_CLOSE):
        return False
    return not text.startswith(f"{REMINDER_TAG_OPEN}\n[LAST_WORDS] ")


def _tool_group_integrity(messages: tuple[Message, ...]) -> bool:
    if any(not _message_has_provider_visible_content(message) for message in messages):
        return False
    calls: list[Content] = []
    results: list[Content] = []
    call_message_indices: set[int] = set()
    result_message_indices: set[int] = set()
    for index, message in enumerate(messages):
        message_calls = [content for content in message.contents if content.type in TOOL_CALL_CONTENT_TYPES]
        message_results = [content for content in message.contents if content.type in _RESULT_CONTENT_TYPES]
        if message_calls:
            calls.extend(message_calls)
            call_message_indices.add(index)
        if message_results:
            results.extend(message_results)
            result_message_indices.add(index)
        if message.role == "tool" and len(message_results) != len(message.contents):
            return False

    call_ids = [content.call_id for content in calls]
    if not calls or any(not call_id for call_id in call_ids) or len(set(call_ids)) != len(call_ids):
        return False
    if len(call_message_indices) != 1 or not result_message_indices:
        return False
    call_index = next(iter(call_message_indices))
    if messages[call_index].role != "assistant":
        return False
    first_result_index = min(result_message_indices)
    if first_result_index <= call_index:
        return False
    if any(
        _message_has_provider_visible_content(messages[index]) for index in range(call_index + 1, first_result_index)
    ):
        return False
    if any(index <= call_index or messages[index].role != "tool" for index in result_message_indices):
        return False

    expected = Counter((content.call_id, _RESULT_TYPE_BY_CALL_TYPE.get(content.type)) for content in calls)
    actual = Counter((content.call_id, content.type) for content in results)
    return expected == actual


def _valid_slice_shape(groups: list[ScopedGroup], messages: list[Message]) -> bool:
    if (
        not messages
        or messages[0].role != "user"
        or any(not _message_has_provider_visible_content(message) for message in messages)
    ):
        return False
    return all(group.kind != "tool_call" or _tool_group_integrity(group.messages) for group in groups)


def _message_has_provider_visible_content(message: Message) -> bool:
    return any(_content_is_provider_visible(content) for content in message.contents)


def _content_is_provider_visible(content: Content) -> bool:
    if content.type == "text":
        return bool(content.text)
    if content.type == "text_reasoning":
        return bool(content.text or content.protected_data)
    return True


def _estimate_messages(messages: list[Message], tokenizer: TokenizerProtocol) -> int:
    return sum(estimate_message_tokens(message, tokenizer) for message in messages)


def _result_candidates(
    groups: list[ScopedGroup], tokenizer: TokenizerProtocol
) -> list[tuple[Content, list[_TextSlot], int]]:
    candidates: list[tuple[Content, list[_TextSlot], int]] = []
    for group in groups:
        if group.kind != "tool_call":
            continue
        for message in group.messages:
            for content in message.contents:
                if content.type not in _RESULT_CONTENT_TYPES:
                    continue
                slots = _result_text_slots(content)
                candidates.append((content, slots, sum(tokenizer.count_tokens(slot.get()) for slot in slots)))
    return sorted(candidates, key=lambda candidate: candidate[2], reverse=True)


def _result_text_slots(content: Content) -> list[_TextSlot]:
    slots: list[_TextSlot] = []
    if content.type in {"function_result", "hosted_tool_result", "search_tool_result"}:
        if content.items:
            _collect_text_slots(content.items, slots, owner=content, key="items")
        else:
            _collect_text_slots(content.result, slots, owner=content, key="result")
    elif content.type == "mcp_server_tool_result":
        _collect_text_slots(content.output, slots, owner=content, key="output")
    else:
        _collect_text_slots(content.outputs, slots, owner=content, key="outputs")
    return slots


def _sync_result_text_mirror(content: Content) -> None:
    """Keep ``result`` aligned with rich-item text after clipping.

    Function results store the same text twice: serializers prefer ``items``
    for rich output, while the kernel estimator deliberately counts only the
    compact ``result`` mirror. Treating both as independent slots would halve
    the effective per-result floor and double-count the clipping target.
    """
    if content.type not in {"function_result", "hosted_tool_result", "search_tool_result"} or not content.items:
        return
    content.result = "\n".join(item.text for item in content.items if item.type == "text" and item.text)


def _collect_text_slots(value: Any, slots: list[_TextSlot], *, owner: Any, key: str | int) -> None:
    if isinstance(value, str):
        slots.append(_TextSlot(owner, key))
        return
    if isinstance(value, Content):
        if value.type == "text" and isinstance(value.text, str):
            slots.append(_TextSlot(value, "text"))
        elif value.type == "shell_command_output":
            if isinstance(value.stdout, str):
                slots.append(_TextSlot(value, "stdout"))
            if isinstance(value.stderr, str):
                slots.append(_TextSlot(value, "stderr"))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _collect_text_slots(item, slots, owner=value, key=index)
        return
    if isinstance(value, dict):
        for child_key, item in value.items():
            _collect_text_slots(item, slots, owner=value, key=child_key)


def _assistant_text_slots(groups: list[ScopedGroup]) -> list[_TextSlot]:
    slots: list[_TextSlot] = []
    for group in groups:
        for message in group.messages:
            if message.role != "assistant":
                continue
            slots.extend(
                _TextSlot(content, "text")
                for content in message.contents
                if content.type == "text" and isinstance(content.text, str)
            )
    return sorted(slots, key=lambda slot: len(slot.get()), reverse=True)


def _truncate_slots(slots: list[_TextSlot], *, tokenizer: TokenizerProtocol, target_tokens: int) -> bool:
    changed = False
    while slots:
        counts = [(slot, tokenizer.count_tokens(slot.get())) for slot in slots]
        total = sum(count for _, count in counts)
        if total <= target_tokens:
            break
        slot, count = max(counts, key=lambda item: item[1])
        target = max(tokenizer.count_tokens(SLICE_TRUNCATION_MARKER), count - (total - target_tokens))
        truncated = _middle_truncate(slot.get(), tokenizer, target)
        if truncated == slot.get():
            slots.remove(slot)
            continue
        slot.set(truncated)
        changed = True
    return changed


def _middle_truncate(text: str, tokenizer: TokenizerProtocol, target_tokens: int) -> str:
    if tokenizer.count_tokens(text) <= target_tokens:
        return text
    marker_tokens = tokenizer.count_tokens(SLICE_TRUNCATION_MARKER)
    if target_tokens <= marker_tokens:
        return SLICE_TRUNCATION_MARKER
    low = 0
    high = len(text)
    while low < high:
        keep = (low + high + 1) // 2
        head = (keep + 1) // 2
        candidate = f"{text[:head]}{SLICE_TRUNCATION_MARKER}{text[len(text) - (keep - head) :]}"
        if tokenizer.count_tokens(candidate) <= target_tokens:
            low = keep
        else:
            high = keep - 1
    head = (low + 1) // 2
    return f"{text[:head]}{SLICE_TRUNCATION_MARKER}{text[len(text) - (low - head) :]}"

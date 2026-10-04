# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared cleaning and truncation helpers for tool output."""

from __future__ import annotations

import re

from .tokenizer import MixedLanguageTokenizer

_ANSI_RE = re.compile(
    r"\x1b(?:"
    r"\[[0-?]*[ -/]*[@-~]"  # CSI sequences (colors, cursor, etc.)
    r"|\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC sequences (title, hyperlinks)
    r"|[()][0-2AB]"  # Charset designation
    r"|[\x20-\x2F][\x30-\x7E]"  # Two-byte ESC sequences
    r"|[=>NOcDEHMZ78\\]"  # Single-byte ESC sequences and a lone string terminator (ST)
    r")"
)

_tokenizer = MixedLanguageTokenizer()


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from *text*."""
    return _ANSI_RE.sub("", text)


def process_carriage_returns(text: str) -> str:
    r"""Collapse carriage-return overwrites while preserving normal newlines."""
    lines = text.split("\n")
    result: list[str] = []
    for index, line in enumerate(lines):
        # PTY ONLCR expands CRLF to CRCRLF. Strip all trailing CRs from
        # completed lines; scanning by line stays linear even for long CR runs.
        if index < len(lines) - 1:
            line = line.rstrip("\r")
        if "\r" in line:
            line = line.rsplit("\r", 1)[-1]
        result.append(line)
    return "\n".join(result)


def _join_sections(*sections: str) -> str:
    return "\n".join(section for section in sections if section)


def _line_marker(omitted: int, total_tokens: int) -> str:
    return f"[...truncated {omitted} lines (estimated {total_tokens} tokens total)...]"


def _char_marker(omitted: int, total_tokens: int) -> str:
    return f"[...truncated ~{omitted} chars (estimated {total_tokens} tokens total)...]"


def _largest_prefix_within_budget(text: str, budget: int) -> str:
    """Return the longest prefix whose full estimate fits *budget*."""
    low = 0
    high = len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if _tokenizer.count_tokens(text[:mid]) <= budget:
            low = mid
        else:
            high = mid - 1
    return text[:low]


def _char_truncate(
    text: str,
    max_tokens: int,
    *,
    head_ratio: float,
    truncation_suffix: str,
    total_tokens: int,
) -> str:
    """Truncate at character granularity, measuring every complete candidate."""

    def candidate(kept: int, suffix: str) -> str:
        head_count = int(kept * head_ratio)
        tail_count = kept - head_count
        head = text[:head_count]
        tail = text[len(text) - tail_count :] if tail_count else ""
        marker = _char_marker(len(text) - kept, total_tokens)
        return _join_sections(head, marker, suffix, tail)

    # A suffix is useful only when the marker and suffix fit together.  If
    # they do not, degrade to an ordinary bounded truncation without it.
    suffix = truncation_suffix
    if suffix and _tokenizer.count_tokens(candidate(0, suffix)) > max_tokens:
        suffix = ""

    if _tokenizer.count_tokens(candidate(0, suffix)) > max_tokens:
        marker = _char_marker(len(text), total_tokens)
        return _largest_prefix_within_budget(marker, max_tokens)

    low = 0
    high = max(0, len(text) - 1)
    while low < high:
        mid = (low + high + 1) // 2
        if _tokenizer.count_tokens(candidate(mid, suffix)) <= max_tokens:
            low = mid
        else:
            high = mid - 1

    result = candidate(low, suffix)
    while low > 0 and _tokenizer.count_tokens(result) > max_tokens:
        low -= 1
        result = candidate(low, suffix)
    return result


def truncate_output(
    text: str,
    max_tokens: int,
    *,
    head_ratio: float = 1 / 3,
    truncation_suffix: str = "",
    force_truncation: bool = False,
) -> str:
    """Return a head/tail view of *text* bounded by *max_tokens*.

    The complete returned string is measured after assembly, including the
    marker and optional suffix.  The suffix is ignored when the original text
    fits unless ``force_truncation`` is set; multi-item callers use that mode
    when later items must be represented by an omission footer.  ``max_tokens``
    must be at least one; zero is a caller-level disable value and is
    intentionally outside this helper's domain.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if not 0 <= head_ratio <= 1:
        raise ValueError("head_ratio must be between 0 and 1")

    total_tokens = _tokenizer.count_tokens(text)
    if total_tokens <= max_tokens and not force_truncation:
        return text

    lines = text.splitlines()
    if len(lines) <= 1:
        return _char_truncate(
            text,
            max_tokens,
            head_ratio=head_ratio,
            truncation_suffix=truncation_suffix,
            total_tokens=total_tokens,
        )

    initial_marker = _line_marker(len(lines), total_tokens)
    footer = _join_sections(initial_marker, truncation_suffix)
    if _tokenizer.count_tokens(footer) > max_tokens:
        return _char_truncate(
            text,
            max_tokens,
            head_ratio=head_ratio,
            truncation_suffix=truncation_suffix,
            total_tokens=total_tokens,
        )

    approximate_payload_budget = max_tokens - _tokenizer.count_tokens(footer)
    head_budget = int(approximate_payload_budget * head_ratio)
    tail_budget = approximate_payload_budget - head_budget

    head_end = 0
    head_used = 0
    while head_end < len(lines):
        line_cost = _tokenizer.count_tokens(lines[head_end]) + 1
        if head_used + line_cost > head_budget:
            break
        head_used += line_cost
        head_end += 1

    tail_start = len(lines)
    tail_used = 0
    while tail_start > head_end:
        line_cost = _tokenizer.count_tokens(lines[tail_start - 1]) + 1
        if tail_used + line_cost > tail_budget:
            break
        tail_used += line_cost
        tail_start -= 1

    # Forced multi-item truncation must actually omit payload; otherwise a
    # zero-line marker could claim truncation while all lines survive.
    if force_truncation and tail_start <= head_end:
        return _char_truncate(
            text,
            max_tokens,
            head_ratio=head_ratio,
            truncation_suffix=truncation_suffix,
            total_tokens=total_tokens,
        )

    # A giant boundary line would otherwise discard all usable payload.
    if head_end == 0 and tail_start == len(lines):
        return _char_truncate(
            text,
            max_tokens,
            head_ratio=head_ratio,
            truncation_suffix=truncation_suffix,
            total_tokens=total_tokens,
        )

    def assemble() -> str:
        omitted = tail_start - head_end
        marker = _line_marker(omitted, total_tokens)
        return _join_sections("\n".join(lines[:head_end]), marker, truncation_suffix, "\n".join(lines[tail_start:]))

    result = assemble()
    while _tokenizer.count_tokens(result) > max_tokens and (head_end > 0 or tail_start < len(lines)):
        head_share = head_used / head_budget if head_budget else (1.0 if head_end else 0.0)
        tail_share = tail_used / tail_budget if tail_budget else (1.0 if tail_start < len(lines) else 0.0)
        if head_end > 0 and (tail_start == len(lines) or head_share >= tail_share):
            head_end -= 1
            head_used -= _tokenizer.count_tokens(lines[head_end]) + 1
        else:
            tail_used -= _tokenizer.count_tokens(lines[tail_start]) + 1
            tail_start += 1
        result = assemble()

    # The complete-string recount is authoritative.  If the line form cannot
    # satisfy it (usually only at tiny budgets), the character path can.
    if _tokenizer.count_tokens(result) > max_tokens:
        return _char_truncate(
            text,
            max_tokens,
            head_ratio=head_ratio,
            truncation_suffix=truncation_suffix,
            total_tokens=total_tokens,
        )
    return result


def truncate_content_texts(
    texts: list[str],
    budget: int,
    *,
    truncation_suffix: str = "",
) -> list[str | None] | None:
    """Bound ordered text payloads while preserving their positional shape."""
    if budget < 1:
        raise ValueError("budget must be at least 1")
    if _tokenizer.count_tokens("\n".join(texts)) <= budget:
        return None

    kept: list[str] = []
    for index, text in enumerate(texts):
        dropped_count = len(texts) - index - 1
        dropped_line = f"[{dropped_count} further text items omitted.]" if dropped_count else ""
        footer_suffix = _join_sections(dropped_line, truncation_suffix)
        marker = _char_marker(len(text), _tokenizer.count_tokens(text))
        fixed_footer = _join_sections(marker, footer_suffix)
        candidate_prefix = "\n".join([*kept, text])
        keep_whole_candidate = _join_sections(candidate_prefix, fixed_footer)
        if _tokenizer.count_tokens(keep_whole_candidate) <= budget:
            kept.append(text)
            continue

        prefix_with_sep = f"{'\n'.join(kept)}\n" if kept else ""
        prefix_cost = _tokenizer.count_tokens(prefix_with_sep) if prefix_with_sep else 0
        slot = budget - prefix_cost
        result: list[str | None] = [*kept]
        if slot < 1:
            result.extend([None] * (len(texts) - len(result)))
            return result

        truncated = truncate_output(text, slot, truncation_suffix=footer_suffix, force_truncation=True)
        result.append(truncated)
        result.extend([None] * (len(texts) - len(result)))

        joined = "\n".join(value for value in result if value is not None)
        while slot > 1 and _tokenizer.count_tokens(joined) > budget:
            overflow = _tokenizer.count_tokens(joined) - budget
            slot = max(1, slot - max(1, overflow))
            result[index] = truncate_output(
                text,
                slot,
                truncation_suffix=footer_suffix,
                force_truncation=True,
            )
            joined = "\n".join(value for value in result if value is not None)
        if _tokenizer.count_tokens(joined) > budget:
            result[index] = None
        return result

    # The aggregate pre-check guarantees the loop normally truncates.  Keep a
    # defensive shape-preserving fallback for estimator boundary oddities.
    return [*kept, *([None] * (len(texts) - len(kept)))]

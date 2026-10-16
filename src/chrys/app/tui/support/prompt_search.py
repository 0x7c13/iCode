# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tiered search over prompt-history entries.

Both sides are NFKC-normalized and case-folded, so full-width letters and
digits match their ASCII forms. The query splits on whitespace into terms, and
an entry matches only when every term matches it. Whitespace between two
letters or digits separates words, and a line break separates lines; any
other whitespace (beside punctuation, or beside Han or kana, within a line)
is layout that people leave out or add freely. So
whether characters, a term or several terms are contiguous is judged on both
sides in a compact form, where layout whitespace is removed and a word or
line break is one space: ``plan:the`` is ``plan: the``, but ``there`` is not
``the rest``, nor ``存在`` a heading ending in ``存`` above a line starting
with ``在``. Each term matches at the best of these tiers, strongest first:

- ``WORD``: the term occurs as a whole word.
- ``PREFIX``: the term occurs starting at a word start.
- ``SUBSTRING``: the term occurs anywhere.
- ``SPACED``: the term occurs in the compact form (``plan:the`` in
  ``plan: the``).

Every tier is the term as typed: letters gathered from here and there
(fuzzy matches, word initials) are what a long prompt holds by chance, so
they never match.

A word is a run of letters and digits; each Han or kana character is a word by
itself, since those scripts are written without spaces. Entries rank by an
exact match of the whole entry first, then by holding all the query's terms
as one run (in order and contiguous in the compact form), then by their
weakest term's tier, then by the sum of their tiers; anything still equal
keeps its input order (newest first for prompt history). Tiers always
outrank recency.

A preview shows where the query occurs, which is a separate question from
how the entry ranked. The query's terms are split into the fewest runs the
entry holds; any part of a run is a run too, so taking the longest run from
the left each time gives the fewest, and typing another term changes only the
last run. A run of several terms is marked whole, whitespace included,
wherever the preview shows it, and lone copies of its terms are not; a term
that is a run by itself is marked wherever it occurs at the tier it earned.
Every term is thus marked, so each row shows why it matched.

Search runs on every keystroke against entries of any length, so
:class:`PromptSearchIndex` normalizes each entry once, and every per-query
scan is a compiled regular expression or ``str`` method rather than a Python
loop over characters.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache

from rich.cells import get_character_cell_size


class MatchTier(IntEnum):
    """How strongly one query term matches an entry; higher is stronger."""

    SPACED = 1
    SUBSTRING = 2
    PREFIX = 3
    WORD = 4


# Hiragana, Katakana, and the CJK Unified/Compatibility Ideograph blocks.
_UNSPACED = "぀-ヿ㐀-䶿一-鿿豈-﫿\U00020000-\U0003ffff"
# A character that continues a word with a neighbor of the same kind:
# alphanumeric (``\w`` without ``_``) and not from an unspaced script.
_JOINS = rf"[^\W_{_UNSPACED}]"
_JOINS_RE = re.compile(_JOINS)
_LINE_SEPARATORS = frozenset("\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029")
# Whitespace within a line.
_INLINE_SPACE = "[^\\S" + re.escape("".join(sorted(_LINE_SEPARATORS))) + "]"
# Whitespace that is layout: within a line and not between two word-joining
# characters. The possessive quantifier takes a whole stretch, so one between
# two such characters is never cut down to a part that qualifies.
_LAYOUT_SPACE_RE = re.compile(rf"{_INLINE_SPACE}++(?!{_JOINS})|(?<!{_JOINS}){_INLINE_SPACE}++")
_SPACE_RUN_RE = re.compile(r"\s+")


def _compact(text: str) -> str:
    """Return *text* without layout whitespace, each word or line break being one space."""
    return _SPACE_RUN_RE.sub(" ", _LAYOUT_SPACE_RE.sub("", text))


def normalize_for_search(text: str) -> str:
    """Return *text* NFKC-normalized and case-folded for comparison."""
    return unicodedata.normalize("NFKC", text).casefold()


@dataclass(frozen=True, slots=True)
class _Entry:
    prompt: str
    text: str
    # ``text`` in the compact form, where contiguity is judged.
    compact: str
    # Whether every character of ``text`` sits at its offset in ``prompt``.
    in_place: bool


def _index_entry(prompt: str) -> _Entry:
    text = normalize_for_search(prompt)
    return _Entry(
        prompt=prompt,
        text=text,
        compact=_compact(text),
        in_place=prompt.isascii() or _normalizes_in_place(prompt, text),
    )


@dataclass(frozen=True, slots=True)
class _TermPatterns:
    term: str
    word: re.Pattern[str]
    prefix: re.Pattern[str]
    substring: re.Pattern[str]
    spaced: re.Pattern[str]


@lru_cache(maxsize=64)
def _term_patterns(term: str) -> _TermPatterns:
    """Compile the per-term scans once per distinct term.

    A start (end) boundary is a lookbehind (lookahead) only when the term's
    first (last) character joins words; otherwise that side is always a
    boundary. The start check looks back from after the term so each pattern
    begins with the literal, which the regex engine finds by fast search
    instead of trying every position of a long prompt.
    """
    escaped = re.escape(term)
    start = f"(?<!{_JOINS}{escaped})" if _JOINS_RE.fullmatch(term[0]) else ""
    end = f"(?!{_JOINS})" if _JOINS_RE.fullmatch(term[-1]) else ""
    return _TermPatterns(
        term=term,
        word=re.compile(escaped + start + end),
        prefix=re.compile(escaped + start),
        substring=re.compile(escaped),
        spaced=_spaced_pattern(term),
    )


def _term_match(patterns: _TermPatterns, entry: _Entry) -> MatchTier | None:
    """Return the strongest tier at which a term matches *entry*, or None."""
    term = patterns.term
    text = entry.text
    if term not in text:
        return MatchTier.SPACED if term in entry.compact else None
    if patterns.word.search(text):
        return MatchTier.WORD
    if patterns.prefix.search(text):
        return MatchTier.PREFIX
    return MatchTier.SUBSTRING


@dataclass(frozen=True, slots=True)
class PromptMatch:
    """One ranked prompt plus the parts of it to preview.

    ``windows`` are the ``[start, end)`` slices of ``prompt`` that fit the
    row, in order and apart; a gap between them, or text cut before the first
    or after the last, is shown as an ellipsis. The last window goes on to
    ``tail_end`` past the row's edge, for a highlighted row to scroll to.
    ``spans`` are the ``[start, end)`` slices inside the windows and that tail
    whose characters matched the query as the module docstring describes,
    sorted and merged.
    """

    prompt: str
    windows: tuple[tuple[int, int], ...]
    spans: tuple[tuple[int, int], ...]
    tail_end: int


# A preview row shows each line break, a CRLF being one, as LINE_BREAK; the
# row's renderer splits on LINE_SEPARATOR_RE so it shows what the cells here count.
LINE_SEPARATOR_RE = re.compile("\r\n|[" + re.escape("".join(sorted(_LINE_SEPARATORS))) + "]")
LINE_BREAK = " ↵ "
_LINE_BREAK_CELLS = sum(get_character_cell_size(char) for char in LINE_BREAK)
# Context to keep before a match, as a share of the cells left after the match.
_CONTEXT_BEFORE_SHARE = 3
# Each of several excerpts keeps at least this many cells of context around its match.
_EXCERPT_CONTEXT_CELLS = 4


def _tier_pattern(patterns: _TermPatterns, text: str) -> tuple[re.Pattern[str], re.Match[str]] | None:
    """Return the word, prefix, substring or spaced pattern that earns the term's tier in *text*, and its first match."""
    for pattern in (patterns.word, patterns.prefix, patterns.substring, patterns.spaced):
        found = pattern.search(text)
        if found is not None:
            return pattern, found
    return None


def _normalizes_in_place(prompt: str, text: str) -> bool:
    """Return whether *text*, the normalization of *prompt*, keeps each character at its offset.

    It does when every distinct character normalizes to one character and
    none joins the one before it: then nothing reorders, and a composition
    would have shortened *text*.
    """
    if len(text) != len(prompt):
        return False
    return all(len(normalize_for_search(char)) == 1 and not _joins_previous(char) for char in set(prompt))


def _joins_previous(char: str) -> bool:
    """Return whether *char* normalizes together with the character before it.

    Combining marks compose with their base, and so do compatibility forms
    that decompose to one (half-width voiced marks); Hangul vowel and final
    jamo compose with the syllable before them.
    """
    if char < "\u0300":
        return False  # ASCII and Latin-1 decompose to a base character first
    first = unicodedata.normalize("NFKD", char)[:1] or char
    return unicodedata.combining(first) != 0 or "\u1160" <= first <= "\u11ff"


def _normalized_with_sources(prompt: str) -> tuple[str, list[tuple[int, int]]]:
    """Normalize *prompt* one character and its marks at a time, mapping each output character to its source.

    Accents and jamo compose with their base inside such a cluster, so this
    matches the whole-string normalization the ranking used, and each output
    character knows the ``[start, end)`` slice of *prompt* it came from. Where
    some other composition crosses clusters (two vowel signs of some Indic
    scripts), an occurrence may go unfound and the preview falls back to the
    prompt's head.
    """
    pieces: list[str] = []
    sources: list[tuple[int, int]] = []
    start = 0
    for end in range(1, len(prompt) + 1):
        if end < len(prompt) and _joins_previous(prompt[end]):
            continue
        piece = normalize_for_search(prompt[start:end])
        pieces.append(piece)
        sources.extend([(start, end)] * len(piece))
        start = end
    return "".join(pieces), sources


def _char_cells(prompt: str, index: int) -> int:
    """Return the terminal cells ``prompt[index]`` takes in a preview row."""
    char = prompt[index]
    if char in _LINE_SEPARATORS:
        # CRLF is one line break, counted on its CR: a window cut inside it shows the CR's.
        return 0 if char == "\n" and prompt[index - 1 : index] == "\r" else _LINE_BREAK_CELLS
    if ord(char) < 0x20 or 0x7F <= ord(char) <= 0x9F:
        return 1  # shown as a replacement character
    return get_character_cell_size(char)


def _extend(prompt: str, start: int, end: int, before: int, after: int, low: int, high: int) -> tuple[int, int]:
    """Grow ``[start, end)`` by up to *before* cells to the left and *after* to the right.

    Growth stays within ``[low, high)``; cells one side cannot use go to the other.
    """
    while before > 0 and start > low:
        cells = _char_cells(prompt, start - 1)
        if cells > before:
            break
        before -= cells
        start -= 1
    after += before
    while after > 0 and end < high:
        cells = _char_cells(prompt, end)
        if cells > after:
            break
        after -= cells
        end += 1
    while after > 0 and start > low:
        cells = _char_cells(prompt, start - 1)
        if cells > after:
            break
        after -= cells
        start -= 1
    return start, end


def _span_cells(prompt: str, start: int, end: int) -> int:
    return sum(_char_cells(prompt, index) for index in range(start, end))


def _shrink_to(prompt: str, start: int, end: int, cells: int) -> int:
    """Return the end that keeps ``prompt[start:end]`` within *cells*."""
    used = 0
    for index in range(start, end):
        used += _char_cells(prompt, index)
        if used > cells:
            return index
    return end


def _fit_windows(prompt: str, groups: list[tuple[int, int]], cells: int) -> list[tuple[int, int]]:
    """Return windows covering the match *groups* (sorted, disjoint) within *cells* terminal cells.

    One window when every group fits with room for ellipses; otherwise one
    excerpt per group, dropping trailing groups that leave an excerpt too
    narrow to read. Each window gets a third of its spare cells as context
    before its match and the rest after it.
    """
    first, last = groups[0][0], groups[-1][1]
    # Groups can be far apart in a long prompt: stop counting once the row is full.
    if _shrink_to(prompt, first, last, cells - 2) == last:
        spare = cells - _span_cells(prompt, first, last)
        start, end = _extend(prompt, first, last, spare // _CONTEXT_BEFORE_SHARE, 0, 0, len(prompt))
        start, end = _extend(prompt, start, end, 0, cells - _span_cells(prompt, start, end), 0, len(prompt))
        # Make room for the ellipses the cut ends need, from the side with more context.
        while _span_cells(prompt, start, end) + (start > 0) + (end < len(prompt)) > cells:
            if end > last and (start >= first or _span_cells(prompt, last, end) >= _span_cells(prompt, start, first)):
                end -= 1
            elif start < first:
                start += 1
            else:
                break
        return [(start, end)]
    while len(groups) > 1 and (cells - len(groups) - 1) // len(groups) < 2 * _EXCERPT_CONTEXT_CELLS + 1:
        groups = groups[:-1]
    count = len(groups)
    share = max(1, (cells - count - 1) // count)
    windows: list[tuple[int, int]] = []
    for index, (low, high) in enumerate(groups):
        floor = windows[-1][1] if windows else 0
        ceiling = groups[index + 1][0] if index + 1 < count else len(prompt)
        high = _shrink_to(prompt, low, high, share)
        if high == low:
            high = low + 1
        spare = max(0, share - _span_cells(prompt, low, high))
        before = spare // _CONTEXT_BEFORE_SHARE
        windows.append(_extend(prompt, low, high, before, spare - before, floor, ceiling))
    return windows


# One run of the query: its pattern when it spans several terms, and its terms.
_Run = tuple[re.Pattern[str] | None, tuple[_TermPatterns, ...]]


def _preview(entry: _Entry, runs: Sequence[_Run], cells: int, head_chars: int) -> PromptMatch:
    """Choose the preview windows of *entry* for a row *cells* wide and the matches inside them.

    Without a query the preview is the head, *head_chars* long, for the row to
    scroll. With one, the first occurrence of each of the query's *runs* is
    shown when the row has room (for a run of one term, the occurrence that
    earned its tier): in one window when they fit together, otherwise in
    excerpts. Only those occurrences place the windows; repeats are then
    marked wherever the windows and the tail show them.

    Normalization usually keeps each character at its offset, so offsets into
    the normalized text are offsets into the prompt. When it does not (``ß``,
    ligatures, decomposed accents, which can also cancel out in length)
    occurrences are found in a cluster-by-cluster normalization and mapped
    back to the characters they came from. Where that misses a run the whole
    normalization found, its terms are marked one by one.
    """
    prompt = entry.prompt
    head = min(len(prompt), head_chars)
    if not runs or not prompt:
        return PromptMatch(prompt, ((0, head),), (), head)
    sources: list[tuple[int, int]] | None = None
    if entry.in_place:
        text = entry.text
    else:
        text, sources = _normalized_with_sources(prompt)

    def in_prompt(positions: Sequence[int]) -> list[int]:
        if sources is None:
            return list(positions)
        return sorted({index for position in positions for index in range(*sources[position])})

    occurrences: list[list[int]] = []
    # Each pattern to mark every match of, and how far before a window its match can start.
    repeated: list[tuple[re.Pattern[str], int | None]] = []
    singles: list[_TermPatterns] = []
    for run, run_terms in runs:
        found = run.search(text) if run is not None else None
        if run is None or found is None:
            singles.extend(term for term in run_terms if term not in singles)
            continue
        occurrences.append(in_prompt(range(found.start(), found.end())))
        repeated.append((run, None))
    for term_patterns in singles:
        tier = _tier_pattern(term_patterns, text)
        if tier is None:
            continue  # only where cluster-by-cluster normalization misses what the whole one found
        pattern, found = tier
        occurrences.append(in_prompt(range(found.start(), found.end())))
        # A spaced match can hold any amount of whitespace, so it can start any distance before a window.
        repeated.append((pattern, None if pattern is term_patterns.spaced else len(term_patterns.term) - 1))
    if not occurrences:
        return PromptMatch(prompt, ((0, head),), (), head)
    groups: list[tuple[int, int]] = []
    for low, high in sorted((positions[0], positions[-1] + 1) for positions in occurrences):
        if groups and low <= groups[-1][1]:
            groups[-1] = (groups[-1][0], max(groups[-1][1], high))
        else:
            groups.append((low, high))
    windows = _fit_windows(prompt, groups, max(cells, 2 * _EXCERPT_CONTEXT_CELLS + 3))
    tail_end = min(len(prompt), windows[-1][1] + head_chars)
    scrolled = [*windows[:-1], (windows[-1][0], tail_end)]
    marked = {position for positions in occurrences for position in positions}
    for pattern, lead in repeated:
        # A long prompt can repeat a term many times before the windows; skip to them when offsets agree.
        skip = 0 if sources is not None or lead is None else max(0, windows[0][0] - lead)
        for found in pattern.finditer(text, skip):
            positions = in_prompt(range(found.start(), found.end()))
            if positions[0] >= tail_end:
                break
            marked.update(positions)
    shown = sorted(position for position in marked if any(low <= position < high for low, high in scrolled))
    spans: list[tuple[int, int]] = []
    for position in shown:
        if spans and spans[-1][1] == position:
            spans[-1] = (spans[-1][0], position + 1)
        else:
            spans.append((position, position + 1))
    return PromptMatch(prompt, tuple(windows), tuple(spans), tail_end)


@dataclass(frozen=True, slots=True)
class PromptSearchResult:
    """The best matches of one search and how many prompts matched in all."""

    matches: list[PromptMatch]
    total: int


class PromptSearchIndex:
    """Prompts normalized once, newest first, ready to rank against any query."""

    __slots__ = ("_entries",)

    def __init__(self, prompts: Sequence[str], *, previous: PromptSearchIndex | None = None) -> None:
        """Index *prompts*, reusing the work *previous* already did for any of them.

        Reopening the history list usually finds the same prompts plus a few
        new ones, so only those are normalized again.
        """
        known = {entry.prompt: entry for entry in previous._entries} if previous is not None else {}
        self._entries = tuple(known.get(prompt) or _index_entry(prompt) for prompt in prompts)

    def __len__(self) -> int:
        return len(self._entries)

    def same_prompts(self, other: PromptSearchIndex) -> bool:
        """Return whether *other* indexes the same prompts in the same order."""
        return len(self._entries) == len(other._entries) and all(
            mine.prompt == theirs.prompt for mine, theirs in zip(self._entries, other._entries, strict=True)
        )

    def _ranked_entries(
        self, terms: list[str], patterns: list[_TermPatterns], should_stop: Callable[[], bool] | None = None
    ) -> list[_Entry] | None:
        """Rank the entries matching every term, or return None once *should_stop* says so.

        An entry equal to the query comes first, then entries holding the
        query's words as one run in order, then the tiers decide.
        """
        if not terms:
            return list(self._entries)
        joined_query = _compact(" ".join(terms))
        phrase = len(terms) > 1
        ranked: list[tuple[bool, bool, int, int, int, _Entry]] = []
        for order, entry in enumerate(self._entries):
            if should_stop is not None and should_stop():
                return None
            tiers: list[MatchTier] = []
            for term_patterns in patterns:
                tier = _term_match(term_patterns, entry)
                if tier is None:
                    break
                tiers.append(tier)
            else:
                exact = entry.compact == joined_query
                in_run = phrase and joined_query in entry.compact
                ranked.append((not exact, not in_run, -min(tiers), -sum(tiers), order, entry))
        ranked.sort(key=lambda item: item[:5])
        return [item[-1] for item in ranked]

    def search(
        self,
        query: str,
        *,
        limit: int,
        cells: int,
        head_chars: int,
        should_stop: Callable[[], bool] | None = None,
    ) -> PromptSearchResult:
        """Return the best *limit* matches, each previewed for a row *cells* terminal cells wide.

        Without a query a preview is the prompt's first *head_chars* characters;
        with one, its last window runs on for up to *head_chars* more. A thread
        cannot be stopped from outside, so *should_stop* is asked once per
        prompt; once it returns True the search ends with no matches.
        """
        terms = normalize_for_search(query).split()
        patterns = [_term_patterns(term) for term in terms]
        ranked = self._ranked_entries(terms, patterns, should_stop)
        if ranked is None:
            return PromptSearchResult(matches=[], total=0)
        matches: list[PromptMatch] = []
        for entry in ranked[:limit]:
            if should_stop is not None and should_stop():
                return PromptSearchResult(matches=[], total=0)
            runs = [
                (
                    _spaced_pattern(_compact(" ".join(terms[start:end]))) if end - start > 1 else None,
                    tuple(patterns[start:end]),
                )
                for start, end in (_runs(terms, entry.compact) if terms else ())
            ]
            matches.append(_preview(entry, runs, cells, head_chars))
        return PromptSearchResult(matches=matches, total=len(ranked))


def _runs(terms: list[str], compact: str) -> tuple[tuple[int, int], ...]:
    """Split *terms* into the fewest runs that occur in *compact*, as ``[start, end)`` slices of *terms*."""
    runs: list[tuple[int, int]] = []
    start = 0
    while start < len(terms):
        end = start + 1
        while end < len(terms) and _compact(" ".join(terms[start : end + 1])) in compact:
            end += 1
        runs.append((start, end))
        start = end
    return tuple(runs)


@lru_cache(maxsize=64)
def _spaced_pattern(compact: str) -> re.Pattern[str]:
    """Compile the scan that finds in a text what ``in`` finds of *compact* in its compact form.

    A space in *compact* is any whitespace in the text; between two other
    characters there may be whitespace within a line, unless both join words.
    """
    parts: list[str] = []
    previous = ""
    for char in compact:
        if char == " ":
            parts.append(r"\s+")
        else:
            if previous and previous != " " and not (_JOINS_RE.fullmatch(previous) and _JOINS_RE.fullmatch(char)):
                parts.append(_INLINE_SPACE + "*")
            parts.append(re.escape(char))
        previous = char
    return re.compile("".join(parts))

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for tiered prompt-history search."""

from __future__ import annotations

import unicodedata

import pytest
from rich.cells import cell_len

from chrys.app.tui.support.prompt_search import (
    PromptSearchIndex,
    PromptSearchResult,
    normalize_for_search,
)


def rank_prompts(prompts: list[str], query: str) -> list[str]:
    """Return the prompts matching *query*, strongest first."""
    matches = PromptSearchIndex(prompts).search(query, limit=len(prompts), cells=80, head_chars=60).matches
    return [match.prompt for match in matches]


def test_empty_query_keeps_every_prompt_in_order() -> None:
    prompts = ["newest", "middle", "oldest"]

    assert rank_prompts(prompts, "") == prompts
    assert rank_prompts(prompts, "   ") == prompts


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("sesn", []),
        ("SESSION", ["fix the session bug"]),
        ("bug session", ["fix the session bug"]),
        ("gub", []),
    ],
    ids=["letters-apart", "case-insensitive", "terms-any-order", "order-matters"],
)
def test_terms_match_in_any_order(query: str, expected: list[str]) -> None:
    assert rank_prompts(["fix the session bug", "write docs"], query) == expected


def test_every_term_must_match() -> None:
    assert rank_prompts(["fix the session bug", "session storage notes"], "session bug") == ["fix the session bug"]


def test_whole_word_outranks_word_prefix_regardless_of_recency() -> None:
    prompts = ["revert local dirty changes", "run tests locally", "LOC for current repo?"]

    assert rank_prompts(prompts, "loc") == [
        "LOC for current repo?",
        "revert local dirty changes",
        "run tests locally",
    ]


def test_tiers_rank_word_prefix_substring() -> None:
    prompts = [
        "s-x-r spread",  # letters apart: no match
        "classroom notes",  # substring: "sr" inside "classroom"
        "sr tool",  # whole word
        "srcmap tool",  # word prefix
    ]

    assert rank_prompts(prompts, "sr") == [
        "sr tool",
        "srcmap tool",
        "classroom notes",
    ]


def test_exact_entry_outranks_newer_entries_with_the_same_word() -> None:
    assert rank_prompts(["fix the bug in session", "fix"], "fix") == ["fix", "fix the bug in session"]
    assert rank_prompts(["fix  bug now", "fix   bug"], "fix bug") == ["fix   bug", "fix  bug now"]


def test_query_words_in_one_run_outrank_the_same_words_apart() -> None:
    # Newest first: without the run, recency alone would put the scattered match ahead.
    prompts = ["restore the session", "session restore failed", "session restore"]

    assert rank_prompts(prompts, "session restore") == [
        "session restore",
        "session restore failed",
        "restore the session",
    ]


def test_a_run_outranks_scattered_words_of_a_stronger_tier() -> None:
    prompts = ["restore the session", "session restored ok"]

    assert rank_prompts(prompts, "session restore") == ["session restored ok", "restore the session"]


def test_weakest_term_decides_before_tier_sum() -> None:
    # Both terms whole words beats one whole word plus one substring.
    prompts = ["session restore and debugging", "restore session bug"]

    assert rank_prompts(prompts, "session bug") == ["restore session bug", "session restore and debugging"]


def test_equal_tiers_keep_input_order() -> None:
    prompts = ["session storage notes", "a much longer prompt that also mentions the session"]

    assert rank_prompts(prompts, "session") == prompts
    assert rank_prompts(list(reversed(prompts)), "session") == list(reversed(prompts))


def test_letters_gathered_from_here_and_there_are_no_match() -> None:
    # A long prompt holds most short terms this way by chance: "bun" in "taBle groUp backeNd".
    assert rank_prompts(["table group backend", "background"], "bun") == []
    assert rank_prompts(["fix the session bug"], "sesn") == []
    # Nor do word initials: "in count on explain" spells "icoe".
    assert rank_prompts(["docs session restore", "docs in count on explain"], "dsr") == []
    assert rank_prompts(["docs in count on explain", "the iCode repo"], "icoe") == []


def test_whitespace_beside_punctuation_or_cjk_need_not_be_typed() -> None:
    assert rank_prompts(["The plan: the other"], "plan:the") == ["The plan: the other"]
    assert rank_prompts(["现在 iCode 怎么全选"], "现在icode") == ["现在 iCode 怎么全选"]
    # Typed where the prompt has none, it still keeps the query in one run.
    assert rank_prompts(["plan:the other", "the plan: notes, other"], "plan : the other") == [
        "plan:the other",
        "the plan: notes, other",
    ]


def test_whitespace_between_letters_is_a_word_break() -> None:
    # A long prompt joins many words into others by chance: "the rest" holds no "there".
    assert rank_prompts(["the rest of it", "in to the log"], "there") == []
    assert rank_prompts(["in to the log"], "into") == []
    # "log in" is a run in "log in to it", not in "login page".
    assert rank_prompts(["login page", "log in to it"], "log in") == ["log in to it", "login page"]


def test_a_line_break_is_never_layout() -> None:
    # A heading ending in "存" above a line starting with "在" holds no "存在".
    assert rank_prompts(["## 提示词缓存\n\n在 icode 默认的"], "存在") == []
    assert rank_prompts(["plan:\nthe"], "plan:the") == []
    # Between words it is a break like a space, so a run of the query may cross it and rank first.
    assert rank_prompts(["the plan, the other", "the plan\nthe other"], "plan the") == [
        "the plan\nthe other",
        "the plan, the other",
    ]


@pytest.mark.parametrize(
    ("query", "multiline", "inline"),
    [
        ("plan:\nthe", "plan:\nthe", "plan: the"),
        ("存\r\n在", "存\r\n在", "存在"),
        ("項目:\u2028説明", "項目:\u2028説明", "項目:説明"),
        ("  \uff30\uff2c\uff21\uff2e\uff1a\n\uff34\uff28\uff25 \n", "plan:\nthe", "plan: the"),
    ],
    ids=["punctuation", "cjk-crlf", "cjk-line-separator", "normalized-padded-query"],
)
def test_multiline_query_keeps_exact_and_phrase_priority(query: str, multiline: str, inline: str) -> None:
    # Newest first: a single-line variant must not outrank the query's own line break.
    assert rank_prompts([inline, multiline], query) == [multiline, inline]
    inline_phrase, multiline_phrase = f"before {inline} after", f"before {multiline} after"
    assert rank_prompts([inline_phrase, multiline_phrase], query) == [multiline_phrase, inline_phrase]


def test_a_spaced_match_ranks_below_a_substring() -> None:
    # Newest first: the spaced match is newer, yet the substring outranks it.
    prompts = ["xplan:thex", "a plan: the", "a plan:x the"]

    assert rank_prompts(prompts, "plan:the") == ["xplan:thex", "a plan: the"]
    # Equal to the query once layout whitespace is gone, a prompt is an exact match.
    assert rank_prompts(["xplan:thex", "plan: the"], "plan:the") == ["plan: the", "xplan:thex"]


def test_matching_is_nfkc_normalized_on_both_sides() -> None:
    composed = "café notes"
    decomposed = unicodedata.normalize("NFD", composed)

    assert rank_prompts([decomposed], "café") == [decomposed]
    assert rank_prompts([composed], unicodedata.normalize("NFD", "café")) == [composed]
    assert normalize_for_search(decomposed) == normalize_for_search(composed)


def test_full_width_letters_and_digits_match_ascii_queries() -> None:
    assert rank_prompts(["ＬＯＣ统计", "第１步"], "loc") == ["ＬＯＣ统计"]
    assert rank_prompts(["ＬＯＣ统计", "第１步"], "1") == ["第１步"]


def test_cjk_characters_are_words_of_their_own() -> None:
    # A contiguous CJK match is a whole word wherever it falls, so recency decides.
    prompts = ["恢复会话失败", "会话列表"]

    assert rank_prompts(prompts, "会话") == prompts
    assert rank_prompts(["会话ID丢失"], "id") == ["会话ID丢失"]
    assert rank_prompts(["会话ID丢失", "valid id"], "id") == ["会话ID丢失", "valid id"]


def test_cjk_words_apart_are_separate_terms() -> None:
    prompts = ["把数据落盘的调查结果写到文档", "检查工作目录"]

    assert rank_prompts(prompts, "落盘文档") == []
    assert rank_prompts(prompts, "落盘 文档") == ["把数据落盘的调查结果写到文档"]


def _preview(prompt: str, query: str, cells: int = 40) -> tuple[str, list[str]]:
    """Render a preview as its row would read, with ``[`` ``]`` around matched characters."""
    (match,) = PromptSearchIndex([prompt]).search(query, limit=10, cells=cells, head_chars=60).matches
    marked = {position for low, high in match.spans for position in range(low, high)}
    parts: list[str] = []
    previous_end = 0
    for start, end in match.windows:
        if start > previous_end or (start > 0 and not parts):
            parts.append("…")
        parts.append("".join(f"[{prompt[i]}]" if i in marked else prompt[i] for i in range(start, end)))
        previous_end = end
    if previous_end < len(prompt):
        parts.append("…")
    row = "".join(parts).replace("][", "")
    return row, [prompt[low:high] for low, high in match.spans]


def test_preview_keeps_the_head_when_the_match_fits_in_it() -> None:
    assert _preview("LOC for repo, local block", "loc") == ("[LOC] for repo, local block", ["LOC"])


def test_preview_moves_to_a_match_past_the_head_with_context_on_both_sides() -> None:
    row, spans = _preview("x" * 100 + " the session restore failed here", "restore")

    assert row == "…xxxxxxx the session [restore] failed here"
    assert cell_len(row.replace("[", "").replace("]", "")) == 40
    assert spans == ["restore"]


def test_preview_context_before_the_match_scales_with_the_row_width() -> None:
    prompt = "x" * 100 + " the session restore failed here " + "y" * 100

    narrow, _ = _preview(prompt, "restore", cells=24)
    wide, _ = _preview(prompt, "restore", cells=60)

    assert narrow.index("[restore]") < wide.index("[restore]")
    assert cell_len(narrow.replace("[", "").replace("]", "")) <= 24
    assert cell_len(wide.replace("[", "").replace("]", "")) <= 60


def test_preview_anchors_on_the_occurrence_that_earned_the_tier() -> None:
    # "loc" occurs mid-word early on, but the whole word further on ranks the entry.
    row, spans = _preview("block " + "x" * 60 + " LOC count", "loc")

    assert row.endswith("[LOC] count")
    assert spans == ["LOC"]


def test_preview_marks_every_shown_occurrence_of_a_term_in_any_case() -> None:
    # Repeats are marked at the tier the term earned: "the" inside "other" is not a whole word.
    assert _preview("The plan, the other fix and THE end", "the") == (
        "[The] plan, [the] other fix and [THE] end",
        ["The", "the", "THE"],
    )


def test_preview_repeats_do_not_move_the_windows() -> None:
    row, spans = _preview("alpha " + "x" * 200 + " alpha", "alpha")

    assert row.startswith("[alpha] ")
    assert spans == ["alpha"]


def test_preview_marks_a_prompt_equal_to_the_query_from_end_to_end() -> None:
    prompt = "explore the repo and report the architecture with the words to file"

    assert _preview(prompt, prompt, cells=80) == (f"[{prompt}]", [prompt])


def test_preview_marks_the_whitespace_inside_a_run_of_the_query() -> None:
    assert _preview("then restore  failed again", "restore failed")[1] == ["restore  failed"]


@pytest.mark.parametrize(
    ("multiline", "inline"),
    [("plan:\nthe", "plan: the"), ("存\r\n在", "存 在"), ("項目:\u2028説明", "項目: 説明")],
    ids=["punctuation", "cjk-crlf", "cjk-line-separator"],
)
def test_preview_of_multiline_query_marks_only_runs_with_the_line_break(multiline: str, inline: str) -> None:
    prompt = f"{inline} before {multiline} after {multiline}"

    assert _preview(prompt, multiline, cells=120)[1] == [multiline, multiline]
    assert _preview(multiline, multiline, cells=80)[1] == [multiline]


def test_preview_keeps_spaces_and_line_breaks_equivalent_between_words() -> None:
    assert _preview("restore failed, restore\nfailed", "restore\nfailed", cells=80)[1] == [
        "restore failed",
        "restore\nfailed",
    ]


def test_preview_without_a_run_of_the_query_marks_every_occurrence_of_each_term() -> None:
    assert _preview("fix the bug, then fix the other bug", "fix bug")[1] == ["fix", "bug", "fix", "bug"]


def test_preview_of_a_run_of_the_query_marks_only_its_runs() -> None:
    # Once the query is found as typed, its words elsewhere are not marked on their own.
    prompt = "explore the repo and report, the end, explore the repo again"

    assert _preview(prompt, "explore the repo", cells=80)[1] == ["explore the repo", "explore the repo"]


def test_preview_marks_a_spaced_match_whole() -> None:
    prompt = "The plan: the other fix goes to THE end"

    assert _preview(prompt, "plan:th", cells=80)[1] == ["plan: th"]
    assert _preview(prompt, "plan:the other", cells=80)[1] == ["plan: the other"]


def test_preview_keeps_a_run_whole_when_the_next_term_is_elsewhere() -> None:
    prompt = "The plan: the other fix goes to THE end, then the tests"

    assert _preview(prompt, "The plan: the other", cells=80)[1] == ["The plan: the other"]
    # Typing on changes only the last run: the earlier one stays whole and lone copies of its words stay unmarked.
    assert _preview(prompt, "The plan: the other g", cells=80)[1] == ["The plan: the other", "g"]
    assert _preview(prompt, "The plan: the other goes", cells=80)[1] == ["The plan: the other", "goes"]
    assert _preview(prompt, "The plan: the other fix", cells=80)[1] == ["The plan: the other fix"]


def test_preview_marks_each_of_the_fewest_runs_of_the_query() -> None:
    assert _preview("restore session ... failed", "restore session failed")[1] == ["restore session", "failed"]
    assert _preview("plan: the end, then failed", "plan:the failed")[1] == ["plan: the", "failed"]


def test_preview_centers_on_a_run_of_the_query_past_an_earlier_lone_word() -> None:
    row, spans = _preview("restore " + "x" * 200 + " the session restore failed here", "restore failed")

    assert row.endswith("the session [restore failed] here")
    assert spans == ["restore failed"]


def test_preview_marks_repeats_through_length_changing_normalization() -> None:
    assert _preview("Straße und STRASSE", "strasse")[1] == ["Straße", "STRASSE"]
    prompt = unicodedata.normalize("NFD", "café, café")
    assert _preview(prompt, "café")[1] == [unicodedata.normalize("NFD", "café")] * 2


def test_preview_joins_far_apart_terms_with_an_ellipsis() -> None:
    row, spans = _preview("alpha " + "x" * 200 + " omega end", "alpha omega")

    assert row.startswith("[alpha] ")
    assert "…" in row[len("[alpha]") :]
    assert row.endswith("[omega] end")
    assert spans == ["alpha", "omega"]
    assert cell_len(row.replace("[", "").replace("]", "")) <= 40


def test_preview_drops_trailing_terms_when_the_row_is_too_narrow() -> None:
    row, spans = _preview("alpha " + "x" * 200 + " omega " + "y" * 200 + " gamma", "alpha omega gamma", cells=30)

    assert spans == ["alpha", "omega"]
    assert cell_len(row.replace("[", "").replace("]", "")) <= 30


def test_preview_runs_on_past_the_row_so_a_highlighted_row_can_scroll() -> None:
    prompt = "x" * 100 + " restore " + "y" * 400 + " omega"
    (match,) = PromptSearchIndex([prompt]).search("restore", limit=1, cells=30, head_chars=60).matches
    (window,) = match.windows

    assert match.tail_end == window[1] + 60


def test_preview_tail_marks_a_term_cut_off_by_the_row_width() -> None:
    prompt = "alpha " + "x" * 40 + " omega"
    (match,) = PromptSearchIndex([prompt]).search("alpha omega", limit=1, cells=20, head_chars=60).matches

    assert all(high <= prompt.index("omega") for _low, high in match.windows)
    assert match.tail_end == len(prompt)
    assert [prompt[low:high] for low, high in match.spans] == ["alpha", "omega"]


def test_preview_counts_wide_characters_as_two_cells() -> None:
    row, spans = _preview("恢复" + "会" * 50 + "话题失败了很久", "失败")

    assert spans == ["失败"]
    assert cell_len(row.replace("[", "").replace("]", "")) <= 40


def test_preview_maps_length_changing_normalization_back_to_the_prompt() -> None:
    # "ß" folds to "ss", so normalized offsets no longer line up with the prompt.
    assert _preview("Straße und file", "strasse") == ("[Straße] und file", ["Straße"])


def test_preview_finds_and_marks_a_match_in_a_decomposed_prompt() -> None:
    # macOS and some input methods hand over NFD text: "é" arrives as "e" and a combining accent.
    prompt = unicodedata.normalize("NFD", "x" * 500 + " café notes")

    row, spans = _preview(prompt, "café")

    assert spans == [unicodedata.normalize("NFD", "café")]
    assert row.startswith("…")


def test_preview_marks_half_width_kana_with_voiced_marks() -> None:
    # The half-width voiced mark is not combining itself; NFKD turns it into one.
    assert _preview("ｶﾞｲﾄﾞ ガイド", "ガイド")[1] == ["ｶﾞｲﾄﾞ", "ガイド"]


def test_preview_maps_compatibility_jamo_to_the_syllable_source() -> None:
    # Compatibility jamo compose like canonical ones: the prompt keeps its own characters.
    assert _preview("ㄱㅏ notes", "가")[1] == ["ㄱㅏ"]


def test_preview_finds_and_marks_a_match_in_a_decomposed_hangul_prompt() -> None:
    prompt = unicodedata.normalize("NFD", "한국어 검색")

    assert _preview(prompt, "한국")[1] == [unicodedata.normalize("NFD", "한국")]


def test_preview_maps_offsets_when_expansion_and_composition_cancel_out() -> None:
    # "ß" grows to "ss" and "e" + accent shrinks to "é": same length, shifted offsets.
    prompt = "ß" + "e\u0301" + " tail"

    assert _preview(prompt, "é")[1] == ["e\u0301"]


def test_preview_without_a_query_is_the_head() -> None:
    assert _preview("a" * 80, "") == ("a" * 60 + "…", [])


def test_search_counts_every_match_beyond_the_limit() -> None:
    result = PromptSearchIndex([f"docs {n}" for n in range(30)] + ["fix"]).search(
        "docs", limit=5, cells=40, head_chars=60
    )

    assert result.total == 30
    assert [match.prompt for match in result.matches] == [f"docs {n}" for n in range(5)]


def test_a_search_stops_as_soon_as_it_is_told_to() -> None:
    checks: list[None] = []

    def should_stop() -> bool:
        checks.append(None)
        return len(checks) >= 3

    result = PromptSearchIndex([f"docs {n}" for n in range(50)]).search(
        "docs", limit=10, cells=40, head_chars=60, should_stop=should_stop
    )

    assert result == PromptSearchResult(matches=[], total=0)
    assert len(checks) == 3


def test_index_reuses_entries_of_a_previous_index() -> None:
    first = PromptSearchIndex(["alpha", "beta"])
    second = PromptSearchIndex(["gamma", "beta", "alpha"], previous=first)

    assert second._entries[1] is first._entries[1]
    assert second._entries[2] is first._entries[0]
    assert [match.prompt for match in second.search("", limit=3, cells=80, head_chars=60).matches] == [
        "gamma",
        "beta",
        "alpha",
    ]

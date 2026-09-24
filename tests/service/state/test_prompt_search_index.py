# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for SessionMeta.user_prompt_search_text — the prompt search index list_sessions derives per session."""

from __future__ import annotations

import json
from pathlib import Path

from chrys.kernel import Message
from chrys.service.state.store import (
    JsonFileStateStore,
)


async def test_list_sessions_user_prompt_search_text_covers_every_turn(tmp_path: Path) -> None:
    """User text from all turns lands in meta; synthetic history does not."""
    from chrys.foundation.models.history_markers import HistoryMarkerKind

    store = JsonFileStateStore(tmp_path)
    marker = Message("user", ["turn 1"])
    marker.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
    nudge = Message("user", ["continue"])
    nudge.additional_properties[HistoryMarkerKind.CONTINUATION_KEY] = True
    await store.save_session(
        "prompty",
        {
            "messages": [
                Message("user", ["first ask about kernels"]),
                Message("assistant", ["assistant reply text"]),
                marker,
                nudge,
                Message("user", ["second ask\nabout   scrollbars"]),
            ],
            "compressed_msgs": [],
        },
    )

    sessions = await store.list_sessions()

    text = sessions[0].user_prompt_search_text
    assert "first ask about kernels" in text
    # Internal whitespace collapses so a space-separated query matches a
    # phrase that wrapped across lines in the original prompt.
    assert "second ask about scrollbars" in text
    assert "assistant reply" not in text
    assert "turn 1" not in text
    assert "continue" not in text


async def test_list_sessions_user_prompt_search_text_keeps_head_and_tail(tmp_path: Path) -> None:
    """A pasted-log prompt keeps both edges: the dump's start AND the ask
    typed after it (middle elided, seam joined by a newline)."""
    store = JsonFileStateStore(tmp_path)
    prompt = "HEAD-ERROR-DUMP " + "x" * 3000 + " TAIL-ACTUAL-ASK"
    await store.save_session("pasty", {"messages": [Message("user", [prompt])], "compressed_msgs": []})

    sessions = await store.list_sessions()

    text = sessions[0].user_prompt_search_text
    assert "HEAD-ERROR-DUMP" in text
    assert "TAIL-ACTUAL-ASK" in text
    assert len(text) <= 2001  # 1000 head + newline seam + 1000 tail
    # The elided middle must not fabricate a match across the seam.
    assert "x" * 2001 not in text


async def test_list_sessions_user_prompt_search_text_total_cap_keeps_both_ends(tmp_path: Path) -> None:
    """Over-budget sessions keep their earliest AND latest prompts — a big
    early paste must not evict every later turn from the search index."""
    store = JsonFileStateStore(tmp_path)
    messages = [Message("user", [f"turn-{index}-marker " + "y" * 900]) for index in range(20)]
    await store.save_session("chatty", {"messages": messages, "compressed_msgs": []})

    sessions = await store.list_sessions()

    text = sessions[0].user_prompt_search_text
    assert len(text) <= 8000
    assert "turn-0-marker" in text
    assert "turn-19-marker" in text


async def test_list_sessions_user_prompt_search_text_exact_budget_keeps_all_turns(tmp_path: Path) -> None:
    """An index whose joined length is EXACTLY the total budget fits —
    separators only exist between excerpts, so charging one for the first
    excerpt overcounted by one and evicted a middle prompt from a
    perfectly fitting session (review-caught)."""
    store = JsonFileStateStore(tmp_path)
    # 9 excerpts of 888 chars join to 9*888 + 8 = 8000 chars exactly.
    messages = [Message("user", [f"turn-{index}-marker" + "y" * 875]) for index in range(9)]
    await store.save_session("snug", {"messages": messages, "compressed_msgs": []})

    sessions = await store.list_sessions()

    text = sessions[0].user_prompt_search_text
    assert len(text) == 8000
    for index in range(9):
        assert f"turn-{index}-marker" in text


async def test_list_sessions_user_prompt_search_text_repeated_final_prompt_survives_cap(tmp_path: Path) -> None:
    """A final-turn prompt that repeats a middle-turn text must stay
    searchable when the session overruns the total budget.  Dedup runs
    INSIDE the front/back selection — collapsing repeats onto their first
    copy up front would pin the text to the elided middle and evict it
    entirely (review-caught)."""
    store = JsonFileStateStore(tmp_path)
    repeat = "repeated ask about flux capacitors"
    messages = [Message("user", [f"turn-{index}-marker " + "y" * 900]) for index in range(10)]
    messages.append(Message("user", [repeat]))
    messages.extend(Message("user", [f"turn-{index}-marker " + "y" * 900]) for index in range(10, 20))
    messages.append(Message("user", [repeat]))
    await store.save_session("echoing", {"messages": messages, "compressed_msgs": []})

    sessions = await store.list_sessions()

    text = sessions[0].user_prompt_search_text
    assert len(text) <= 8000
    # The latest copy sits in the back window even though the first copy
    # was elided with the middle — and it is still indexed only once.
    assert text.count(repeat) == 1
    assert "turn-0-marker" in text
    assert "turn-19-marker" in text


async def test_list_sessions_user_prompt_search_text_front_repeat_skips_free(tmp_path: Path) -> None:
    """A final-turn repeat of a text already kept in the front window must
    skip for free during the back walk — neither breaking it nor spending
    its budget — so the latest unique prompt still gets indexed."""
    store = JsonFileStateStore(tmp_path)
    repeat = "opening ask typed again at the end"
    messages = [Message("user", [repeat])]
    messages.extend(Message("user", [f"turn-{index}-marker " + "y" * 900]) for index in range(20))
    messages.append(Message("user", [repeat]))
    await store.save_session("bookended", {"messages": messages, "compressed_msgs": []})

    sessions = await store.list_sessions()

    text = sessions[0].user_prompt_search_text
    assert len(text) <= 8000
    assert text.count(repeat) == 1
    assert "turn-19-marker" in text


async def test_list_sessions_user_prompt_search_text_reads_legacy_string_contents(tmp_path: Path) -> None:
    """Legacy sessions serialize contents as plain strings — those prompts
    must be searchable too (the title fallback only covers the first one).
    Mirrors replay's rule: a "Content(type=" repr blob is not text."""
    store = JsonFileStateStore(tmp_path)
    await store.save_session("legacy", {"messages": [], "compressed_msgs": []})
    session_file = store.session_dir("legacy") / "session.json"
    envelope = json.loads(session_file.read_text(encoding="utf-8"))
    envelope["state"]["messages"] = [
        {
            "type": "message",
            "role": "user",
            "contents": ["first ask about endianness"],
            "message_id": "m1",
            "additional_properties": {},
        },
        {
            "type": "message",
            "role": "user",
            "contents": ["later legacy\nask   about semaphores", "Content(type=whatever repr)"],
            "message_id": "m2",
            "additional_properties": {},
        },
    ]
    session_file.write_text(json.dumps(envelope), encoding="utf-8")

    sessions = await store.list_sessions()

    text = sessions[0].user_prompt_search_text
    assert "first ask about endianness" in text
    assert "later legacy ask about semaphores" in text
    assert "Content(type=" not in text


def test_collapsed_prompt_excerpt_matches_naive_collapse() -> None:
    """The fixed-block collapse scan must be output-identical to capping
    a full ``" ".join(text.split())`` — including when a 4096-char block
    boundary falls mid-word or mid-whitespace-run, and when
    whitespace-dominated input makes each block yield almost nothing."""
    edge = 1000  # scan blocks are a fixed 4096 chars, independent of edge

    def naive(text: str) -> str:
        collapsed = " ".join(text.split())
        if len(collapsed) <= 2 * edge:
            return collapsed
        return f"{collapsed[:edge]}\n{collapsed[-edge:]}"

    cases = [
        "",
        "hi there",
        "word " * 2000,  # dense short words: one block already fills the cap
        "x" * 5000,  # single mega-word longer than both caps, glued across blocks
        "a" * 3990 + "b" * 30 + " " + "tail words " * 200,  # word straddles the first block boundary
        ("w " * 1995) + "\t\n  " + ("v " * 1995),  # whitespace run straddles a block boundary
        " " * 6000 + "short tail after huge indent",  # all-whitespace leading blocks yield nothing
        "lead words " + " " * 6000 + "z" * 3000,  # whitespace-dominated middle
        "w" * 1500 + " " * 5000 + "z" * 1500,  # tail-side blocks open on whitespace
        "p" * 999 + " " + "q" * 1000,  # collapsed length exactly 2000: uncapped
        "p" * 1000 + " " + "q" * 1000,  # collapsed length 2001: capped
        "word  " * 640,  # raw 3840 shorter than one block yet collapsed 3199 > cap:
        # the tail scan must clamp its first block to the string start, not
        # wrap through a negative index into a tiny suffix
        # (real-session-caught against the earlier growing-window version)
        "　　混合  空白\tacross\nlines　" * 400,  # unicode whitespace
        # ~2MB whitespace-dominated with period 509 (does not divide the
        # 4096 scan block): boundaries drift through mid-run, mid-word and
        # word-edge phases across thousands of stitches.
        (" " * 507 + "wx") * 4000,
        # Words straddling exact block boundaries in an under-cap text:
        # the head path must glue the split word, not space it apart.
        " " * 4090 + "straddleword" + " " * 4090 + "tail",
    ]
    for text in cases:
        assert JsonFileStateStore._collapsed_prompt_excerpt(text, edge) == naive(text), repr(text[:60])


def test_message_prompt_excerpt_is_memory_bounded() -> None:
    """A multi-megabyte pasted log must not be word-split whole —
    ``str.split()`` transiently costs ~14x the text size; one meta scan
    parses sessions serially in its to_thread worker, but independent
    callers can overlap and multiply the peak.  Fixed-block collapse keeps
    the peak near the excerpt caps for dense AND whitespace-dominated
    input (a growing window slice would copy O(input) raw text on the
    latter before yielding enough collapsed chars)."""
    import tracemalloc

    cases = [
        ("err " * 500_000, "err", 1_000_000),  # 2MB dense short words: split() peaks ~28MB
        ((" " * 499 + "z") * 4000, "z", 400_000),  # 2MB, 0.2% density: doubling windows peaked ~512KB
    ]
    for text, needle, bound in cases:
        message = {
            "type": "message",
            "role": "user",
            "contents": [{"type": "text", "text": text}],
            "additional_properties": {},
        }
        tracemalloc.start()
        excerpt = JsonFileStateStore._message_prompt_excerpt(message)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert excerpt is not None
        assert needle in excerpt
        assert len(excerpt) <= 2001
        assert peak < bound, f"peak {peak} for needle {needle!r}"


async def test_list_sessions_user_prompt_search_text_includes_compacted_turns(tmp_path: Path) -> None:
    """Prompts folded into compressed_msgs blocks stay searchable, and a
    message present in both a block and the live list counts once."""
    store = JsonFileStateStore(tmp_path)
    await store.save_session("compacted-prompts", {"messages": [], "compressed_msgs": []})
    session_file = store.session_dir("compacted-prompts") / "session.json"
    envelope = json.loads(session_file.read_text(encoding="utf-8"))

    def user_msg(message_id: str, text: str) -> dict:
        return {
            "type": "message",
            "role": "user",
            "contents": [{"type": "text", "text": text}],
            "message_id": message_id,
            "additional_properties": {},
        }

    envelope["state"]["messages"] = [user_msg("m2", "live twin popcount ask")]
    envelope["state"]["compressed_msgs"] = [
        {
            "compressed_context_id": "c1",
            "messages": [
                user_msg("m1", "folded ask about quaternions"),
                user_msg("m2", "live twin popcount ask"),
            ],
            "summary_text": "…",
            "marker_id": "turn_1",
            "turn_range": [1, 2],
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    ]
    session_file.write_text(json.dumps(envelope), encoding="utf-8")

    sessions = await store.list_sessions()

    text = sessions[0].user_prompt_search_text
    assert "quaternions" in text
    assert text.count("popcount") == 1


async def test_list_sessions_user_prompt_search_text_id_reuse_keeps_both_prompts(tmp_path: Path) -> None:
    """message_id restarts across runs, so a folded and a live prompt can
    share an id while being DIFFERENT messages — both must stay searchable
    (dedup is by indexed text, never by id).  A synthetic marker squatting
    on a live prompt's id must not suppress it either."""
    from chrys.foundation.models.history_markers import HistoryMarkerKind

    store = JsonFileStateStore(tmp_path)
    await store.save_session("id-reuse", {"messages": [], "compressed_msgs": []})
    session_file = store.session_dir("id-reuse") / "session.json"
    envelope = json.loads(session_file.read_text(encoding="utf-8"))

    def user_msg(message_id: str, text: str, *, kind: str = "") -> dict:
        props: dict = {HistoryMarkerKind.KEY: kind} if kind else {}
        return {
            "type": "message",
            "role": "user",
            "contents": [{"type": "text", "text": text}],
            "message_id": message_id,
            "additional_properties": props,
        }

    envelope["state"]["messages"] = [
        user_msg("msg_1", "later ask about heisenbugs"),
        user_msg("msg_2", "final ask about ringbuffers"),
    ]
    envelope["state"]["compressed_msgs"] = [
        {
            "compressed_context_id": "c1",
            "messages": [
                user_msg("msg_1", "early ask about monoids"),
                user_msg("msg_2", "turn 2", kind=HistoryMarkerKind.TURN),
            ],
            "summary_text": "…",
            "marker_id": "turn_1",
            "turn_range": [1, 2],
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    ]
    session_file.write_text(json.dumps(envelope), encoding="utf-8")

    sessions = await store.list_sessions()

    text = sessions[0].user_prompt_search_text
    assert "monoids" in text
    assert "heisenbugs" in text
    assert "ringbuffers" in text
    assert "turn 2" not in text

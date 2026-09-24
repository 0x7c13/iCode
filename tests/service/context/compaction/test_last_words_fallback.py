# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The reconstruction fallback: prompt composition, option allowlist, output budgets and the admission ladder."""

from __future__ import annotations

import pytest

from chrys.kernel import Content, Message
from chrys.service.context.compaction.last_words import (
    _BASE_GUIDANCE,
    _FORMAT_CONTRACT,
    _SUPPLEMENT_LABEL,
    LastWordsGenerationError,
    LastWordsGenerator,
    LastWordsSpendBudgetExceeded,
)
from chrys.service.profiles.agents.schema import DEFAULT_LAST_WORDS_MAX_OUTPUT_TOKENS
from chrys.service.profiles.models.schema import ModelProfile
from tests.service.context.compaction._last_words_helpers import (
    FailingFallbackClient,
    FakeCompleter,
    FallbackClient,
    generate,
    long_structured_note,
    make_generator,
    structured_note,
    user,
)

pytestmark = pytest.mark.usefixtures("no_note_floor")


@pytest.mark.parametrize("template", ["", " \n\t", "TEMPLATE TEXT"])
async def test_fallback_always_has_contract_and_base_with_optional_labeled_supplement(tmp_path, template):
    from chrys.service.context.compaction.last_words import (
        _MIN_NOTE_TOKENS,
        _NO_TOOLS_FINAL,
        _NO_TOOLS_OPENER,
        _NO_TOOLS_REMINDER,
    )

    gen = make_generator(tmp_path, template=template)
    fallback = FallbackClient(structured_note())
    gen._client = fallback  # type: ignore[assignment]

    await generate(gen, user_request="do X", previous_last_words=None, dropped_messages=[])

    instruction = fallback.messages[0][0].text
    opener = _NO_TOOLS_OPENER.format(min_note_tokens=_MIN_NOTE_TOKENS, max_output_tokens=20_000)
    final = _NO_TOOLS_FINAL.format(min_note_tokens=_MIN_NOTE_TOKENS, max_output_tokens=20_000)
    assert instruction.index(opener) < instruction.index("Everything in this conversation")
    assert instruction.index("<previous_progress_note>") < instruction.index(_FORMAT_CONTRACT)
    assert instruction.index(_FORMAT_CONTRACT) < instruction.index(_BASE_GUIDANCE)
    if template.strip():
        assert instruction.index(_BASE_GUIDANCE) < instruction.index(_SUPPLEMENT_LABEL)
        assert instruction.index(_SUPPLEMENT_LABEL) < instruction.index(template)
        assert instruction.index(template) < instruction.index(_NO_TOOLS_REMINDER)
    else:
        assert _SUPPLEMENT_LABEL not in instruction
        assert instruction.index(_BASE_GUIDANCE) < instruction.index(_NO_TOOLS_REMINDER)
    assert instruction.index(_NO_TOOLS_REMINDER) < instruction.index(final)


async def test_fallback_prompt_states_length_contract(tmp_path):
    """The reconstruction prompt states the 500-token minimum and the output cap.

    The completer instruction carries this contract in its no-tools wrapper;
    the fallback prompt must state it too, or the base guidance's "keep the
    note tight" direction would steer compliant models under the note floor."""
    from chrys.service.context.compaction.last_words import LastWordsGenerator
    from chrys.service.profiles.models.resolver import default_profile

    captured: dict = {}

    class _CapturingClient:
        async def get_response(self, messages, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            captured["user_prompt"] = messages[1].text

            class _Response:
                usage_details = None
                raw_text = structured_note()

            return _Response()

    gen = LastWordsGenerator(profile=default_profile(), log_dir=tmp_path)
    gen._client = _CapturingClient()  # type: ignore[assignment]

    await generate(gen, user_request="do X", previous_last_words=None, dropped_messages=[])

    prompt = captured["user_prompt"]
    assert "at least 500 tokens" in prompt
    assert "20000-token output cap" in prompt  # DEFAULT_LAST_WORDS_MAX_OUTPUT_TOKENS


@pytest.mark.parametrize(
    (
        "profile",
        "generator_kwargs",
        "expected_side_call_max",
        "expected_stated_cap",
        "expected_wire_max",
        "expected_thinking",
    ),
    [
        pytest.param(
            ModelProfile(id="t", name="t", model_id="deepseek-chat", max_output_tokens=8192),
            {},
            8192,
            "8192-token output cap",
            8192,
            None,
            id="model_output_cap_clamps_both_note_paths",
        ),
        pytest.param(
            ModelProfile(
                id="t",
                name="t",
                provider="anthropic",
                model_id="claude-test",
                max_output_tokens=0,  # unknown cap — this case targets the unclamped math
                chat_options='{"thinking": {"type": "enabled", "budget_tokens": 16000}}',
            ),
            {"max_output_tokens": 12000},
            12000 + 16000,
            "12000-token output cap",
            12000 + 16000,
            {"type": "enabled", "budget_tokens": 16000},
            id="thinking_budget_added_to_wire_max_tokens",
        ),
        pytest.param(
            ModelProfile(
                id="t",
                name="t",
                model_id="deepseek-chat",
                max_output_tokens=8192,
                chat_options='{"max_tokens": 20000}',
            ),
            {},
            8192,
            None,
            8192,
            None,
            id="profile_max_tokens_cannot_override_note_call_clamp",
        ),
    ],
)
async def test_note_call_output_budget_is_clamped_on_both_paths(
    tmp_path,
    profile: ModelProfile,
    generator_kwargs: dict,
    expected_side_call_max: int,
    expected_stated_cap: str | None,
    expected_wire_max: int,
    expected_thinking: dict | None,
) -> None:
    """The wire ``max_tokens`` and the stated note budget both respect the model cap.

    DeepSeek rejects ``max_tokens`` above 8192 outright, on the side call and on
    the reconstruction fallback alike, and a profile ``chat_options`` value must
    not undo the computed clamp.  Anthropic goes the other way: extended
    thinking spends from ``max_tokens``, so the wire value adds the thinking
    budget on top while the instruction keeps stating only the visible-note
    share.
    """
    gen = LastWordsGenerator(profile=profile, log_dir=tmp_path, **generator_kwargs)
    completer = FakeCompleter([structured_note()])
    captured: dict = {}

    class _CapturingClient:
        async def get_response(self, messages, *_args, **kwargs):  # type: ignore[no-untyped-def]
            captured["options"] = kwargs.get("options")

            class _Response:
                usage_details = None
                raw_text = structured_note()

            return _Response()

    gen._client = _CapturingClient()  # type: ignore[assignment]

    await generate(
        gen,
        user_request="do X",
        previous_last_words=None,
        dropped_messages=[],
        completer=completer,
    )
    # Side call: clamped wire value, clamped stated budget.
    call = completer.calls[0]
    assert call["max_output_tokens"] == expected_side_call_max
    if expected_stated_cap is not None:
        assert expected_stated_cap in call["instruction"]

    # Fallback: same clamp via the reconstruction client options.
    await generate(gen, user_request="do X", previous_last_words=None, dropped_messages=[])
    assert captured["options"]["max_tokens"] == expected_wire_max
    if expected_thinking is not None:
        # The thinking config itself rides through to the fallback call untouched.
        assert captured["options"]["thinking"] == expected_thinking


def test_output_budgets_math(tmp_path):
    """Budget corner cases: disabled thinking, cap-only, thinking re-clamped to cap."""
    from chrys.service.context.compaction.last_words import LastWordsGenerator
    from chrys.service.profiles.models.schema import ModelProfile

    def budgets(*, cap: int = 0, chat_options: str = "", configured: int = 12000) -> tuple[int, int]:
        profile = ModelProfile(id="t", name="t", max_output_tokens=cap, chat_options=chat_options)
        return LastWordsGenerator(profile=profile, log_dir=tmp_path, max_output_tokens=configured)._output_budgets()

    # No cap, no thinking: both values are the configured budget.
    assert budgets() == (12000, 12000)
    # Disabled thinking is ignored.
    assert budgets(chat_options='{"thinking": {"type": "disabled", "budget_tokens": 16000}}') == (12000, 12000)
    # Cap alone clamps both.
    assert budgets(cap=8192) == (8192, 8192)
    # Thinking budget rides on top of the wire value only.
    assert budgets(chat_options='{"thinking": {"type": "enabled", "budget_tokens": 16000}}') == (12000, 28000)
    # Cap re-clamps the combined wire value; the stated note share shrinks.
    assert budgets(cap=20000, chat_options='{"thinking": {"type": "enabled", "budget_tokens": 16000}}') == (
        4000,
        20000,
    )
    # No explicit cap: a user-set profile max_tokens is the ceiling instead
    # (provider-validated by every live call).
    assert budgets(chat_options='{"max_tokens": 8000}') == (8000, 8000)
    # A generous profile max_tokens does not inflate the note budget.
    assert budgets(chat_options='{"max_tokens": 20000}') == (12000, 12000)
    # The explicit field wins over the chat-options fallback.
    assert budgets(cap=8192, chat_options='{"max_tokens": 20000}') == (8192, 8192)
    # Non-integer / non-positive max_tokens values are ignored, not crashes.
    assert budgets(chat_options='{"max_tokens": "8000"}') == (12000, 12000)
    assert budgets(chat_options='{"max_tokens": 0}') == (12000, 12000)
    # Provider-native output-cap spellings serve as the ceiling too
    # (programmatic profiles bypass the loader migration).
    assert budgets(chat_options='{"max_output_tokens": 4096}') == (4096, 4096)
    assert budgets(chat_options='{"max_completion_tokens": 4096}') == (4096, 4096)
    # A bool budget_tokens is malformed, not a 1-token thinking budget.
    assert budgets(chat_options='{"thinking": {"type": "enabled", "budget_tokens": true}}') == (12000, 12000)


def test_output_budgets_warns_when_thinking_squeezes_note_below_prompt_minimum(tmp_path, caplog):
    """A thinking budget just under the cap silently starved the note before.

    The prompt asks for at least 500 tokens of note content; when the
    clamped note share falls below that, every attempt is truncated under
    the note floor and retried — warn instead of failing silently.  When
    the thinking budget meets/exceeds the cap the provider rejects the call
    outright, which has its own (mutually exclusive) warning.
    """
    from chrys.service.context.compaction.last_words import LastWordsGenerator
    from chrys.service.profiles.models.schema import ModelProfile

    def budgets(chat_options: str) -> tuple[int, int]:
        profile = ModelProfile(id="t", name="t", max_output_tokens=8192, chat_options=chat_options)
        return LastWordsGenerator(profile=profile, log_dir=tmp_path, max_output_tokens=12000)._output_budgets()

    with caplog.at_level("WARNING"):
        assert budgets('{"thinking": {"type": "enabled", "budget_tokens": 8100}}') == (92, 8192)
    assert "below the 500-token minimum" in caplog.text
    assert "meets or exceeds" not in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        budgets('{"thinking": {"type": "enabled", "budget_tokens": 8192}}')
    assert "meets or exceeds" in caplog.text
    assert "below the 500-token minimum" not in caplog.text


async def test_fallback_uses_three_scoped_blocks_and_interleaves_followups(tmp_path):
    from chrys.service.context.compaction.last_words import LastWordsGenerator
    from chrys.service.profiles.models.resolver import default_profile

    captured: dict = {}

    class _Client:
        async def get_response(self, messages, *_args, **kwargs):  # type: ignore[no-untyped-def]
            captured["prompt"] = messages[1].text
            captured["options"] = kwargs["options"]

            class _Response:
                usage_details = None
                raw_text = structured_note()

            return _Response()

    gen = LastWordsGenerator(profile=default_profile(), log_dir=tmp_path)
    gen._client = _Client()  # type: ignore[assignment]
    await generate(
        gen,
        user_request="fix it",
        previous_last_words="previous",
        dropped_messages=[Message("assistant", ["working"])],
        followup_texts=["also test it"],
    )

    prompt = captured["prompt"]
    assert all(f"<{tag}>" in prompt for tag in ("user_request", "previous_progress_note", "work_done_since"))
    assert "<prior_conversation>" not in prompt
    assert "<injected_followups>" not in prompt
    assert "- user said: also test it" in prompt


async def test_fallback_option_allowlist_drops_all_input_shaping_fields(tmp_path):
    from chrys.service.context.compaction.last_words import LastWordsGenerator
    from chrys.service.profiles.models.schema import ModelProfile

    captured: dict = {}

    class _Client:
        async def get_response(self, _messages, *_args, **kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs["options"])

            class _Response:
                usage_details = None
                raw_text = structured_note()

            return _Response()

    profile = ModelProfile(
        id="bounded",
        name="bounded",
        model_id="model",
        chat_options=(
            '{"model":"safe","temperature":0.2,"reasoning_effort":"low",'
            '"instructions":"huge","tools":[{"type":"function"}],"tool_choice":"required",'
            '"response_format":{"type":"json_schema"},"schema":{"huge":true},"store":true,'
            '"previous_response_id":"resp","conversation_id":"conv","unknown_input":"drop"}'
        ),
    )
    gen = LastWordsGenerator(profile=profile, log_dir=tmp_path)
    gen._client = _Client()  # type: ignore[assignment]
    await generate(gen, user_request="do X", previous_last_words=None, dropped_messages=[])

    assert captured == {
        "model": "safe",
        "temperature": 0.2,
        "reasoning_effort": "low",
        "max_tokens": DEFAULT_LAST_WORDS_MAX_OUTPUT_TOKENS,
    }


async def test_fallback_caps_supplement_and_middle_truncates_previous_note(tmp_path):
    from chrys.service.context.compaction.last_words import (
        _FALLBACK_PREV_NOTE_MAX_CHARS,
        _FALLBACK_TEMPLATE_MAX_CHARS,
        LastWordsGenerator,
    )
    from chrys.service.profiles.models.resolver import default_profile

    captured: dict = {}

    class _Client:
        async def get_response(self, messages, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            captured["system"] = messages[0].text
            captured["prompt"] = messages[1].text

            class _Response:
                usage_details = None
                raw_text = structured_note()

            return _Response()

    template = "T" * (_FALLBACK_TEMPLATE_MAX_CHARS * 20)
    previous = "START" + "p" * (_FALLBACK_PREV_NOTE_MAX_CHARS * 4) + "FRESHEST"
    gen = LastWordsGenerator(profile=default_profile(), template=template, log_dir=tmp_path)
    gen._client = _Client()  # type: ignore[assignment]
    await generate(gen, user_request="do X", previous_last_words=previous, dropped_messages=[])

    system_instruction = captured["system"]
    supplement = system_instruction.split(f"{_SUPPLEMENT_LABEL}\n", 1)[1].split("\n\nReminder:", 1)[0]
    assert _FORMAT_CONTRACT in system_instruction
    assert _BASE_GUIDANCE in system_instruction
    assert len(supplement) <= _FALLBACK_TEMPLATE_MAX_CHARS
    assert "template truncated" in supplement
    assert "START" in captured["prompt"]
    assert "FRESHEST" in captured["prompt"]
    assert "older note content truncated" in captured["prompt"]


def test_fallback_admission_counter_is_utf8_byte_conservative() -> None:
    from chrys.service.context.compaction.last_words import _fallback_admission_tokens

    ascii_request = (Message("system", ["a"]), Message("user", ["plain"] * 4))
    adversarial = (Message("system", ["🙂"]), Message("user", ['é\\"🙂'] * 4))
    assert _fallback_admission_tokens(adversarial, output_reserve=0) > _fallback_admission_tokens(
        ascii_request,
        output_reserve=0,
    )


async def test_fallback_shrinks_timeline_without_truncating_fixed_guidance(tmp_path):
    from chrys.service.context.compaction.last_words import (
        _FALLBACK_TEMPLATE_COMPACT_CHARS,
        LastWordsGenerator,
        _fallback_admission_tokens,
    )
    from chrys.service.profiles.models.schema import ModelProfile

    captured: dict = {}

    class _Client:
        async def get_response(self, messages, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            captured["messages"] = tuple(messages)

            class _Response:
                usage_details = None
                raw_text = structured_note()

            return _Response()

    profile = ModelProfile(
        id="small",
        name="small",
        model_id="small",
        max_context_tokens=12_000,
        max_output_tokens=500,
    )
    gen = LastWordsGenerator(profile=profile, template="T" * 100_000, log_dir=tmp_path)
    gen._client = _Client()  # type: ignore[assignment]
    previous = "START" + "p" * 100_000 + "LATEST"
    await generate(
        gen,
        user_request="do X",
        previous_last_words=previous,
        dropped_messages=[Message("assistant", ["work " * 20_000])],
    )

    messages = captured["messages"]
    assert _FORMAT_CONTRACT in messages[0].text
    assert _BASE_GUIDANCE in messages[0].text
    assert _SUPPLEMENT_LABEL in messages[0].text
    assert "template truncated" in messages[0].text
    supplement = messages[0].text.split(f"{_SUPPLEMENT_LABEL}\n", 1)[1].split("\n\nReminder:", 1)[0]
    assert len(supplement) <= _FALLBACK_TEMPLATE_COMPACT_CHARS
    assert _fallback_admission_tokens(messages, output_reserve=500) <= profile.max_context_tokens


async def test_fallback_format_correction_is_readmitted_and_shrunk(tmp_path, monkeypatch):
    from chrys.service.context.compaction.last_words import (
        LastWordsGenerator,
        _fallback_admission_tokens,
    )
    from chrys.service.profiles.models.schema import ModelProfile

    monkeypatch.setattr(LastWordsGenerator, "_BACKOFF_SCHEDULE", (0,))
    invalid = "## Task\nDo it\n\n## Progress\n" + "work " * 100
    valid = long_structured_note()

    class _Client:
        def __init__(self) -> None:
            self.messages: list[tuple[Message, ...]] = []

        async def get_response(self, messages, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            self.messages.append(tuple(messages))

            class _Response:
                usage_details = None
                raw_text = invalid if len(self.messages) == 1 else valid

            return _Response()

    profile = ModelProfile(
        id="correction-budget",
        name="correction-budget",
        model_id="correction-budget",
        # Calibrated between the first attempt's admission estimate and the
        # correction attempt's (larger, rejection-notice-bearing) one; growing
        # the shared guidance/contract text shifts both and moves this line.
        max_context_tokens=9_560,
        max_output_tokens=500,
    )
    client = _Client()
    gen = LastWordsGenerator(profile=profile, log_dir=tmp_path)
    gen._client = client  # type: ignore[assignment]

    out = await generate(
        gen,
        user_request="do X",
        previous_last_words=None,
        dropped_messages=[Message("assistant", ["work " * 9_000])],
    )

    assert out == valid
    assert len(client.messages) == 2
    first, corrected = client.messages
    assert "Your previous note was rejected:" not in first[0].text
    assert 'Your previous note was rejected: missing required heading "## Next".' in corrected[0].text
    assert len(corrected[1].text) < len(first[1].text)
    assert _FORMAT_CONTRACT in corrected[0].text
    assert _BASE_GUIDANCE in corrected[0].text
    assert _fallback_admission_tokens(corrected, output_reserve=500) <= profile.max_context_tokens


async def test_fallback_tiny_window_sends_final_candidate_before_failing(tmp_path):
    """Local admission never dead-ends Phase 4: the terminal candidate is sent
    even when the byte-conservative estimate says it cannot fit, and only a
    genuine provider context rejection ends the shrink ladder."""
    from chrys.service.context.compaction.last_words import LastWordsGenerator
    from chrys.service.profiles.models.schema import ModelProfile

    class _Client:
        calls = 0

        async def get_response(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            raise RuntimeError("prompt is too long for this model")

    profile = ModelProfile(id="tiny", name="tiny", model_id="tiny", max_context_tokens=32, max_output_tokens=32)
    client = _Client()
    gen = LastWordsGenerator(profile=profile, log_dir=tmp_path)
    gen._client = client  # type: ignore[assignment]
    with pytest.raises(LastWordsGenerationError) as excinfo:
        await generate(gen, user_request="do X", previous_last_words=None, dropped_messages=[])
    assert client.calls == 1
    assert isinstance(excinfo.value.__cause__, RuntimeError)


async def test_fallback_small_window_final_candidate_is_provider_authoritative(tmp_path):
    """A realistic small-context profile succeeds via the terminal candidate even
    though every candidate exceeds the byte-conservative admission estimate."""
    from chrys.service.context.compaction.last_words import (
        _MIN_NOTE_TOKENS,
        LastWordsGenerator,
        _fallback_admission_tokens,
    )
    from chrys.service.profiles.models.schema import ModelProfile

    captured: dict = {}

    class _Client:
        def __init__(self) -> None:
            self.calls = 0

        async def get_response(self, messages, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            captured["messages"] = tuple(messages)
            captured["options"] = dict(_kwargs.get("options") or {})

            class _Response:
                usage_details = None
                raw_text = structured_note()

            return _Response()

    profile = ModelProfile(
        id="small-window",
        name="small-window",
        model_id="small-window",
        max_context_tokens=9_000,
        max_output_tokens=8_192,
    )
    client = _Client()
    gen = LastWordsGenerator(profile=profile, log_dir=tmp_path)
    gen._client = client  # type: ignore[assignment]

    out = await generate(
        gen,
        user_request="do X",
        previous_last_words=None,
        dropped_messages=[Message("assistant", ["work " * 5_000])],
    )

    assert out == structured_note()
    assert client.calls == 1
    messages = captured["messages"]
    assert _fallback_admission_tokens(messages, output_reserve=8_192) > profile.max_context_tokens
    # The output reserve is exact, so it must be clamped to the room the
    # conservative input estimate leaves — a provider enforcing
    # input + max_tokens <= context would reject the full 8_192 reserve.
    sent_max_tokens = captured["options"]["max_tokens"]
    input_estimate = _fallback_admission_tokens(messages, output_reserve=0)
    assert sent_max_tokens >= _MIN_NOTE_TOKENS
    assert input_estimate + sent_max_tokens <= profile.max_context_tokens
    # The prompt's length directive must advertise the clamped budget, not the
    # original one — otherwise the model writes past max_tokens and the note
    # truncates mid-section.
    assert f"{sent_max_tokens}-token" in messages[1].text
    assert "20000-token" not in messages[1].text


async def test_fallback_bypass_never_raises_max_tokens_above_model_cap(tmp_path):
    """The note floor must not push the bypass request above the configured
    output ceiling — providers enforcing the hard cap would reject it."""
    from chrys.service.context.compaction.last_words import LastWordsGenerator
    from chrys.service.profiles.models.schema import ModelProfile

    captured: dict = {}

    class _Client:
        def __init__(self) -> None:
            self.calls = 0

        async def get_response(self, messages, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            captured["options"] = dict(_kwargs.get("options") or {})

            class _Response:
                usage_details = None
                raw_text = structured_note()

            return _Response()

    profile = ModelProfile(
        id="tiny-cap",
        name="tiny-cap",
        model_id="tiny-cap",
        max_context_tokens=2_000,
        max_output_tokens=256,
    )
    client = _Client()
    gen = LastWordsGenerator(profile=profile, log_dir=tmp_path, max_output_tokens=256)
    gen._client = client  # type: ignore[assignment]

    out = await generate(
        gen,
        user_request="do X",
        previous_last_words=None,
        dropped_messages=[Message("assistant", ["work " * 2_000])],
    )

    assert out == structured_note()
    assert client.calls == 1
    assert captured["options"]["max_tokens"] == 256


async def test_fallback_short_note_acceptance_still_canonicalizes(tmp_path, monkeypatch):
    """A structurally valid note accepted below the length floor must still be
    canonicalized (heading levels normalized, sections ordered)."""
    from chrys.service.context.compaction.last_words import LastWordsGenerator

    monkeypatch.setattr(LastWordsGenerator, "_MAX_CORRECTIVE_RETRIES", 0)
    short_relaxed = "# Task\nDo it\n\n### Progress\nStarted\n\n## Next\nFinish"

    class _Client:
        async def get_response(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            class _Response:
                usage_details = None
                raw_text = short_relaxed

            return _Response()

    gen = make_generator(tmp_path)
    gen._client = _Client()  # type: ignore[assignment]

    out = await generate(gen, user_request="do X", previous_last_words=None, dropped_messages=[])

    assert out == "## Task\nDo it\n\n## Progress\nStarted\n\n## Next\nFinish"


async def test_provider_context_rejection_advances_fallback_shrink_sequence(tmp_path):
    from chrys.service.context.compaction.last_words import LastWordsGenerator
    from chrys.service.profiles.models.resolver import default_profile

    prompt_lengths: list[int] = []

    class _Client:
        async def get_response(self, messages, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            prompt_lengths.append(len(messages[1].text))
            if len(prompt_lengths) == 1:
                raise RuntimeError("maximum context length exceeded")

            class _Response:
                usage_details = None
                raw_text = structured_note()

            return _Response()

    gen = LastWordsGenerator(profile=default_profile(), log_dir=tmp_path)
    gen._client = _Client()  # type: ignore[assignment]
    await generate(
        gen,
        user_request="do X",
        previous_last_words="previous",
        dropped_messages=[Message("assistant", [f"work-{index} " * 1_000]) for index in range(20)],
    )
    assert len(prompt_lengths) == 2
    assert prompt_lengths[1] < prompt_lengths[0]


async def test_provider_context_rejection_traverses_shrink_ladder_at_zero_transient_budget(tmp_path):
    from chrys.service.context.compaction.last_words import LastWordsGenerator
    from chrys.service.profiles.models.resolver import default_profile

    class _Client:
        calls = 0

        async def get_response(self, _messages, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            raise RuntimeError("maximum context length exceeded")

    client = _Client()
    gen = LastWordsGenerator(profile=default_profile(), log_dir=tmp_path, max_transient_retries=0)
    gen._client = client  # type: ignore[assignment]

    with pytest.raises(LastWordsGenerationError) as exc_info:
        await generate(gen, user_request="do X", previous_last_words=None, dropped_messages=[])

    assert client.calls == 5
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "maximum context length exceeded" in str(exc_info.value.__cause__)


@pytest.mark.parametrize("use_completer", [False, True], ids=["fallback", "completer"])
async def test_mid_round_spend_exhaustion_aborts_before_retry(tmp_path, monkeypatch, use_completer: bool) -> None:
    """A spend gate that refuses the retry charge aborts the round before the second provider call."""
    charges: list[int] = []

    def spend(estimated_tokens: int) -> bool:
        charges.append(estimated_tokens)
        return len(charges) == 1

    completer = FakeCompleter([ConnectionError("retry me"), "must not run"]) if use_completer else None
    client = None
    if use_completer:
        monkeypatch.setattr(LastWordsGenerator, "_BACKOFF_SCHEDULE", (0,))
        gen = make_generator(tmp_path)
    else:
        monkeypatch.setattr(LastWordsGenerator, "_MAX_RETRIES", 2)
        monkeypatch.setattr(LastWordsGenerator, "_BACKOFF_SCHEDULE", (0, 0))
        gen = make_generator(tmp_path)
        client = FailingFallbackClient(ConnectionError("retry me"))
        gen._client = client  # type: ignore[assignment]

    with pytest.raises(LastWordsSpendBudgetExceeded):
        await generate(
            gen,
            user_request="do X",
            previous_last_words=None,
            dropped_messages=[],
            completer=completer,
            spend_side_call_tokens=spend,
        )

    assert len(charges) == 2
    assert all(charge > 0 for charge in charges)
    if completer is not None:
        assert len(completer.calls) == 1
    else:
        assert client is not None
        assert client.calls == 1


async def test_sibling_call_run_admission_charges_one_merged_timeline(tmp_path):
    """Budget-boundary pin for the merged exchange unit: the annotation
    pipeline charges the fallback candidate as ONE group carrying both
    sibling calls — the same amount as a manually merged-annotation
    reference — and with the side-call budget set to exactly that merged
    estimate the strict ``<`` spend gate refuses the first attempt —
    Phase 4 raises and the strategy retains all current-turn groups
    instead of proceeding toward spill/exclusion."""
    from chrys.kernel import annotate_message_groups
    from chrys.kernel.compaction import (
        GROUP_ANNOTATION_KEY,
        GROUP_HAS_REASONING_KEY,
        GROUP_ID_KEY,
        GROUP_INDEX_KEY,
        GROUP_KIND_KEY,
    )
    from chrys.service.context.compaction.last_words import LastWordsGenerator
    from chrys.service.context.compaction.scoped import build_scoped_group_timeline
    from chrys.service.profiles.models.resolver import default_profile

    def _shape() -> list[Message]:
        return [
            user("do X"),
            Message(
                role="assistant",
                contents=[Content.from_function_call("call_a", "tool_a", arguments={"value": "a"})],
            ),
            Message(
                role="assistant",
                contents=[Content.from_function_call("call_b", "tool_b", arguments={"value": "b"})],
            ),
            Message(
                role="tool",
                contents=[
                    Content.from_function_result("call_a", result="alpha outcome"),
                    Content.from_function_result("call_b", result="beta outcome"),
                ],
            ),
        ]

    def _pipeline_groups():  # type: ignore[no-untyped-def]
        messages = _shape()
        annotate_message_groups(messages, force_reannotate=True)
        timeline = build_scoped_group_timeline(messages, span_start=0, span_end=len(messages), degraded=False)
        return timeline.groups

    def _merged_reference_groups():  # type: ignore[no-untyped-def]
        messages = _shape()
        annotate_message_groups(messages, force_reannotate=True)
        for message in messages[1:]:
            message.additional_properties[GROUP_ANNOTATION_KEY] = {
                GROUP_ID_KEY: "group_merged",
                GROUP_KIND_KEY: "tool_call",
                GROUP_INDEX_KEY: 1,
                GROUP_HAS_REASONING_KEY: False,
            }
        timeline = build_scoped_group_timeline(messages, span_start=0, span_end=len(messages), degraded=False)
        return timeline.groups

    class _NoteClient:
        async def get_response(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            class _Response:
                usage_details = None
                raw_text = structured_note()

            return _Response()

    async def _first_charge(groups) -> int:  # type: ignore[no-untyped-def]
        charges: list[int] = []

        def refuse(estimated_tokens: int) -> bool:
            charges.append(estimated_tokens)
            return False

        gen = LastWordsGenerator(profile=default_profile(), log_dir=tmp_path)
        gen._client = _NoteClient()  # type: ignore[assignment]
        with pytest.raises(LastWordsSpendBudgetExceeded):
            await gen.generate(
                list(groups),
                None,
                degraded_opener=False,
                has_continuation_nudges=False,
                completer=None,
                spend_side_call_tokens=refuse,
            )
        return charges[0]

    charge_pipeline = await _first_charge(_pipeline_groups())
    charge_merged = await _first_charge(_merged_reference_groups())
    assert charge_pipeline == charge_merged

    spent = 0

    def strict_gate(estimated_tokens: int) -> bool:
        nonlocal spent
        spent += estimated_tokens
        return spent < charge_merged

    gen = LastWordsGenerator(profile=default_profile(), log_dir=tmp_path)
    gen._client = _NoteClient()  # type: ignore[assignment]
    with pytest.raises(LastWordsSpendBudgetExceeded):
        await gen.generate(
            list(_pipeline_groups()),
            None,
            degraded_opener=False,
            has_continuation_nudges=False,
            completer=None,
            spend_side_call_tokens=strict_gate,
        )

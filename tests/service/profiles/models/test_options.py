# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for model-profile JSON option parsing."""

from __future__ import annotations

import logging
from typing import Any
from unittest import mock

import pytest

from chrys.foundation.util.chrys_headers import MODEL_ID_HEADER, SESSION_ID_HEADER
from chrys.foundation.util.env_templates import EnvVarResolutionError
from chrys.service.profiles.models import options as options_module
from chrys.service.profiles.models.options import (
    AUTO_INTERLEAVED_THINKING_OPTION,
    STREAM_REQUIRES_FINISH_REASON_OPTION,
    THINKING_BLOCK_BINDING_OPTION,
    effective_chat_options,
    is_anthropic_claude_profile,
    lacks_anthropic_prompt_cache_option,
    parse_chat_options,
    protected_chat_option_keys_warning,
    protected_chat_option_keys_warning_structured,
    responses_store_continuation_warning,
    uses_responses_compact_continuation,
    with_anthropic_prompt_cache_option,
)
from chrys.service.profiles.models.schema import ModelProfile, uses_responses_wire_dialect


def _profile(chat_options: str) -> ModelProfile:
    return ModelProfile(id="p", name="Profile", model_id="gpt-test", chat_options=chat_options)


def test_parse_chat_options_resolves_nested_env_templates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHRYS_TEMP", "0.7")
    monkeypatch.setenv("CHRYS_METADATA_TOKEN", "meta-token")
    monkeypatch.setenv("CHRYS_STOP", "DONE")

    opts = parse_chat_options(
        _profile(
            '{"temperature": "{{CHRYS_TEMP}}", '
            '"metadata": {"token": "{{CHRYS_METADATA_TOKEN}}"}, '
            '"stop": ["{{CHRYS_STOP}}"]}'
        )
    )

    assert opts == {
        "temperature": "0.7",
        "metadata": {"token": "meta-token"},
        "stop": ["DONE"],
    }


def test_parse_chat_options_keeps_resolved_values_as_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHRYS_MAX_TOKENS", "4096")

    opts = parse_chat_options(_profile('{"max_tokens": "{{CHRYS_MAX_TOKENS}}"}'))

    assert opts == {"max_tokens": "4096"}


def test_parse_chat_options_missing_env_template_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHRYS_MISSING_OPTION", raising=False)

    with pytest.raises(EnvVarResolutionError) as info:
        parse_chat_options(_profile('{"metadata": {"token": "{{CHRYS_MISSING_OPTION}}"}}'))

    message = str(info.value)
    assert "CHRYS_MISSING_OPTION" in message
    assert "model profile 'Profile' chat option['metadata']['token']" in message


def test_parse_chat_options_drops_managed_chrys_extra_headers(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="chrys.service.profiles.models.options"):
        opts = parse_chat_options(
            _profile(
                '{"extra_headers": {'
                '"X-Team": "platform", '
                f'"{MODEL_ID_HEADER}": "wrong", '
                f'"{SESSION_ID_HEADER}": "wrong-session", '
                '"X-Session-Id": "wrong-x-session", '
                '"chrys-debug": "wrong-lower", '
                '"CHRYS_TRACE": "wrong-upper"'
                "}}"
            )
        )

    assert opts == {"extra_headers": {"X-Team": "platform"}}
    assert "chat_options.extra_headers contains Chrys-managed header(s)" in caplog.text
    assert MODEL_ID_HEADER in caplog.text
    assert SESSION_ID_HEADER in caplog.text
    assert "X-Session-Id" in caplog.text
    assert "chrys-debug" in caplog.text
    assert "CHRYS_TRACE" in caplog.text
    assert "wrong" not in caplog.text


def test_parse_chat_options_drops_empty_extra_headers_after_managed_chrys_removal() -> None:
    opts = parse_chat_options(_profile(f'{{"extra_headers": {{"{MODEL_ID_HEADER}": "wrong"}}}}'))

    assert opts == {}


def test_parse_chat_options_drops_managed_extra_header_before_env_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CHRYS_MISSING_RESERVED_HEADER", raising=False)

    opts = parse_chat_options(
        _profile(f'{{"extra_headers": {{"{MODEL_ID_HEADER}": "{{{{CHRYS_MISSING_RESERVED_HEADER}}}}"}}}}')
    )

    assert opts == {}


def test_parse_chat_options_strips_protected_keys_before_env_resolution(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("CHRYS_PROTECTED_MISSING", raising=False)
    raw = (
        '{"messages": "{{CHRYS_PROTECTED_MISSING}}", "prompt": "reusable", '
        '"conversation_id": "conv", "previous_response_id": "prev", "conversation": "thread", '
        '"continuation_token": "token", "input": "kept", "temperature": 0.2, '
        '"extra_body": {"messages": [], "input": [], "prompt": {}, "tools": [], "system": "s", '
        '"instructions": "i", "max_tokens": 1, "max_output_tokens": 2, '
        '"max_completion_tokens": 3, "custom": "kept"}}'
    )

    with caplog.at_level(logging.WARNING, logger="chrys.service.profiles.models.options"):
        options = parse_chat_options(_profile(raw))

    assert options == {"input": "kept", "temperature": 0.2, "extra_body": {"custom": "kept"}}
    assert "Profile" in caplog.text
    assert "messages" in caplog.text
    assert "extra_body.max_tokens" in caplog.text


def test_parse_chat_options_joins_list_instructions_and_drops_other_non_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert parse_chat_options(_profile('{"instructions": "Plain."}')) == {"instructions": "Plain."}
    assert parse_chat_options(_profile('{"instructions": null}')) == {"instructions": None}
    assert parse_chat_options(_profile('{"instructions": ["First.", "Second."]}')) == {
        "instructions": "First.\nSecond."
    }

    with caplog.at_level(logging.WARNING, logger="chrys.service.profiles.models.options"):
        dropped_dict = parse_chat_options(_profile('{"instructions": {"text": "x"}, "temperature": 0.2}'))
        dropped_mixed = parse_chat_options(_profile('{"instructions": ["x", 1]}'))

    assert dropped_dict == {"temperature": 0.2}
    assert dropped_mixed == {}
    assert caplog.text.count("chat_options key 'instructions' must be a string or a list of strings") == 2


def test_protected_chat_option_warning_is_pure_and_explains_migration() -> None:
    profile = _profile('{"prompt": {"id": "pmpt_1"}, "conversation_id": "resp_1"}')

    structured = protected_chat_option_keys_warning_structured(profile)
    warning = protected_chat_option_keys_warning(profile)

    assert structured is not None
    assert structured.keys == ("conversation_id", "prompt")
    assert structured.message == warning
    assert warning is not None
    assert "prompt" in warning
    assert "conversation_id" in warning
    assert "store: true" in warning
    assert protected_chat_option_keys_warning(_profile('{"input": "kept"}')) is None
    assert protected_chat_option_keys_warning(_profile("not json")) is None


def test_effective_chat_options_defaults_responses_to_store_false() -> None:
    """Store defaults compose with the default output cap injection."""
    assert effective_chat_options(
        ModelProfile(id="responses", name="Responses", provider="openai", api_style="responses")
    ) == {"store": False, "max_tokens": 32000}
    assert effective_chat_options(
        ModelProfile(
            id="responses",
            name="Responses",
            provider="openai",
            api_style="responses",
            chat_options='{"temperature": 0.1}',
        )
    ) == {"temperature": 0.1, "store": False, "max_tokens": 32000}
    assert effective_chat_options(
        ModelProfile(
            id="responses",
            name="Responses",
            provider="openai",
            api_style="responses",
            chat_options='{"store": null}',
        )
    ) == {"store": False, "max_tokens": 32000}
    assert effective_chat_options(
        ModelProfile(
            id="responses",
            name="Responses",
            provider="openai",
            api_style="responses",
            chat_options='{"store": true}',
        )
    ) == {"store": True, "max_tokens": 32000}


def test_effective_chat_options_extra_body_store_false_vetoes_top_level_true() -> None:
    """A present extra_body store:false wins on the wire — the effective
    top-level value must mirror it so downstream gates never record service
    handles the service never persisted."""

    def _responses(chat_options: str) -> ModelProfile:
        return ModelProfile(
            id="responses", name="Responses", provider="openai", api_style="responses", chat_options=chat_options
        )

    assert effective_chat_options(_responses('{"store": true, "extra_body": {"store": false}}')) == {
        "store": False,
        "extra_body": {"store": False},
        "max_tokens": 32000,
    }
    # A present null is NOT a veto: it reaches the wire as JSON null and
    # selects the provider default (Responses stores).
    assert effective_chat_options(_responses('{"store": true, "extra_body": {"store": null}}')) == {
        "store": True,
        "extra_body": {"store": None},
        "max_tokens": 32000,
    }
    # extra_body store:true never PROMOTES the top-level value — the
    # client-side-history-over-a-storing-wire degenerate mode keeps its
    # injected store:false and stays out of compact continuation.
    assert effective_chat_options(_responses('{"extra_body": {"store": true}}')) == {
        "store": False,
        "extra_body": {"store": True},
        "max_tokens": 32000,
    }
    # Outside openai+responses no normalization applies.
    assert effective_chat_options(
        ModelProfile(
            id="cc",
            name="CC",
            provider="openai",
            api_style="chat_completions",
            chat_options='{"store": true, "extra_body": {"store": false}}',
        )
    ) == {"store": True, "extra_body": {"store": False}, "max_tokens": 32000}


def test_effective_chat_options_chat_completions_gets_only_the_cap_default() -> None:
    assert effective_chat_options(
        ModelProfile(
            id="chat",
            name="Chat Completions",
            provider="openai",
            api_style="chat_completions",
            chat_options='{"temperature": 0.1}',
        )
    ) == {"temperature": 0.1, "max_tokens": 32000}


@pytest.mark.parametrize(
    ("chat_options", "expected_extra_body"),
    [
        ("", None),
        ('{"store": true}', None),
        ('{"store": false}', None),
        ('{"extra_body": {"store": true}}', {"store": False}),
        ('{"extra_body": {"store": false}}', {"store": False}),
        ('{"extra_body": {"store": null}}', {"store": False}),
        ('{"store": true, "extra_body": {"store": false}}', {"store": False}),
        ('{"store": false, "extra_body": {"store": true}}', {"store": False}),
    ],
)
def test_effective_chat_options_forces_deepseek_responses_profile_store_false(
    chat_options: str,
    expected_extra_body: dict[str, object] | None,
) -> None:
    profile = ModelProfile(
        id="deepseek-responses",
        name="DeepSeek Responses",
        provider="deepseek-openai",
        api_style="responses",
        chat_options=chat_options,
    )

    effective = effective_chat_options(profile)

    assert effective is not None
    assert effective["store"] is False
    if expected_extra_body is None:
        assert "extra_body" not in effective
    else:
        assert effective["extra_body"] == expected_extra_body


def test_deepseek_responses_store_normalization_warns_once_with_profile_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    profile = ModelProfile(
        id="deepseek-responses",
        name="My DeepSeek",
        provider="deepseek-openai",
        api_style="responses",
        chat_options='{"store": true, "extra_body": {"store": null}}',
    )

    with caplog.at_level(logging.WARNING, logger="chrys.service.profiles.models.options"):
        effective_chat_options(profile)
        effective_chat_options(profile)

    warnings = [record.message for record in caplog.records if "stateless DeepSeek Responses" in record.message]
    assert len(warnings) == 1
    assert "My DeepSeek" in warnings[0]


def test_uses_responses_wire_dialect_includes_deepseek_but_continuation_stays_openai_only() -> None:
    deepseek = ModelProfile(
        id="deepseek",
        name="DeepSeek",
        provider="deepseek-openai",
        api_style="responses",
    )
    openai = ModelProfile(id="openai", name="OpenAI", provider="openai", api_style="responses")

    assert uses_responses_wire_dialect(deepseek) is True
    assert uses_responses_wire_dialect(openai) is True
    assert uses_responses_compact_continuation(deepseek, {"store": True}) is False
    assert uses_responses_compact_continuation(openai, {"store": True}) is True


def test_effective_chat_options_applies_max_output_tokens_as_max_tokens_default() -> None:
    """The profile output cap becomes the live max_tokens default.

    The loader migrates legacy chat-options ``max_tokens`` into
    ``max_output_tokens``; re-applying it here keeps migrated profiles
    sending byte-identical live requests."""
    # Field set, no chat options at all: injected.
    assert effective_chat_options(ModelProfile(id="p", name="P", model_id="gpt-test", max_output_tokens=8192)) == {
        "max_tokens": 8192
    }
    # Field set alongside other options: injected without clobbering them.
    assert effective_chat_options(
        ModelProfile(
            id="p",
            name="P",
            model_id="gpt-test",
            max_output_tokens=8192,
            chat_options='{"temperature": 0.1}',
        )
    ) == {"temperature": 0.1, "max_tokens": 8192}
    # An explicit max_tokens in the parsed options wins (programmatic callers).
    assert effective_chat_options(
        ModelProfile(
            id="p",
            name="P",
            model_id="gpt-test",
            max_output_tokens=8192,
            chat_options='{"max_tokens": 4000}',
        )
    ) == {"max_tokens": 4000}
    # A provider-native output-cap spelling also blocks injection: adding
    # canonical max_tokens next to it would override it at serialization time.
    assert effective_chat_options(
        ModelProfile(
            id="p",
            name="P",
            provider="openai",
            api_style="responses",
            model_id="gpt-test",
            max_output_tokens=8192,
            chat_options='{"max_output_tokens": 4096}',
        )
    ) == {"max_output_tokens": 4096, "store": False}
    assert effective_chat_options(
        ModelProfile(
            id="p",
            name="P",
            model_id="gpt-test",
            max_output_tokens=8192,
            chat_options='{"max_completion_tokens": 4096}',
        )
    ) == {"max_completion_tokens": 4096}
    # Defensive: a programmatic profile may still carry 0 (the loader never
    # produces one) — nothing injected, empty options stay None.
    assert effective_chat_options(ModelProfile(id="p", name="P", model_id="gpt-test", max_output_tokens=0)) is None
    # Composes with the Responses store default.
    assert effective_chat_options(
        ModelProfile(
            id="p",
            name="P",
            provider="openai",
            api_style="responses",
            model_id="gpt-test",
            max_output_tokens=8192,
        )
    ) == {"store": False, "max_tokens": 8192}


def test_uses_responses_compact_continuation_requires_openai_responses_store_true() -> None:
    responses = ModelProfile(id="responses", name="Responses", provider="openai", api_style="responses")
    chat = ModelProfile(id="chat", name="Chat", provider="openai", api_style="chat_completions")
    compatible = {"store": True}

    assert uses_responses_compact_continuation(responses, compatible) is True
    assert uses_responses_compact_continuation(responses, {"store": False}) is False
    assert uses_responses_compact_continuation(responses, None) is False
    assert uses_responses_compact_continuation(chat, compatible) is False
    # extra_body store:false vetoes the top-level value; null and true don't.
    assert uses_responses_compact_continuation(responses, {"store": True, "extra_body": {"store": False}}) is False
    assert uses_responses_compact_continuation(responses, {"store": True, "extra_body": {"store": None}}) is True
    assert uses_responses_compact_continuation(responses, {"store": False, "extra_body": {"store": True}}) is False
    assert (
        uses_responses_compact_continuation(
            ModelProfile(id="other", name="Other", provider="anthropic", api_style="responses"),
            compatible,
        )
        is False
    )


def test_responses_store_continuation_warning_fires_only_for_store_true_responses() -> None:
    def profile(**overrides: str) -> ModelProfile:
        defaults = {"id": "p", "name": "P", "model_id": "gpt-test"}
        defaults.update(overrides)
        return ModelProfile(**defaults)

    warning = responses_store_continuation_warning(
        profile(provider="openai", api_style="responses", chat_options='{"store": true}')
    )
    assert warning is not None
    assert "compaction" in warning

    # store absent (defaults to false), store false, wrong api style, wrong provider.
    assert responses_store_continuation_warning(profile(provider="openai", api_style="responses")) is None
    assert (
        responses_store_continuation_warning(
            profile(provider="openai", api_style="responses", chat_options='{"store": false}')
        )
        is None
    )
    # extra_body store:false vetoes at save time too — the runtime never
    # enters store mode for this profile, so warning would be wrong.
    assert (
        responses_store_continuation_warning(
            profile(
                provider="openai",
                api_style="responses",
                chat_options='{"store": true, "extra_body": {"store": false}}',
            )
        )
        is None
    )
    assert (
        responses_store_continuation_warning(
            profile(
                provider="openai",
                api_style="responses",
                chat_options='{"store": true, "extra_body": {"store": null}}',
            )
        )
        is not None
    )
    assert (
        responses_store_continuation_warning(
            profile(provider="openai", api_style="chat_completions", chat_options='{"store": true}')
        )
        is None
    )
    assert (
        responses_store_continuation_warning(
            profile(provider="anthropic", api_style="responses", chat_options='{"store": true}')
        )
        is None
    )


def test_responses_store_continuation_warning_tolerates_unresolvable_templates_and_bad_json() -> None:
    """The save-time warning must not raise on env templates that resolve only at run time."""

    def profile(chat_options: str) -> ModelProfile:
        return ModelProfile(
            id="p",
            name="P",
            provider="openai",
            api_style="responses",
            model_id="gpt-test",
            chat_options=chat_options,
        )

    # An unresolvable template elsewhere in the options must not mask the warning.
    warning = responses_store_continuation_warning(profile('{"store": true, "api_key": "{{NOT_SET_VAR_FOR_TEST}}"}'))
    assert warning is not None

    # Malformed options never enable store mode, so no warning either.
    assert responses_store_continuation_warning(profile("not json")) is None
    assert responses_store_continuation_warning(profile('["store"]')) is None


# ---------------------------------------------------------------------------
# Anthropic prompt caching
# ---------------------------------------------------------------------------


def _anthropic(
    model_id: str = "claude-sonnet-5-5", chat_options: str = "", provider: str = "anthropic"
) -> ModelProfile:
    return ModelProfile(id="p", name="Profile", provider=provider, model_id=model_id, chat_options=chat_options)


@pytest.mark.parametrize(
    ("provider", "model_id", "expected"),
    [
        ("anthropic", "claude-sonnet-5-5", True),
        ("anthropic", "anthropic/Claude-Opus-5-5", True),
        ("anthropic", "kimi-k2", False),
        ("openai", "claude-sonnet-5-5", False),
    ],
)
def test_is_anthropic_claude_profile(provider: str, model_id: str, expected: bool) -> None:
    assert is_anthropic_claude_profile(_anthropic(model_id, provider=provider)) is expected


@pytest.mark.parametrize(
    ("chat_options", "expected"),
    [
        pytest.param("", True, id="no-options"),
        pytest.param('{"temperature": 0.2}', True, id="other-options"),
        pytest.param('{"extra_body": {"top_k": 5}}', True, id="extra-body-without-it"),
        pytest.param('{"extra_body": null}', True, id="null-extra-body"),
        pytest.param('{"cache_control": {"type": "ephemeral"}}', False, id="top-level"),
        pytest.param('{"extra_body": {"cache_control": {"type": "ephemeral"}}}', False, id="in-extra-body"),
        pytest.param('{"extra_body": "raw"}', False, id="extra-body-not-an-object"),
        pytest.param("[1, 2]", False, id="options-not-an-object"),
        pytest.param("{not json", False, id="malformed"),
    ],
)
def test_lacks_anthropic_prompt_cache_option(chat_options: str, expected: bool) -> None:
    assert lacks_anthropic_prompt_cache_option(_anthropic(chat_options=chat_options)) is expected


def test_lacks_anthropic_prompt_cache_option_ignores_other_profiles() -> None:
    assert lacks_anthropic_prompt_cache_option(_anthropic("kimi-k2")) is False
    assert lacks_anthropic_prompt_cache_option(_anthropic(provider="openai")) is False


def test_with_anthropic_prompt_cache_option_merges_without_mutating() -> None:
    extra_body = {"top_k": 5, "cache_control": {"type": "persistent"}}

    merged = with_anthropic_prompt_cache_option(extra_body)

    assert merged == {"top_k": 5, "cache_control": {"type": "ephemeral"}}
    assert extra_body == {"top_k": 5, "cache_control": {"type": "persistent"}}
    assert with_anthropic_prompt_cache_option(None) == {"cache_control": {"type": "ephemeral"}}
    # Each call hands out its own dict: an edit to one result never reaches the next.
    first = with_anthropic_prompt_cache_option(None)
    first["cache_control"]["ttl"] = "1h"
    assert with_anthropic_prompt_cache_option(None) == {"cache_control": {"type": "ephemeral"}}


@pytest.mark.parametrize(
    ("provider", "api_style", "sent"),
    [
        ("openai", "chat_completions", True),
        ("deepseek-openai", "chat_completions", True),
        ("glm-openai", "chat_completions", True),
        ("openai", "responses", False),
        ("deepseek-openai", "responses", False),
        ("anthropic", "chat_completions", False),
        ("mock", "chat_completions", False),
    ],
)
def test_the_finish_reason_requirement_reaches_only_chat_completions_clients(
    provider: str, api_style: str, sent: bool
) -> None:
    profile = ModelProfile(
        id="p",
        name="Profile",
        provider=provider,
        api_style=api_style,
        model_id="m",
        chat_options='{"temperature": 0.2}',
        stream_requires_finish_reason=True,
    )

    options = effective_chat_options(profile) or {}

    assert options.get(STREAM_REQUIRES_FINISH_REASON_OPTION) is (True if sent else None)
    assert options["temperature"] == 0.2
    # The option rides only on the effective copy.
    assert STREAM_REQUIRES_FINISH_REASON_OPTION not in (parse_chat_options(profile) or {})


def test_a_profile_without_the_finish_reason_requirement_adds_no_option() -> None:
    profile = ModelProfile(id="p", name="Profile", provider="glm-openai", model_id="m")

    assert STREAM_REQUIRES_FINISH_REASON_OPTION not in (effective_chat_options(profile) or {})


@pytest.mark.parametrize("provider", ["anthropic", "openai", "deepseek-openai", "glm-openai", "mock"])
@pytest.mark.parametrize(
    ("binding", "interleaved", "expected"),
    [
        ("auto", True, {}),
        ("drop_block", True, {THINKING_BLOCK_BINDING_OPTION: "drop_block"}),
        ("error", True, {THINKING_BLOCK_BINDING_OPTION: "error"}),
        ("off", False, {THINKING_BLOCK_BINDING_OPTION: "off", AUTO_INTERLEAVED_THINKING_OPTION: False}),
        ("auto", False, {AUTO_INTERLEAVED_THINKING_OPTION: False}),
    ],
    ids=["defaults", "drop-block", "error", "both", "interleaved-off"],
)
def test_thinking_settings_reach_only_anthropic_clients_and_only_when_changed(
    provider: str, binding: Any, interleaved: bool, expected: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = ModelProfile(
        id="p",
        name="Profile",
        provider=provider,
        model_id="m",
        max_output_tokens=0,
        thinking_block_binding=binding,
        auto_interleaved_thinking=interleaved,
    )
    parsed = {"thinking": {"type": "adaptive"}}
    monkeypatch.setattr(
        options_module, "parse_chat_options", mock.create_autospec(parse_chat_options, return_value=parsed)
    )

    options = effective_chat_options(profile) or {}

    settings = {
        key: options[key] for key in (THINKING_BLOCK_BINDING_OPTION, AUTO_INTERLEAVED_THINKING_OPTION) if key in options
    }
    assert settings == (expected if provider == "anthropic" else {})
    assert options["thinking"] == {"type": "adaptive"}
    # The settings ride only on the effective copy.
    assert parsed == {"thinking": {"type": "adaptive"}}

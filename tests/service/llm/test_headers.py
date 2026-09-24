# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for _build_default_headers in chrys.service.llm.clients."""

from __future__ import annotations

import logging
import sys

import pytest

from chrys.foundation.util.chrys_headers import (
    MODEL_ID_HEADER,
    PARENT_SESSION_ID_HEADER,
    SESSION_ID_HEADER,
    X_PARENT_SESSION_ID_HEADER,
    X_SESSION_ID_HEADER,
)
from chrys.foundation.util.env_templates import EnvVarResolutionError
from chrys.service.llm.clients import _build_default_headers, _provider_sdk_user_agent
from chrys.service.profiles.models.schema import ModelProfile


def _expected_user_agent(sdk_user_agent: str | None = None) -> str:
    from chrys import __version__

    runtime = sys.version_info
    products = [f"Chrys/{__version__}", f"Python/{runtime.major}.{runtime.minor}.{runtime.micro}"]
    if sdk_user_agent:
        products.append(sdk_user_agent)
    return " ".join(products)


def _profile(http_headers: str = "") -> ModelProfile:
    return ModelProfile(
        id="p",
        name="p",
        provider="openai",
        model_id="x",
        http_headers=http_headers,
    )


def test_default_headers_without_session_id():
    headers = _build_default_headers(None, _profile())
    assert headers["User-Agent"] == _expected_user_agent()
    assert headers["X-Client-Name"] == "chrys"
    assert "X-Client-Version" in headers
    assert headers[MODEL_ID_HEADER] == "x"
    assert X_SESSION_ID_HEADER not in headers
    assert SESSION_ID_HEADER not in headers
    assert X_PARENT_SESSION_ID_HEADER not in headers
    assert PARENT_SESSION_ID_HEADER not in headers


def test_default_headers_with_session_id():
    headers = _build_default_headers("abc123def456", _profile())
    assert headers["User-Agent"] == _expected_user_agent()
    assert headers["X-Client-Name"] == "chrys"
    assert headers[X_SESSION_ID_HEADER] == "abc123def456"
    assert headers[SESSION_ID_HEADER] == "abc123def456"
    assert headers[MODEL_ID_HEADER] == "x"


def test_default_headers_with_parent_session_id():
    headers = _build_default_headers(
        "child-session",
        _profile(),
        parent_session_id="parent-session",
    )

    assert headers[X_SESSION_ID_HEADER] == "child-session"
    assert headers[SESSION_ID_HEADER] == "child-session"
    assert headers[X_PARENT_SESSION_ID_HEADER] == "parent-session"
    assert headers[PARENT_SESSION_ID_HEADER] == "parent-session"


def test_model_id_header_comes_from_profile():
    profile = _profile()
    profile.model_id = "gpt-5.2"

    headers = _build_default_headers(None, profile)

    assert headers[MODEL_ID_HEADER] == "gpt-5.2"


def test_version_matches_package():
    from chrys import __version__

    headers = _build_default_headers(None, _profile())
    assert headers["X-Client-Version"] == __version__
    assert headers["User-Agent"] == _expected_user_agent()


def test_user_agent_can_include_provider_sdk_token():
    sdk_user_agent = "AsyncOpenAI/openai2.36.0"

    headers = _build_default_headers(None, _profile(), sdk_user_agent=sdk_user_agent)

    assert headers["User-Agent"] == _expected_user_agent(sdk_user_agent)


def test_openai_sdk_user_agent_matches_sdk_property():
    from openai import AsyncOpenAI

    expected = AsyncOpenAI(api_key="sk-test").user_agent

    assert _provider_sdk_user_agent("openai") == expected
    assert _provider_sdk_user_agent("deepseek-openai") == expected


def test_anthropic_sdk_user_agent_matches_sdk_property():
    from anthropic import AsyncAnthropic

    assert _provider_sdk_user_agent("anthropic") == AsyncAnthropic(api_key="sk-test").user_agent


def test_custom_headers_merge():
    headers = _build_default_headers(
        None,
        _profile(http_headers='{"X-Team": "platform", "X-Env": "staging"}'),
    )
    assert headers["X-Client-Name"] == "chrys"
    assert headers["X-Team"] == "platform"
    assert headers["X-Env"] == "staging"


def test_custom_headers_resolve_env_templates(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CHRYS_HEADER_TOKEN", "secret-token")
    headers = _build_default_headers(
        None,
        _profile(http_headers='{"Authorization": "Bearer {{CHRYS_HEADER_TOKEN}}"}'),
    )

    assert headers["Authorization"] == "Bearer secret-token"


def test_custom_headers_missing_env_template_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CHRYS_MISSING_HEADER", raising=False)

    with pytest.raises(EnvVarResolutionError) as info:
        _build_default_headers(None, _profile(http_headers='{"Authorization": "{{CHRYS_MISSING_HEADER}}"}'))

    message = str(info.value)
    assert "CHRYS_MISSING_HEADER" in message
    assert "model profile 'p' extra header['Authorization']" in message


def test_custom_headers_override_defaults():
    headers = _build_default_headers(
        None,
        _profile(http_headers='{"X-Client-Name": "my-app"}'),
    )
    assert headers["X-Client-Name"] == "my-app"


def test_custom_headers_cannot_override_model_id():
    headers = _build_default_headers(
        None,
        _profile(http_headers='{"Chrys-Model-Id": "wrong-model"}'),
    )

    assert headers[MODEL_ID_HEADER] == "x"


def test_custom_headers_cannot_override_x_session_id_case_insensitively(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING, logger="chrys.service.profiles.models.options"):
        headers = _build_default_headers(
            "real-session",
            _profile(http_headers='{"X-Session-Id": "wrong-session"}'),
        )

    assert headers[X_SESSION_ID_HEADER] == "real-session"
    assert "X-Session-Id" not in headers
    assert "http_headers contains Chrys-managed header(s)" in caplog.text
    assert "X-Session-Id" in caplog.text
    assert "wrong-session" not in caplog.text


def test_custom_headers_cannot_override_chrys_session_id(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING, logger="chrys.service.profiles.models.options"):
        headers = _build_default_headers(
            "real-session",
            _profile(http_headers=f'{{"{SESSION_ID_HEADER}": "wrong-session"}}'),
        )

    assert headers[SESSION_ID_HEADER] == "real-session"
    assert "http_headers contains Chrys-managed header(s)" in caplog.text
    assert SESSION_ID_HEADER in caplog.text
    assert "wrong-session" not in caplog.text


def test_custom_headers_cannot_override_parent_session_id(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING, logger="chrys.service.profiles.models.options"):
        headers = _build_default_headers(
            "child-session",
            _profile(http_headers=f'{{"{X_PARENT_SESSION_ID_HEADER}": "wrong-parent"}}'),
            parent_session_id="real-parent",
        )

    assert headers[X_PARENT_SESSION_ID_HEADER] == "real-parent"
    assert "http_headers contains Chrys-managed header(s)" in caplog.text
    assert X_PARENT_SESSION_ID_HEADER in caplog.text
    assert "wrong-parent" not in caplog.text


def test_custom_headers_drop_managed_chrys_headers(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING, logger="chrys.service.profiles.models.options"):
        headers = _build_default_headers(
            None,
            _profile(http_headers='{"X-Team": "platform", "chrys-debug": "1", "CHRYS_TRACE": "secret"}'),
        )

    assert headers["X-Team"] == "platform"
    assert "chrys-debug" not in headers
    assert "CHRYS_TRACE" not in headers
    assert "http_headers contains Chrys-managed header(s)" in caplog.text
    assert "chrys-debug" in caplog.text
    assert "CHRYS_TRACE" in caplog.text
    assert "secret" not in caplog.text


def test_invalid_json_ignored(caplog):
    with caplog.at_level(logging.WARNING, logger="chrys.service.profiles.models.options"):
        headers = _build_default_headers(None, _profile(http_headers="not-json{"))
    assert headers["X-Client-Name"] == "chrys"
    assert headers[MODEL_ID_HEADER] == "x"
    assert "not valid JSON" in caplog.text


def test_empty_custom_headers():
    headers = _build_default_headers(None, _profile(http_headers=""))
    assert headers["X-Client-Name"] == "chrys"
    assert len(headers) == 4  # user agent + name + version + model id


def test_non_dict_json_ignored():
    """JSON that parses to a non-dict (e.g. a list) is silently ignored."""
    headers = _build_default_headers(None, _profile(http_headers='["a", "b"]'))
    assert headers["X-Client-Name"] == "chrys"
    assert len(headers) == 4

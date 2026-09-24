# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for runtime environment-template resolution."""

from __future__ import annotations

import pytest

from chrys.foundation.util.env_templates import EnvVarResolutionError, resolve_env_templates


def test_resolve_env_templates_replaces_multiple_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHRYS_AUTH_TYPE", "Bearer")
    monkeypatch.setenv("CHRYS_AUTH_TOKEN", "abc123")

    assert resolve_env_templates("{{CHRYS_AUTH_TYPE}} {{CHRYS_AUTH_TOKEN}}", location="test field") == "Bearer abc123"


def test_resolve_env_templates_treats_empty_env_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHRYS_EMPTY_SECRET", "")

    with pytest.raises(EnvVarResolutionError, match="CHRYS_EMPTY_SECRET"):
        resolve_env_templates("{{CHRYS_EMPTY_SECRET}}", location="test field")


def test_resolve_env_templates_does_not_re_resolve_env_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHRYS_OUTER", "literal {{CHRYS_INNER}}")
    monkeypatch.setenv("CHRYS_INNER", "resolved")

    assert resolve_env_templates("{{CHRYS_OUTER}}", location="test field") == "literal {{CHRYS_INNER}}"


def test_resolve_env_templates_preserves_non_env_double_brace_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHRYS_TOKEN", "resolved")

    value = 'payload={{"kind":"json"}} token={{CHRYS_TOKEN}}'

    assert resolve_env_templates(value, location="test field") == 'payload={{"kind":"json"}} token=resolved'


def test_resolve_env_templates_preserves_hyphenated_double_brace_payload() -> None:
    assert resolve_env_templates("{{MY-TOKEN}}", location="test field") == "{{MY-TOKEN}}"


def test_resolve_env_templates_allows_escaped_literal_double_braces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHRYS_TOKEN", "resolved")

    value = 'payload={{{{"kind":"json"}}}} token={{CHRYS_TOKEN}}'

    assert resolve_env_templates(value, location="test field") == 'payload={{"kind":"json"}} token=resolved'


def test_resolve_env_templates_rejects_spaces_inside_braces() -> None:
    with pytest.raises(EnvVarResolutionError, match="no spaces"):
        resolve_env_templates("{{ CHRYS_TOKEN }}", location="test field")


def test_resolve_env_templates_rejects_empty_placeholder() -> None:
    with pytest.raises(EnvVarResolutionError, match="Invalid environment variable placeholder"):
        resolve_env_templates("{{}}", location="test field")


def test_resolve_env_templates_rejects_stray_closing_braces() -> None:
    with pytest.raises(EnvVarResolutionError, match="Invalid environment variable placeholder"):
        resolve_env_templates("value }}", location="test field")

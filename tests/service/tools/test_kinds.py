# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for ``chrys.service.tools.kinds`` — bare constants and the out-of-band kind channel."""

from __future__ import annotations

import logging

import pytest

from chrys.kernel import FunctionTool
from chrys.service.tools import kinds
from chrys.service.tools.kinds import TOOL_KINDS, get_tool_kind, set_tool_kind, strip_legacy_kind_prefix, tool


def _exported_kind_constants() -> dict[str, str]:
    """All ``KIND_*`` string constants exported by :mod:`chrys.service.tools.kinds`."""
    return {name: value for name, value in vars(kinds).items() if name.startswith("KIND_") and isinstance(value, str)}


def _plain_tool(name: str = "run") -> FunctionTool:
    def run(command: str) -> str:
        return command

    return FunctionTool(name=name, description="d", func=run)


def test_kind_constants_are_bare() -> None:
    """Constants use the bare form — no namespace prefix, shared by YAML and runtime."""
    for name, value in _exported_kind_constants().items():
        assert value, f"{name} is empty"
        assert not value.startswith("chrys."), f"{name}={value!r} still carries the removed 'chrys.' prefix"


def test_every_kind_constant_is_registered_in_tool_kinds() -> None:
    """``TOOL_KINDS`` is exactly the set of ``KIND_*`` constant values.

    Without this, adding a new kind without registering it means
    ``ApprovalPolicy`` would never recognize the kind-only override key for a
    toolset that doesn't currently include the kind.
    """
    constants = _exported_kind_constants()
    assert set(constants.values()) == set(TOOL_KINDS), (
        "KIND_* constants and TOOL_KINDS diverged — register new kinds in both."
    )
    assert len(constants.values()) == len(set(constants.values())), "duplicate KIND_* values"


def test_set_get_roundtrip_leaves_function_tool_kind_none() -> None:
    t = _plain_tool()
    assert get_tool_kind(t) is None
    set_tool_kind(t, kinds.KIND_SHELL)
    assert get_tool_kind(t) == "shell"
    # FunctionTool.kind is never written: chrys_kind is the one kind channel
    # (see test_tool_kind_wire_boundary.py).
    assert t.kind is None


def test_get_tool_kind_handles_objects_without_attribute() -> None:
    assert get_tool_kind(object()) is None


def test_tool_decorator_records_kind_out_of_band() -> None:
    @tool(kind=kinds.KIND_SEARCH)
    def find(pattern: str) -> str:
        return pattern

    assert isinstance(find, FunctionTool)
    assert get_tool_kind(find) == "search"
    assert find.kind is None


def test_tool_decorator_without_kind() -> None:
    @tool
    def bare(x: int) -> str:
        return str(x)

    @tool()
    def empty_parens(x: int) -> str:
        return str(x)

    assert isinstance(bare, FunctionTool)
    assert get_tool_kind(bare) is None
    assert bare.kind is None
    assert isinstance(empty_parens, FunctionTool)
    assert get_tool_kind(empty_parens) is None


def test_tool_decorator_passes_other_kwargs_through() -> None:
    @tool(kind=kinds.KIND_SLEEP, name="custom_name")
    def f(seconds: int) -> str:
        return str(seconds)

    assert f.name == "custom_name"
    assert get_tool_kind(f) == "sleep"
    assert f.kind is None


def test_instance_method_tool_keeps_kind_across_binding() -> None:
    """``FunctionTool.__get__`` clones per access; the clone must carry ``chrys_kind``."""

    class Tools:
        @tool(kind=kinds.KIND_SHELL)
        def execute(self, command: str) -> str:
            return command

    bound = Tools().execute
    assert bound is not Tools.__dict__["execute"]
    assert get_tool_kind(bound) == "shell"
    assert bound.kind is None


def test_strip_legacy_prefix_known_kind_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="chrys.foundation.tool_kinds"):
        assert strip_legacy_kind_prefix("chrys.shell", source="unit test") == "shell"
    assert any("'chrys.shell'" in r.message and "unit test" in r.message for r in caplog.records)


def test_strip_legacy_prefix_compound_key_requires_opt_in() -> None:
    legacy = "chrys.filesystem.write.write_file"
    assert strip_legacy_kind_prefix(legacy, allow_tool_suffix=True) == "filesystem.write.write_file"
    # Kind-only fields (hook ``match.tool_kind``) never carried tool suffixes.
    assert strip_legacy_kind_prefix(legacy) == legacy


def test_strip_legacy_prefix_leaves_unrecognized_values_alone(caplog: pytest.LogCaptureFixture) -> None:
    """Only values meaningful under the old prefixed scheme are rewritten."""
    with caplog.at_level(logging.WARNING, logger="chrys.foundation.tool_kinds"):
        # Possibly a literal tool name — never matched a runtime kind pre-P1 either.
        assert strip_legacy_kind_prefix("chrys.vendor_tool", allow_tool_suffix=True) == "chrys.vendor_tool"
        assert strip_legacy_kind_prefix("chrys.") == "chrys."
        # Bare forms are already canonical.
        assert strip_legacy_kind_prefix("shell") == "shell"
        assert strip_legacy_kind_prefix("filesystem.write.write_file", allow_tool_suffix=True) == (
            "filesystem.write.write_file"
        )
    assert not caplog.records

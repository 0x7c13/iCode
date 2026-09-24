# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for ToolRegistry."""

from __future__ import annotations

from pathlib import Path

import pytest

from chrys.foundation.models.session_env import SessionEnvironment
from chrys.service.tools.registry import ToolRegistry


def _dummy_tool() -> str:
    return "dummy"


def test_register_and_get() -> None:
    reg = ToolRegistry()
    reg.register(_dummy_tool, name="my_tool")
    assert reg.get("my_tool") is _dummy_tool
    assert reg.get("unknown") is None


def test_unregister() -> None:
    reg = ToolRegistry()
    reg.register(_dummy_tool, name="my_tool", category="test")
    reg.unregister("my_tool")
    assert reg.get("my_tool") is None
    assert reg.get_by_category("test") == []


def test_get_all() -> None:
    reg = ToolRegistry()
    reg.register(_dummy_tool, name="a")
    reg.register(_dummy_tool, name="b")
    assert len(reg.get_all()) == 2


def test_get_by_category() -> None:
    reg = ToolRegistry()
    reg.register(_dummy_tool, name="t1", category="cat_a")
    reg.register(_dummy_tool, name="t2", category="cat_b")
    assert len(reg.get_by_category("cat_a")) == 1
    assert len(reg.get_by_category("cat_b")) == 1
    assert reg.get_by_category("unknown") == []


def test_load_builtins_filesystem_write() -> None:
    reg = ToolRegistry()
    tools = reg.load_builtins(["filesystem.write"])
    assert len(tools) == 2  # write_file, edit_file
    assert len(reg.get_all()) == 2


def test_load_builtins_all_categories() -> None:
    from chrys.foundation.models.session_env import SessionEnvironment

    runtime = SessionEnvironment.capture()
    reg = ToolRegistry()
    tools = reg.load_builtins(["filesystem.write", "filesystem.read", "shell"], runtime=runtime)
    # filesystem.write: 2 (write_file, edit_file) + filesystem.read: 2 + shell: 1 detected shell tool
    # On Windows with Git Bash installed, shell adds an extra git_bash tool
    extra_shells = len(runtime.platform.extra_shells)
    assert len(tools) == 5 + extra_shells
    assert len(reg.get_all()) == 5 + extra_shells


def test_reserved_chrys_names_cover_every_builtin_tool() -> None:
    from chrys.foundation.models.session_env import SessionEnvironment
    from chrys.service.tools.names import chrys_reserved_tool_names

    runtime = SessionEnvironment.capture()
    reg = ToolRegistry()
    tools = reg.load_builtins(
        [
            "filesystem.read",
            "filesystem.write",
            "search",
            "shell",
            "ask_user",
            "sleep",
            "doc_converter",
            "todo",
        ],
        runtime=runtime,
    )

    assert {tool.name for tool in tools} <= chrys_reserved_tool_names()


def test_load_builtins_unknown_category() -> None:
    reg = ToolRegistry()
    tools = reg.load_builtins(["nonexistent"])
    assert tools == []


def test_clear() -> None:
    reg = ToolRegistry()
    reg.load_builtins(["filesystem.write"])
    reg.clear()
    assert reg.get_all() == []


def test_register_auto_name() -> None:
    """register() without explicit name should use __name__."""
    reg = ToolRegistry()
    reg.register(_dummy_tool)
    assert reg.get("_dummy_tool") is _dummy_tool


def test_load_builtins_search() -> None:
    reg = ToolRegistry()
    tools = reg.load_builtins(["search"])
    assert len(tools) == 2  # grep, glob


def test_runtime_bound_filesystem_and_search_tools_preserve_tool_schema() -> None:
    """Instance tools preserve function schemas, with search descriptions reflecting runtime settings."""
    from chrys.foundation.models.session_env import SessionEnvironment
    from chrys.service.tools.builtins.filesystem import edit_file, read_file, view_image, write_file
    from chrys.service.tools.builtins.search import glob, grep

    runtime = SessionEnvironment.capture()
    reg = ToolRegistry()
    tools = reg.load_builtins(["filesystem.read", "filesystem.write", "search"], runtime=runtime)
    loaded = {tool.name: tool for tool in tools}

    originals = {
        "read_file": read_file,
        "view_image": view_image,
        "write_file": write_file,
        "edit_file": edit_file,
        "grep": grep,
        "glob": glob,
    }
    for name, original in originals.items():
        generated = loaded[name]
        expected_schema = original.to_json_schema_spec()
        if name in {"grep", "glob"}:
            assert "Respect .gitignore is enabled" in generated.description
            expected_schema["function"]["description"] = generated.description
        assert generated.to_json_schema_spec() == expected_schema


def test_load_builtins_ask_user() -> None:
    reg = ToolRegistry()
    tools = reg.load_builtins(["ask_user"])
    assert len(tools) == 1


def test_load_builtins_sleep() -> None:
    reg = ToolRegistry()
    tools = reg.load_builtins(["sleep"])
    assert len(tools) == 1
    assert getattr(tools[0], "name", "") == "sleep"


def test_load_builtins_todo() -> None:
    reg = ToolRegistry()
    tools = reg.load_builtins(["todo"])
    assert len(tools) == 1
    assert getattr(tools[0], "name", "") == "todo_write"


def test_load_builtins_filesystem_read() -> None:
    reg = ToolRegistry()
    tools = reg.load_builtins(["filesystem.read"])
    assert len(tools) == 2  # read_file, view_image


def test_load_builtins_shell_auto_captures_runtime() -> None:
    """Shell category creates SessionEnvironment automatically if not provided."""
    reg = ToolRegistry()
    tools = reg.load_builtins(["shell"])
    assert len(tools) >= 1  # at least the primary shell


def test_load_multiple_categories() -> None:
    reg = ToolRegistry()
    tools = reg.load_builtins(["filesystem.read", "search", "ask_user"])
    assert len(tools) == 5  # 2 + 2 + 1


def test_unregister_nonexistent() -> None:
    """Unregistering a tool that doesn't exist should not raise."""
    reg = ToolRegistry()
    reg.unregister("nonexistent")  # no-op


@pytest.mark.parametrize(
    "categories",
    [
        ["doc_converter", "filesystem.read"],
        ["filesystem.read", "doc_converter"],
    ],
)
def test_doc_converter_image_capability_is_independent_of_category_order(
    categories: list[str],
    tmp_path: Path,
) -> None:
    reg = ToolRegistry(vision_enabled=True)
    reg.load_builtins(categories, runtime=SessionEnvironment.capture(), session_dir=tmp_path)

    assert len(reg._doc_converters) == 1
    assert reg._doc_converters[0]._image_extraction_enabled is True


def test_doc_converter_image_capability_defaults_off(tmp_path: Path) -> None:
    reg = ToolRegistry()
    reg.load_builtins(
        ["doc_converter", "filesystem.read"],
        runtime=SessionEnvironment.capture(),
        session_dir=tmp_path,
    )

    assert reg._doc_converters[0]._image_extraction_enabled is False


def test_doc_converter_image_capability_requires_view_image(tmp_path: Path) -> None:
    reg = ToolRegistry(vision_enabled=True)
    reg.load_builtins(["doc_converter"], runtime=SessionEnvironment.capture(), session_dir=tmp_path)

    assert reg._doc_converters[0]._image_extraction_enabled is False


def test_doc_converter_image_capability_refreshes_across_load_calls(tmp_path: Path) -> None:
    runtime = SessionEnvironment.capture()
    reg = ToolRegistry(vision_enabled=True)
    reg.load_builtins(["doc_converter"], runtime=runtime, session_dir=tmp_path)
    converter = reg._doc_converters[0]
    assert converter._image_extraction_enabled is False

    reg.load_builtins(["filesystem.read"], runtime=runtime, session_dir=tmp_path)

    assert converter._image_extraction_enabled is True


def test_doc_converter_image_capability_refreshes_every_instance(tmp_path: Path) -> None:
    runtime = SessionEnvironment.capture()
    reg = ToolRegistry(vision_enabled=True)
    reg.load_builtins(["doc_converter", "doc_converter"], runtime=runtime, session_dir=tmp_path)

    assert len(reg._doc_converters) == 2
    assert all(not converter._image_extraction_enabled for converter in reg._doc_converters)

    reg.load_builtins(["filesystem.read"], runtime=runtime, session_dir=tmp_path)

    assert all(converter._image_extraction_enabled for converter in reg._doc_converters)


def test_doc_converter_image_capability_tracks_unregister_and_register(tmp_path: Path) -> None:
    reg = ToolRegistry(vision_enabled=True)
    reg.load_builtins(
        ["doc_converter", "filesystem.read"],
        runtime=SessionEnvironment.capture(),
        session_dir=tmp_path,
    )
    converter = reg._doc_converters[0]
    view_image = reg.get("view_image")
    assert view_image is not None
    assert converter._image_extraction_enabled is True

    reg.unregister("view_image")
    assert converter._image_extraction_enabled is False

    reg.register(view_image, category="filesystem.read")
    assert converter._image_extraction_enabled is True


def test_clear_revokes_stale_doc_converter_capability(tmp_path: Path) -> None:
    reg = ToolRegistry(vision_enabled=True)
    reg.load_builtins(
        ["doc_converter", "filesystem.read"],
        runtime=SessionEnvironment.capture(),
        session_dir=tmp_path,
    )
    converter = reg._doc_converters[0]
    assert converter._image_extraction_enabled is True

    reg.clear()

    assert converter._image_extraction_enabled is False
    assert reg._doc_converters == []

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Round-trip tests for the memory section of agent profile YAML."""

from __future__ import annotations

import pytest

from chrys.service.profiles.agents.loader import load_profile_from_yaml
from chrys.service.profiles.agents.schema import AgentProfile, MemoryConfig
from chrys.service.profiles.agents.serializer import profile_to_dict


def test_memory_defaults_omitted_from_serialization() -> None:
    """Empty MemoryConfig must not pollute saved YAML."""
    profile = AgentProfile(name="test")

    data = profile_to_dict(profile)

    assert "memory" not in data


def test_memory_serialization_includes_lists() -> None:
    profile = AgentProfile(
        name="test",
        memory=MemoryConfig(files=["docs/spec.md"], folders=["knowledge"]),
    )

    data = profile_to_dict(profile)

    assert data["memory"] == {"files": ["docs/spec.md"], "folders": ["knowledge"]}


def test_memory_yaml_roundtrip(tmp_path) -> None:
    """Save → load preserves files and folders lists."""
    import yaml

    profile = AgentProfile(
        name="rt-profile",
        memory=MemoryConfig(
            files=["a.md", "b.txt"],
            folders=["docs", "knowledge"],
        ),
    )
    path = tmp_path / "rt-profile.yaml"
    path.write_text(yaml.safe_dump(profile_to_dict(profile)), encoding="utf-8")

    loaded = load_profile_from_yaml(path)

    assert loaded.memory.files == ["a.md", "b.txt"]
    assert loaded.memory.folders == ["docs", "knowledge"]


def test_loader_accepts_missing_memory_block(tmp_path) -> None:
    """Profiles written before this feature load fine with empty memory."""
    p = tmp_path / "min.yaml"
    p.write_text("name: minimal\n", encoding="utf-8")

    profile = load_profile_from_yaml(p)

    assert profile.memory.files == []
    assert profile.memory.folders == []


def test_loader_accepts_empty_memory_block(tmp_path) -> None:
    p = tmp_path / "empty-memory.yaml"
    p.write_text(
        """\
name: empty-mem
memory: {}
""",
        encoding="utf-8",
    )

    profile = load_profile_from_yaml(p)

    assert profile.memory.files == []
    assert profile.memory.folders == []


def test_loader_accepts_partial_memory_block(tmp_path) -> None:
    """Specifying just files (no folders) or vice versa must not crash."""
    p = tmp_path / "partial.yaml"
    p.write_text(
        """\
name: partial
memory:
  files:
    - notes.md
""",
        encoding="utf-8",
    )

    profile = load_profile_from_yaml(p)

    assert profile.memory.files == ["notes.md"]
    assert profile.memory.folders == []


@pytest.mark.parametrize(
    "files,folders",
    [
        (["a.md"], []),
        ([], ["docs"]),
        (["a.md", "b.txt"], ["x", "y"]),
    ],
)
def test_roundtrip_preserves_various_shapes(tmp_path, files, folders) -> None:
    import yaml

    profile = AgentProfile(name="shape", memory=MemoryConfig(files=files, folders=folders))
    path = tmp_path / "shape.yaml"
    path.write_text(yaml.safe_dump(profile_to_dict(profile)), encoding="utf-8")

    loaded = load_profile_from_yaml(path)

    assert loaded.memory.files == files
    assert loaded.memory.folders == folders

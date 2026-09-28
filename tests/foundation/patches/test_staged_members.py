# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for installing file-patched class members on classes that are already imported."""

from __future__ import annotations

import importlib.util
import logging
import sys
from typing import TYPE_CHECKING, Any, NoReturn

import pytest

from chrys.foundation.patches import staged_members
from chrys.foundation.patches.patcher import FilePatch
from chrys.foundation.patches.staged_members import (
    StagedSourceDriftError,
    install_patched_members,
    members_installed,
    stage_members,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path
    from types import ModuleType

_SOURCE = """\
from functools import cached_property


class Base:
    def greet(self):
        return "base"


class Thing(Base):
    counter = 0

    def greet(self):
        return "old " + super().greet()

    @property
    def size(self):
        return 1

    @cached_property
    def box(self):
        return "unboxed"
"""

_GREET = FilePatch(
    package="sample",
    module_file="sample.py",
    old_fragment='        return "old " + super().greet()',
    new_fragment='        return "new " + super().greet()',
    description="New greeting",
)

_SIZE = FilePatch(
    package="sample",
    module_file="sample.py",
    old_fragment="""\
    @property
    def size(self):
        return 1
""",
    new_fragment="""\
    label = "added"

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        self._size = value

    _size = 2
""",
    description="Settable size",
)

_MEMBERS = {"Thing": ["greet", "size", "counter", "label", "_size"]}

_BOX = FilePatch(
    package="sample",
    module_file="sample.py",
    old_fragment='        return "unboxed"',
    new_fragment="        return Box()",
    description="Boxed",
)

_ADD_BOX = FilePatch(
    package="sample",
    module_file="sample.py",
    old_fragment="class Base:",
    new_fragment="""\
class Box:
    def open(self):
        return "open"


class Base:""",
    description="Box class",
)


@pytest.fixture
def sample(tmp_path: Path) -> Iterator[ModuleType]:
    path = tmp_path / "sample.py"
    path.write_text(_SOURCE, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("chrys_staged_members_sample", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    del sys.modules[spec.name]


def _install(module: ModuleType, patches: list[FilePatch]) -> bool:
    return install_patched_members(module, patches, _MEMBERS, marker="_test_marker", label="sample")


def test_installs_methods_properties_and_new_values_on_the_live_class(sample: Any) -> None:
    thing = sample.Thing()

    assert _install(sample, [_GREET, _SIZE])

    # Existing instances and a zero-argument super() both reach the live classes.
    assert thing.greet() == "new base"
    assert thing.size == 2
    thing.size = 5
    assert thing.size == 5
    assert sample.Thing.label == "added"
    assert members_installed(sample, _MEMBERS, "_test_marker")


def test_a_repeated_install_stages_nothing(sample: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _install(sample, [_GREET, _SIZE])

    def fail(
        _module: ModuleType,
        _patches: Sequence[FilePatch],
        _members: Mapping[str, Sequence[str]],
        *,
        label: str,
        classes: Sequence[str] = (),
    ) -> NoReturn:
        raise AssertionError("an installed patch was staged again")

    monkeypatch.setattr(staged_members, "stage_members", fail)

    assert _install(sample, [_GREET, _SIZE])


def test_a_value_the_class_already_has_keeps_its_live_state(sample: Any) -> None:
    sample.Thing.counter = 5

    assert _install(sample, [_GREET, _SIZE])

    assert sample.Thing.counter == 5
    assert sample.Thing._size == 2


def test_an_already_patched_source_stages_the_same_members(sample: Any, tmp_path: Path) -> None:
    path = tmp_path / "sample.py"
    path.write_text(_SOURCE.replace(_GREET.old_fragment, _GREET.new_fragment), encoding="utf-8")

    assert _install(sample, [_GREET, _SIZE])

    assert sample.Thing().greet() == "new base"


@pytest.mark.parametrize(
    ("patches", "members", "reason"),
    [
        (
            [_GREET, FilePatch("sample", "sample.py", "absent", "present", "Drifted fragment")],
            _MEMBERS,
            "fragment drifted: Drifted fragment",
        ),
        ([_GREET, _SIZE], {"Thing": ["greet", "missing"]}, "Thing is missing ['missing']"),
        ([_GREET, _SIZE], {"Absent": ["greet"]}, "class Absent is missing"),
    ],
    ids=["fragment", "member", "class"],
)
def test_drift_changes_nothing(
    sample: Any,
    caplog: pytest.LogCaptureFixture,
    patches: list[FilePatch],
    members: dict[str, list[str]],
    reason: str,
) -> None:
    greet = sample.Thing.greet
    caplog.set_level(logging.WARNING, logger=staged_members.__name__)

    assert not install_patched_members(sample, patches, members, marker="_test_marker", label="sample")

    assert sample.Thing.greet is greet
    assert "label" not in vars(sample.Thing)
    assert [record.getMessage() for record in caplog.records] == [f"Skipping Textual sample runtime patch: {reason}"]


def test_a_cached_property_is_replaced_and_an_added_class_lands_on_its_module(sample: Any) -> None:
    staged = stage_members(sample, [_ADD_BOX, _BOX], {"Thing": ["box"]}, label="sample", classes=["Box"])

    staged.install("_test_marker")

    assert sample.Thing().box.open() == "open"
    assert sample.Box.__module__ == sample.__name__
    assert members_installed(sample, {"Thing": ["box"]}, "_test_marker")


def test_a_class_the_module_already_has_is_kept(sample: Any) -> None:
    existing = type("Box", (), {"open": lambda self: "kept"})
    sample.Box = existing

    stage_members(sample, [_ADD_BOX, _BOX], {"Thing": ["box"]}, label="sample", classes=["Box"]).install("_test_marker")

    assert sample.Box is existing
    assert sample.Thing().box.open() == "kept"


def test_a_missing_added_class_is_drift(sample: Any) -> None:
    with pytest.raises(StagedSourceDriftError, match="class Absent is missing"):
        stage_members(sample, [_BOX], {"Thing": ["box"]}, label="sample", classes=["Absent"])

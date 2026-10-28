# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Copies of an installed package: what is copied, and a copy taken while a launch patches it."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from chrys.foundation.patches import patcher
from chrys.foundation.platform.files import replace_with_retry
from tests.support import installed_packages
from tests.support.installed_packages import copy_installed_package

_THREAD_WAIT_SECONDS = 10.0


def test_a_copy_taken_while_a_launch_patches_the_package_leaves_out_the_patch_being_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "site" / "chrys_copy_probe"
    (package / "__pycache__").mkdir(parents=True)
    (package / "__pycache__" / "module.cpython-314.pyc").write_bytes(b"")
    (package / "__init__.py").write_text("", encoding="utf-8")
    target = package / "module.py"
    target.write_text("patched = False\n", encoding="utf-8")
    monkeypatch.syspath_prepend(os.fspath(tmp_path / "site"))

    staged = threading.Event()
    finish = threading.Event()

    def held_replace(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        staged.set()
        if not finish.wait(_THREAD_WAIT_SECONDS):
            raise TimeoutError("the copy did not list the package while the patch was written")
        replace_with_retry(source, destination)

    monkeypatch.setattr(patcher, "replace_with_retry", held_replace)
    errors: list[BaseException] = []

    def write_patch() -> None:
        try:
            patcher._atomic_write_text(target, "patched = True\n")
        except BaseException as exc:
            errors.append(exc)
            staged.set()

    writer = threading.Thread(target=write_patch, name="patch-writer")
    skipped = installed_packages._skipped
    listings: list[list[str]] = []

    def listed(directory: str, names: list[str]) -> set[str]:
        if os.path.samefile(directory, package):
            listings.append(list(names))
            # The launch finishes its patch after the copy listed the package, before it copies the files.
            finish.set()
            writer.join(_THREAD_WAIT_SECONDS)
        return skipped(directory, names)

    monkeypatch.setattr(installed_packages, "_skipped", listed)
    writer.start()
    try:
        assert staged.wait(_THREAD_WAIT_SECONDS)
        assert errors == []
        copy = copy_installed_package("chrys_copy_probe", tmp_path / "copy")
    finally:
        finish.set()
        writer.join(_THREAD_WAIT_SECONDS)
    assert not writer.is_alive()
    assert errors == []
    # The copy listed the patch's temporary file, which was gone by the time it copied the files.
    assert [any(name.endswith(".tmp") for name in names) for names in listings] == [True]
    assert sorted(path.name for path in copy.iterdir()) == ["__init__.py", "module.py"]
    assert (copy / "module.py").read_text(encoding="utf-8") == "patched = True\n"


@pytest.mark.parametrize("name", ["chrys_copy_module", "chrys_copy_package.sub"], ids=["module", "subpackage"])
def test_only_a_top_level_package_is_copied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    site = tmp_path / "site"
    (site / "chrys_copy_package" / "sub").mkdir(parents=True)
    (site / "chrys_copy_package" / "__init__.py").write_text("", encoding="utf-8")
    (site / "chrys_copy_package" / "sub" / "__init__.py").write_text("", encoding="utf-8")
    (site / "chrys_copy_module.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(os.fspath(site))

    with pytest.raises(ValueError, match=name):
        copy_installed_package(name, tmp_path / "copy")
    assert not (tmp_path / "copy").exists()

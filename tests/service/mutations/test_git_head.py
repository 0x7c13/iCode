# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Distinguish absent Git references and tree entries from unreadable evidence."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from unittest.mock import create_autospec

import pytest

from chrys.service.mutations import git_state
from chrys.service.mutations.git_state import GitHeadState, git_path_exists, read_git_head, read_git_head_oid

type HeadReader = Callable[[str], str | GitHeadState | None]


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=root, stdin=subprocess.DEVNULL, capture_output=True, check=True).stdout


@pytest.fixture
def repo(tmp_path: Path, git_repo_factory: Callable[[Path], Path]) -> Path:
    return git_repo_factory(tmp_path / "repo")


@pytest.mark.parametrize("detached", [False, True])
def test_head_readers_keep_committed_and_detached_heads(repo: Path, detached: bool) -> None:
    expected = _git(repo, "rev-parse", "HEAD").decode().strip()
    branch = _git(repo, "symbolic-ref", "--short", "HEAD").decode().strip()
    if detached:
        _git(repo, "checkout", "--detach", "HEAD")
    assert read_git_head_oid(str(repo)) == expected
    assert read_git_head(str(repo)) == GitHeadState(expected, None if detached else branch)


def test_head_readers_recognize_a_confirmed_unborn_branch(repo: Path) -> None:
    _git(repo, "symbolic-ref", "HEAD", "refs/heads/unborn")
    assert read_git_head_oid(str(repo)) is None
    assert read_git_head(str(repo)) == GitHeadState(None, "unborn")


@pytest.mark.parametrize("reader", [read_git_head_oid, read_git_head])
def test_transient_head_failure_is_not_an_unborn_repository(
    repo: Path, monkeypatch: pytest.MonkeyPatch, reader: HeadReader
) -> None:
    original = git_state._run_git
    injected = False

    def run(root: str, args: list[str], *, timeout: float) -> subprocess.CompletedProcess[bytes] | None:
        nonlocal injected
        if args[:2] == ["rev-parse", "--verify"] and not injected:
            injected = True
            return subprocess.CompletedProcess(args, 128, b"", b"fatal: cannot read HEAD")
        return original(root, args, timeout=timeout)

    monkeypatch.setattr(git_state, "_run_git", create_autospec(original, side_effect=run))
    with pytest.raises(OSError):
        reader(str(repo))
    assert injected


@pytest.mark.parametrize("reader", [read_git_head_oid, read_git_head])
@pytest.mark.parametrize("contents", ["not an object id\n", "f" * 40 + "\n"])
def test_broken_branch_or_missing_commit_is_not_unborn(repo: Path, reader: HeadReader, contents: str) -> None:
    _git(repo, "symbolic-ref", "HEAD", "refs/heads/broken")
    (repo / ".git" / "refs" / "heads" / "broken").write_text(contents, encoding="ascii")
    with pytest.raises(OSError):
        reader(str(repo))


@pytest.mark.parametrize("reader", [read_git_head_oid, read_git_head])
def test_non_repository_is_not_unborn(tmp_path: Path, reader: HeadReader) -> None:
    with pytest.raises(OSError):
        reader(str(tmp_path))


@pytest.mark.parametrize(
    ("exit_code", "output"),
    [
        (None, b""),
        (128, b"# branch.oid (initial)\n# branch.head unborn\n"),
        (0, b""),
        (0, b"# branch.oid (initial)\n# branch.head (unknown)\n"),
        (0, b"# branch.oid (initial)\n# branch.head other-branch\n"),
        (0, b"# branch.oid abc123\n# branch.head unborn\n"),
    ],
)
def test_unborn_requires_matching_initial_branch_headers(
    repo: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int | None, output: bytes
) -> None:
    _git(repo, "symbolic-ref", "HEAD", "refs/heads/unborn")
    original = git_state._run_git

    def run(root: str, args: list[str], *, timeout: float) -> subprocess.CompletedProcess[bytes] | None:
        if args[0] == "status":
            return None if exit_code is None else subprocess.CompletedProcess(args, exit_code, output, b"probe failed")
        return original(root, args, timeout=timeout)

    monkeypatch.setattr(git_state, "_run_git", create_autospec(original, side_effect=run))
    with pytest.raises(OSError):
        read_git_head_oid(str(repo))


def test_unborn_confirmation_does_not_require_show_ref_exists(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _git(repo, "symbolic-ref", "HEAD", "refs/heads/新分支")
    original = git_state._run_git

    def run(root: str, args: list[str], *, timeout: float) -> subprocess.CompletedProcess[bytes] | None:
        assert args[0] != "show-ref"
        return original(root, args, timeout=timeout)

    monkeypatch.setattr(git_state, "_run_git", create_autospec(original, side_effect=run))
    assert read_git_head_oid(str(repo)) is None


def test_empty_successful_head_output_is_unknown(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = subprocess.CompletedProcess([], 0, b"\n", b"")
    monkeypatch.setattr(git_state, "_run_git", create_autospec(git_state._run_git, return_value=result))
    with pytest.raises(OSError):
        read_git_head_oid(str(repo))


def test_unborn_confirmation_shares_the_head_deadline(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _git(repo, "symbolic-ref", "HEAD", "refs/heads/unborn")
    original = git_state._run_git
    remaining: list[float] = []
    now = 100.0

    def clock() -> float:
        return now

    def run(root: str, args: list[str], *, timeout: float) -> subprocess.CompletedProcess[bytes] | None:
        nonlocal now
        remaining.append(timeout)
        result = original(root, args, timeout=timeout)
        now += 0.25
        return result

    monkeypatch.setattr(git_state.time, "monotonic", create_autospec(git_state.time.monotonic, side_effect=clock))
    monkeypatch.setattr(git_state, "_run_git", create_autospec(original, side_effect=run))
    assert read_git_head_oid(str(repo), timeout=1.0) is None
    assert remaining == [1.0, 0.75, 0.5]


def test_tree_entry_absence_requires_a_successful_tree_read(repo: Path) -> None:
    assert git_path_exists(str(repo), "HEAD", "README.md") is True
    assert git_path_exists(str(repo), "HEAD", "missing.txt") is False
    assert git_path_exists(str(repo), "f" * 40, "README.md") is None


@pytest.mark.parametrize("exit_code", [None, 1, 128])
def test_tree_entry_read_failures_are_unknown(
    repo: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int | None
) -> None:
    result = None if exit_code is None else subprocess.CompletedProcess([], exit_code, b"", b"read failed")
    monkeypatch.setattr(git_state, "_run_git", create_autospec(git_state._run_git, return_value=result))
    assert git_path_exists(str(repo), "HEAD", "README.md") is None

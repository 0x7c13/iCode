# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for SubprocessScriptRunner — real subprocess execution.

Script paths are absolute (validated by the chrys loader at discovery time)
and the runner also re-checks containment before execution because it is the
final boundary.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from typing import TYPE_CHECKING

import pytest

from chrys.foundation.platform import runtime_paths
from chrys.foundation.text.tokenizer import MixedLanguageTokenizer
from chrys.kernel.tools import SyncToolCancelledAfterCompletion
from chrys.service.skills import runner as runner_mod
from chrys.service.skills.loader import load_file_skill
from chrys.service.skills.model import Skill, SkillScript
from chrys.service.skills.runner import SubprocessScriptRunner
from chrys.service.tools.spill import TOOL_RESULTS_DIR_NAME

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> SubprocessScriptRunner:
    monkeypatch.setattr(runner_mod, "_find_python_runner", lambda: [sys.executable])
    return SubprocessScriptRunner(timeout=30)


def _make_skill(tmp_path: Path, script_name: str, script_content: str) -> tuple[Skill, SkillScript]:
    """Create a skill directory with a single Python script.

    Builds a :class:`Skill` rooted at ``tmp_path/test-skill`` and a
    :class:`SkillScript` whose ``full_path`` points at the on-disk
    script file (the runner requires the script path to be absolute).
    """
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir(exist_ok=True)
    script_path = skill_dir / script_name
    script_path.write_text(script_content, encoding="utf-8")

    skill = Skill(name="test-skill", description="test skill", content="body", path=str(skill_dir))
    script = SkillScript(name=script_name, full_path=str(script_path))
    return skill, script


def _exe(path: Path) -> Path:
    if sys.platform == "win32" and not path.suffix:
        return path.with_name(path.name + ".exe")
    return path


def _make_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    path.chmod(0o755)
    return path


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    python: Path,
    prefix: Path,
    scripts: Path,
) -> None:
    monkeypatch.setattr(runtime_paths.sys, "executable", str(python))
    monkeypatch.setattr(runtime_paths.sys, "prefix", str(prefix))
    monkeypatch.setattr(runtime_paths.sys, "exec_prefix", str(prefix))
    monkeypatch.setattr(
        runtime_paths.sysconfig,
        "get_path",
        lambda name: str(scripts) if name == "scripts" else "",
    )


def test_find_python_runner_prefers_system_uv_when_frozen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_bin = tmp_path / "runtime" / "bin"
    system_bin = tmp_path / "system" / "bin"
    _make_executable(_exe(runtime_bin / "uv"))
    system_uv = _make_executable(_exe(system_bin / "uv"))
    _patch_runtime(
        monkeypatch,
        python=_exe(runtime_bin / "python"),
        prefix=tmp_path / "runtime",
        scripts=runtime_bin,
    )
    if sys.platform == "win32":
        monkeypatch.setenv("PATHEXT", ".exe")
    monkeypatch.setenv("PYAPP", "1")
    monkeypatch.setenv("PATH", os.pathsep.join([str(runtime_bin), str(system_bin)]))

    assert runner_mod._find_python_runner() == [str(system_uv), "run"]


def test_find_python_runner_prefers_system_python_over_runtime_uv_when_frozen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_bin = tmp_path / "runtime" / "bin"
    system_bin = tmp_path / "system" / "bin"
    _make_executable(_exe(runtime_bin / "uv"))
    system_python = _make_executable(_exe(system_bin / "python"))
    _patch_runtime(
        monkeypatch,
        python=_exe(runtime_bin / "python"),
        prefix=tmp_path / "runtime",
        scripts=runtime_bin,
    )
    if sys.platform == "win32":
        monkeypatch.setenv("PATHEXT", ".exe")
    monkeypatch.setenv("PYAPP", "1")
    monkeypatch.setenv("PATH", os.pathsep.join([str(runtime_bin), str(system_bin)]))

    assert runner_mod._find_python_runner() == [str(system_python)]


async def test_skill_subprocess_env_demotes_runtime_path_when_frozen(
    runner: SubprocessScriptRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_bin = tmp_path / "runtime" / "bin"
    system_bin = tmp_path / "system" / "bin"
    _patch_runtime(
        monkeypatch,
        python=_exe(runtime_bin / "python"),
        prefix=tmp_path / "runtime",
        scripts=runtime_bin,
    )
    monkeypatch.setenv("PYAPP", "1")
    monkeypatch.setenv("PATH", os.pathsep.join([str(runtime_bin), str(system_bin)]))
    skill, script = _make_skill(tmp_path, "env_probe.py", "print('ok')")
    captured_env: dict[str, str] = {}

    @contextlib.asynccontextmanager
    async def fake_managed_subprocess(*_cmd: object, **kwargs: object):
        captured_env.update(kwargs["env"])

        class _Proc:
            returncode = 0

            async def communicate(self) -> tuple[bytes, bytes]:
                return b"ok\n", b""

        yield _Proc()

    monkeypatch.setattr(runner_mod, "managed_subprocess", fake_managed_subprocess)

    result = await runner(skill, script)

    assert result == "ok"
    assert captured_env["PATH"].split(os.pathsep)[:2] == [str(system_bin), str(runtime_bin)]


async def test_skill_subprocess_env_strips_inherited_pythonhome_but_preserves_pythonpath(
    runner: SubprocessScriptRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONHOME", "/bad/home")
    monkeypatch.setenv("PYTHONPATH", "/shared/helpers")
    skill, script = _make_skill(tmp_path, "env_probe.py", "print('ok')")
    captured_env: dict[str, str] = {}

    @contextlib.asynccontextmanager
    async def fake_managed_subprocess(*_cmd: object, **kwargs: object):
        captured_env.update(kwargs["env"])

        class _Proc:
            returncode = 0

            async def communicate(self) -> tuple[bytes, bytes]:
                return b"ok\n", b""

        yield _Proc()

    monkeypatch.setattr(runner_mod, "managed_subprocess", fake_managed_subprocess)

    result = await runner(skill, script)

    assert result == "ok"
    env_keys = {key.upper() for key in captured_env}
    assert "PYTHONHOME" not in env_keys
    assert captured_env["PYTHONPATH"] == "/shared/helpers"


async def test_returns_error_when_script_file_missing(runner: SubprocessScriptRunner, tmp_path: Path) -> None:
    """Script discovery validated full_path at construction, but the file may be deleted later."""
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    placeholder = skill_dir / "placeholder.py"
    placeholder.write_text("", encoding="utf-8")

    skill = Skill(name="test", description="test", content="body", path=str(skill_dir))
    script = SkillScript(name="missing.py", full_path=str(skill_dir / "missing.py"))
    result = await runner(skill, script)
    assert "not found" in result


async def test_rejects_resolved_script_path_outside_skill_dir(
    runner: SubprocessScriptRunner,
    tmp_path: Path,
) -> None:
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text('print("escaped")\n', encoding="utf-8")

    skill = Skill(name="test", description="test", content="body", path=str(skill_dir))
    script = SkillScript(name="outside.py", full_path=str(outside))

    result = await runner(skill, script)

    assert result.startswith("Error:")
    assert "escapes skill directory" in result
    assert "escaped" not in result


async def test_runs_python_script(runner: SubprocessScriptRunner, tmp_path: Path) -> None:
    skill, script = _make_skill(tmp_path, "hello.py", 'print("hello world")')
    result = await runner(skill, script)
    assert "hello world" in result


async def test_loader_script_can_read_unlisted_binary_and_deep_files(
    runner: SubprocessScriptRunner,
    tmp_path: Path,
) -> None:
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: test skill\n---\n\nBody.\n",
        encoding="utf-8",
    )
    script_path = skill_dir / "run.py"
    script_path.write_text(
        """\
from pathlib import Path

root = Path(__file__).parent
payload = (root / "payload.bin").read_bytes().decode("utf-8")
query = (root / "assets" / "nested" / "query.sql").read_text(encoding="utf-8")
has_skill = "name: test-skill" in (root / "SKILL.md").read_text(encoding="utf-8")
print(f"{payload}|{query}|{has_skill}")
""",
        encoding="utf-8",
    )
    (skill_dir / "payload.bin").write_bytes(b"binary payload")
    helper_dir = skill_dir / "assets" / "nested"
    helper_dir.mkdir(parents=True)
    (helper_dir / "query.sql").write_text("SELECT 1", encoding="utf-8")
    skill = load_file_skill(str(skill_dir), script_extensions=(".py",), search_depth=1)

    assert isinstance(skill, Skill)
    assert skill.resources == []
    assert [script.name for script in skill.scripts] == ["run.py"]

    result = await runner(skill, skill.scripts[0])

    assert result == "binary payload|SELECT 1|True"


async def test_stdin_is_devnull(runner: SubprocessScriptRunner, tmp_path: Path) -> None:
    """Skill scripts must not inherit the parent's stdin handle.

    On Windows this prevents a child TUI from calling ``SetConsoleMode``
    on the outer chrys's console input.  Here we just verify the
    observable: ``sys.stdin.read()`` returns immediately with an empty
    string (DEVNULL), instead of blocking or reading test-runner input.
    """
    skill, script = _make_skill(
        tmp_path,
        "stdin_probe.py",
        "import sys\nprint('stdin_empty=' + str(len(sys.stdin.read()) == 0))",
    )
    result = await runner(skill, script)
    assert "stdin_empty=True" in result


async def test_captures_stderr(runner: SubprocessScriptRunner, tmp_path: Path) -> None:
    skill, script = _make_skill(tmp_path, "warn.py", 'import sys; sys.stderr.write("warning\\n")')
    result = await runner(skill, script)
    assert "[stderr]" in result
    assert "warning" in result


async def test_reports_nonzero_exit(runner: SubprocessScriptRunner, tmp_path: Path) -> None:
    skill, script = _make_skill(tmp_path, "fail.py", "import sys; sys.exit(42)")
    result = await runner(skill, script)
    assert "[exit_code: 42]" in result


async def test_converts_args_to_flags(runner: SubprocessScriptRunner, tmp_path: Path) -> None:
    skill, script = _make_skill(
        tmp_path,
        "args.py",
        """\
import argparse
p = argparse.ArgumentParser()
p.add_argument("--name")
p.add_argument("--count", type=int)
a = p.parse_args()
print(f"{a.name} {a.count}")
""",
    )
    result = await runner(skill, script, args={"name": "alice", "count": 3})
    assert "alice 3" in result


async def test_arguments_precede_args_for_subcommand_flags(
    runner: SubprocessScriptRunner,
    tmp_path: Path,
) -> None:
    skill, script = _make_skill(
        tmp_path,
        "subcommand.py",
        """\
import argparse
p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="cmd", required=True)
log = sub.add_parser("log")
log.add_argument("--oneline", action="store_true")
log.add_argument("--max-count", type=int)
a = p.parse_args()
print(f"{a.cmd} {a.oneline} {a.max_count}")
""",
    )

    result = await runner(script=script, skill=skill, arguments=["log"], args={"oneline": True, "max-count": 5})

    assert "log True 5" in result


async def test_dash_prefixed_arg_keys_pass_through(
    runner: SubprocessScriptRunner,
    tmp_path: Path,
) -> None:
    skill, script = _make_skill(
        tmp_path,
        "short_flags.py",
        """\
import sys
print("|".join(sys.argv[1:]))
""",
    )

    result = await runner(skill, script, args={"-i": True, "-xzf": True})

    assert "-i|-xzf" in result


async def test_list_args_are_treated_as_positional_arguments(
    runner: SubprocessScriptRunner,
    tmp_path: Path,
) -> None:
    skill, script = _make_skill(
        tmp_path,
        "positional.py",
        """\
import sys
print("|".join(sys.argv[1:]))
""",
    )

    result = await runner(skill, script, args=["input.txt", "output.txt"])

    assert "input.txt|output.txt" in result


async def test_list_value_expands_to_nargs(runner: SubprocessScriptRunner, tmp_path: Path) -> None:
    skill, script = _make_skill(
        tmp_path,
        "nargs.py",
        """\
import argparse
p = argparse.ArgumentParser()
p.add_argument("--key", nargs="+")
a = p.parse_args()
print("|".join(a.key))
""",
    )
    result = await runner(skill, script, args={"key": ["1", "2", "3"]})
    assert "1|2|3" in result


async def test_empty_list_omits_flag(runner: SubprocessScriptRunner, tmp_path: Path) -> None:
    skill, script = _make_skill(
        tmp_path,
        "empty_list.py",
        """\
import argparse
p = argparse.ArgumentParser()
p.add_argument("--key", nargs="+", default=["fallback"])
a = p.parse_args()
print("|".join(a.key))
""",
    )
    # Empty list should be skipped entirely — script sees its default,
    # not a bare ``--key`` (which would error under nargs='+').
    result = await runner(skill, script, args={"key": []})
    assert "fallback" in result


async def test_bool_true_becomes_flag(runner: SubprocessScriptRunner, tmp_path: Path) -> None:
    skill, script = _make_skill(
        tmp_path,
        "flag.py",
        """\
import argparse
p = argparse.ArgumentParser()
p.add_argument("--verbose", action="store_true")
a = p.parse_args()
print(f"verbose={a.verbose}")
""",
    )
    result = await runner(skill, script, args={"verbose": True})
    assert "verbose=True" in result


async def test_bool_false_not_added(runner: SubprocessScriptRunner, tmp_path: Path) -> None:
    skill, script = _make_skill(
        tmp_path,
        "flag2.py",
        """\
import argparse
p = argparse.ArgumentParser()
p.add_argument("--verbose", action="store_true")
a = p.parse_args()
print(f"verbose={a.verbose}")
""",
    )
    result = await runner(skill, script, args={"verbose": False})
    assert "verbose=False" in result


async def test_none_value_not_added(runner: SubprocessScriptRunner, tmp_path: Path) -> None:
    skill, script = _make_skill(tmp_path, "noop.py", 'print("ok")')
    result = await runner(skill, script, args={"key": None})
    assert "ok" in result


async def test_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_mod, "_find_python_runner", lambda: [sys.executable])
    runner = SubprocessScriptRunner(timeout=1)
    skill, script = _make_skill(tmp_path, "slow.py", "import time; time.sleep(5)")
    result = await runner(skill, script)
    assert "timed out" in result


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX stopped state only")
async def test_stopped_script_returns_promptly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_mod, "_find_python_runner", lambda: [sys.executable])
    runner = SubprocessScriptRunner(timeout=10)
    skill, script = _make_skill(
        tmp_path,
        "stopped.py",
        "import os, signal, time\nos.kill(os.getpid(), signal.SIGSTOP)\ntime.sleep(30)\n",
    )

    started = time.monotonic()
    result = await runner(skill, script)

    assert "stopped state" in result
    assert time.monotonic() - started < 5


async def test_no_output(runner: SubprocessScriptRunner, tmp_path: Path) -> None:
    skill, script = _make_skill(tmp_path, "empty.py", "")
    result = await runner(skill, script)
    assert result == "(no output)"


async def test_large_output_is_cleaned_truncated_and_spilled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_mod, "_find_python_runner", lambda: [sys.executable])
    runner = SubprocessScriptRunner(timeout=30, session_dir=tmp_path)
    skill, script = _make_skill(
        tmp_path,
        "large.py",
        """\
import sys
sys.stdout.write("\\x1b[31mstart\\x1b[0m\\rfinal:" + "x" * 10000 + "\\n")
sys.stderr.write("important stderr\\n")
raise SystemExit(7)
""",
    )

    result = await runner(skill, script, max_tokens=100)

    assert "truncated" in result
    assert "Full output saved to:" in result
    assert MixedLanguageTokenizer().count_tokens(result) <= 100
    spills = list((tmp_path / TOOL_RESULTS_DIR_NAME).glob("skill_*.txt"))
    assert len(spills) == 1
    full = spills[0].read_text()
    assert "\x1b" not in full
    assert "start" not in full
    assert full.startswith("final:")
    assert "[stderr]\nimportant stderr" in full
    assert full.endswith("[exit_code: 7]")


async def test_script_max_tokens_zero_clamps_to_one_hundred(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_mod, "_find_python_runner", lambda: [sys.executable])
    skill, script = _make_skill(tmp_path, "large.py", 'print("x" * 10000)')

    result = await SubprocessScriptRunner(timeout=30)(skill, script, max_tokens=0)

    assert "truncated" in result
    assert MixedLanguageTokenizer().count_tokens(result) <= 100
    assert "Full output saved to:" not in result


async def test_completed_script_result_propagates_spill_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_mod, "_find_python_runner", lambda: [sys.executable])
    completed = "[bounded completed skill result]"

    async def cancelled_after_completion(*_args: object) -> str:
        raise SyncToolCancelledAfterCompletion(completed)

    monkeypatch.setattr(runner_mod, "truncate_with_spill", cancelled_after_completion)
    skill, script = _make_skill(tmp_path, "large.py", 'print("x" * 10000)')

    with pytest.raises(SyncToolCancelledAfterCompletion) as exc_info:
        await SubprocessScriptRunner(timeout=30, session_dir=tmp_path)(skill, script, max_tokens=100)

    assert exc_info.value.completed_result == completed


# ---------------------------------------------------------------------------
# CWD resolution
# ---------------------------------------------------------------------------


def _cwd_script() -> str:
    """Script body that prints its own os.getcwd()."""
    return "import os; print(os.getcwd())"


async def test_cwd_defaults_to_runtime_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When no cwd is passed, the subprocess runs in runtime.cwd."""
    import dataclasses

    from chrys.foundation.models.session_env import SessionEnvironment

    work_dir = tmp_path / "user-workspace"
    work_dir.mkdir()

    monkeypatch.setattr(runner_mod, "_find_python_runner", lambda: [sys.executable])
    runtime = dataclasses.replace(SessionEnvironment.capture(), cwd=str(work_dir))

    runner = SubprocessScriptRunner(timeout=30, runtime=runtime)
    skill, script = _make_skill(tmp_path, "cwd.py", _cwd_script())
    result = await runner(skill, script)
    assert str(work_dir) in result


async def test_explicit_absolute_cwd_is_honored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An absolute cwd supplied by the agent overrides the default."""
    monkeypatch.setattr(runner_mod, "_find_python_runner", lambda: [sys.executable])
    runner = SubprocessScriptRunner(timeout=30)
    alt_dir = tmp_path / "elsewhere"
    alt_dir.mkdir()

    skill, script = _make_skill(tmp_path, "cwd.py", _cwd_script())
    result = await runner(skill, script, cwd=str(alt_dir))
    assert str(alt_dir) in result


async def test_relative_cwd_is_rejected(tmp_path: Path) -> None:
    runner = SubprocessScriptRunner(timeout=30)
    skill, script = _make_skill(tmp_path, "cwd.py", _cwd_script())
    result = await runner(skill, script, cwd="relative/path")
    assert result.startswith("Error:")
    assert "absolute path" in result


async def test_nonexistent_cwd_is_rejected(tmp_path: Path) -> None:
    runner = SubprocessScriptRunner(timeout=30)
    skill, script = _make_skill(tmp_path, "cwd.py", _cwd_script())
    missing = tmp_path / "does-not-exist"
    result = await runner(skill, script, cwd=str(missing))
    assert result.startswith("Error:")
    assert "not an existing directory" in result


async def test_cwd_falls_back_to_script_parent_without_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a runtime and without an explicit cwd, use script_path.parent (legacy)."""
    monkeypatch.setattr(runner_mod, "_find_python_runner", lambda: [sys.executable])
    runner = SubprocessScriptRunner(timeout=30)  # no runtime bound
    skill, script = _make_skill(tmp_path, "cwd.py", _cwd_script())
    result = await runner(skill, script)
    assert str(tmp_path / "test-skill") in result

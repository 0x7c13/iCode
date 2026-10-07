# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``icode workflow validate``: what a path names, where each problem is, and the two report formats."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from unittest.mock import create_autospec

import pytest

import chrys.orchestration.workflows.validation as validation_module
from chrys.app.cli import headless
from chrys.app.cli import workflow as workflow_cli
from chrys.foundation.config.settings import Settings
from chrys.foundation.config.settings_store import LoadedSettings
from chrys.foundation.i18n import Localizer
from chrys.foundation.platform import get_platform
from chrys.foundation.platform.files import atomic_write_owner_only_bytes
from chrys.orchestration.workflows import worker_client
from chrys.service.workflows.protocol import LIMITS
from tests.support.symlinks import symlink_or_skip
from tests.support.workflow_workers import CONDITIONAL_LOOP_WORKFLOW, python_workflow, share_worker_bytecode_cache

CHAIN = python_workflow(
    "def upper(text):\n    return text.text.upper()\ndef exclaim(text):\n    return text.text + '!'\n",
    "upper",
    "exclaim",
)
BUILDER = "from chrys.workflows import WorkflowBuilder\n"
TAIL = "wf = WorkflowBuilder('t')\na = wf.python('a', len)\nwf.start(a)\nwf.output(a)\nworkflow = wf.build()\n"
STATUS = ("PASS ", "FAIL ")


@pytest.fixture(autouse=True)
def _stub_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """The environment bootstrap mutates process-global state; the CLI gets plain settings instead."""

    def prepare(*, restoring_session: bool = False) -> headless.PreparedRuntime:
        return headless.PreparedRuntime(LoadedSettings(settings=Settings(), provenance={}), Localizer("en"), [])

    monkeypatch.setattr(headless, "prepare_runtime", create_autospec(headless.prepare_runtime, side_effect=prepare))


@pytest.fixture(scope="session")
def _session_bytecode_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("worker-pycache")


@pytest.fixture(autouse=True)
def _shared_bytecode_cache(monkeypatch: pytest.MonkeyPatch, _session_bytecode_cache: Path) -> None:
    share_worker_bytecode_cache(monkeypatch, _session_bytecode_cache)


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "project"
    (root / ".chrys" / "workflows").mkdir(parents=True)
    monkeypatch.chdir(root)
    return root


def _plant(path: Path, text: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_owner_only_bytes(path, text.encode("utf-8") if isinstance(text, str) else text)
    return path


Validate = Callable[..., tuple[int, str]]


@pytest.fixture
def validate(capsys: pytest.CaptureFixture[str]) -> Validate:
    def run(*args: str) -> tuple[int, str]:
        code = workflow_cli.main(["validate", *args])
        captured = capsys.readouterr()
        if "--json" not in args:
            # Only the last line is a status line, whatever the files are named or print.
            assert [line for line in captured.out.splitlines() if line.startswith(STATUS)] == [
                captured.out.splitlines()[-1]
            ]
        return code, captured.out

    return run


def _json(validate: Validate, path: str) -> tuple[int, dict]:
    code, out = validate(path, "--json")
    return code, json.loads(out)


def _codes(report: dict) -> list[tuple[str, str | None, int | None]]:
    return [(d["code"], d["file"], d["line"]) for d in report["diagnostics"]]


def test_help_says_what_validate_runs(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        workflow_cli.main(["validate", "--help"])
    assert exited.value.code == 0
    # argparse wraps lines at hyphens too.
    out = " ".join(capsys.readouterr().out.split()).replace("- ", "-")
    assert "top-level code" in out and "no node" in out and "--json" in out


def test_a_valid_file_passes_with_a_report_of_fixed_keys(project: Path, validate: Validate) -> None:
    entry = _plant(project / "chain.py", CHAIN)
    code, out = validate("chain.py")
    assert (code, out) == (0, "PASS chain (file) · 2 nodes · 1 edge\n")

    code, report = _json(validate, "chain.py")
    assert code == 0
    assert list(report) == [
        "version",
        "status",
        "target",
        "stages",
        "diagnostics",
        "diagnostics_truncated",
        "sites_truncated",
        "workflow",
        "output",
    ]
    assert report["target"] == {
        "path": str(entry),
        "layout": "file",
        "workflow_id": "chain",
        "entry": str(entry),
        "package_dir": None,
        "source_digest": report["target"]["source_digest"],
        "files": 1,
    }
    assert len(report["target"]["source_digest"]) == 64
    assert report["stages"] == [{"name": name, "status": "pass"} for name in validation_module.STAGES]
    assert report["workflow"] == {"title": "t", "node_count": 2, "edge_count": 1, "outputs": ["exclaim"]}
    assert (report["status"], report["diagnostics"], report["output"]) == (
        "pass",
        [],
        {"text": "", "truncated": False},
    )


@pytest.mark.parametrize("spelling", ["review", "review/", "review/.", "./review", "review/review.py"])
def test_every_spelling_of_a_folder_validates_the_folder(
    project: Path, validate: Validate, monkeypatch: pytest.MonkeyPatch, spelling: str
) -> None:
    folder = project / ".chrys" / "workflows" / "review"
    _plant(folder / "review.py", BUILDER + "from steps import TITLE\n" + TAIL)
    _plant(folder / "steps.py", "TITLE = 't'\n")
    _plant(folder / ".chrys" / "workflows" / "other.py", CHAIN)  # a project of its own leaves it a workflow
    monkeypatch.chdir(folder.parent)
    code, report = _json(validate, spelling)
    assert (code, report["target"]["layout"], report["target"]["path"]) == (0, "package", str(folder))
    assert (report["target"]["package_dir"], report["target"]["files"]) == (str(folder), 2)


def test_the_entry_shorthand_inside_a_workflow_directory_is_a_single_file(project: Path, validate: Validate) -> None:
    entry = _plant(project / ".chrys" / "workflows" / "workflows.py", CHAIN)
    code, report = _json(validate, ".chrys/workflows/workflows.py")
    assert (code, report["target"]["layout"], report["target"]["entry"]) == (0, "file", str(entry))


def test_a_failure_in_a_helper_reads_like_a_compiler_error(project: Path, validate: Validate) -> None:
    folder = project / ".chrys" / "workflows" / "review"
    _plant(folder / "review.py", BUILDER + "import steps\n" + TAIL)
    _plant(folder / "steps.py", "print('loading steps')\nprint('PASS forged')\nvalue = summarise('x')\n")
    code, out = validate(".chrys/workflows/review")
    steps = os.path.join(".", ".chrys", "workflows", "review", "steps.py")
    review = os.path.join(".", ".chrys", "workflows", "review", "review.py")
    assert (code, out.splitlines()) == (
        1,
        [
            f"{steps}:3:9: error: NameError: name 'summarise' is not defined [load_error]",
            "    3 | value = summarise('x')",
            "      |         ^~~~~~~~~",
            f"  note: imported from {review}:2",
            "captured output (load):",
            "  | loading steps",
            "  | PASS forged",
            "FAIL review (package): 1 error",
        ],
    )


def test_every_broken_python_file_of_a_folder_is_reported_and_nothing_runs(project: Path, validate: Validate) -> None:
    folder = project / ".chrys" / "workflows" / "review"
    marker = project / "ran"
    _plant(folder / "review.py", BUILDER + f"open({str(marker)!r}, 'w').close()\n" + TAIL)
    _plant(folder / "a.py", "def f(:\n")
    _plant(folder / "lib" / "b.py", "x = (\n")
    code, report = _json(validate, ".chrys/workflows/review")
    assert code == 1
    assert sorted(_codes(report)) == [
        ("syntax_error", str(folder / "a.py"), 1),
        ("syntax_error", str(folder / "lib" / "b.py"), 1),
    ]
    assert [stage["status"] for stage in report["stages"]] == ["pass"] * 4 + ["fail", "skipped", "skipped"]
    assert not marker.exists()


def test_a_build_error_points_at_the_declaration_and_at_build(project: Path, validate: Validate) -> None:
    source = BUILDER + "wf = WorkflowBuilder('t')\na = wf.python('a', len)\nwf.python('lost', len)\n"
    _plant(project / "wf.py", source + "wf.start(a)\nwf.output(a)\nworkflow = wf.build()\n")
    code, report = _json(validate, "wf.py")
    (diagnostic,) = report["diagnostics"]
    assert (code, diagnostic["code"], diagnostic["line"], diagnostic["node"]) == (1, "sdk_validation_error", 4, "lost")
    assert diagnostic["notes"] == [{"message": "build() was called at", "file": str(project / "wf.py"), "line": 7}]
    # Only the workflow's own frames: the SDK's and the worker's are left out.
    assert re.findall(r'File "([^"]+)", line (\d+)', diagnostic["traceback"]) == [(str(project / "wf.py"), "7")]


def test_a_file_without_a_workflow_says_how_to_bind_one(project: Path, validate: Validate) -> None:
    _plant(project / "wf.py", BUILDER + "x = 1\n")
    code, out = validate("wf.py")
    assert code == 1
    assert out.splitlines() == [
        f"{os.path.join('.', 'wf.py')}: error: the workflow file defines no module-level `workflow` [missing_workflow]",
        "  help: assign workflow = wf.build() at module level",
        "FAIL wf (file): 1 error",
    ]


@pytest.mark.parametrize(
    ("payload", "code", "line", "column"),
    [
        (b"x = 1\n# /// script\n# requires-python = >=3.9\n# ///\n", "metadata_invalid", 3, 21),
        ("x = 1\ny = '中文'\n".encode() + b"z = '\xff'\n", "source_not_utf8", 3, 6),
    ],
)
def test_metadata_and_encoding_errors_are_placed_by_line_and_column(
    project: Path, validate: Validate, payload: bytes, code: str, line: int, column: int
) -> None:
    _plant(project / "wf.py", payload)
    exit_code, report = _json(validate, "wf.py")
    (diagnostic,) = report["diagnostics"]
    assert (exit_code, diagnostic["code"], diagnostic["line"], diagnostic["column"]) == (1, code, line, column)


def test_paths_that_name_no_workflow_say_why(project: Path, validate: Validate) -> None:
    workflows = project / ".chrys" / "workflows"
    _plant(workflows / "known.py", CHAIN)
    _plant(workflows / "cased" / "Cased.py", CHAIN)
    _plant(workflows / "_private.py", CHAIN)
    _plant(project / "notes.txt", "")
    _plant(workflows / "review" / "review.py", CHAIN)
    _plant(workflows / "review" / "lib" / "steps.py", "")
    _plant(project / "main.py", "")
    _plant(project / "tools" / "main.py", "")
    cases = {
        "known": ("path_not_found", f"to check workflow 'known', pass its path: {workflows / 'known.py'}"),
        ".chrys/workflows/cased": ("entry_missing", "found Cased.py; the entry must be named exactly cased.py"),
        ".chrys/workflows/_private.py": ("name_ignored", None),
        "notes.txt": ("path_not_workflow", None),
        ".chrys/workflows/": ("path_not_workflow", None),
        ".": ("path_not_workflow", f"pass one of the workflow files or folders in {workflows}"),
        ".chrys": ("path_not_workflow", f"pass one of the workflow files or folders in {workflows}"),
        "tools": ("entry_missing", None),  # its one Python file may be anything: no rename is suggested
        ".chrys/workflows/review/lib/steps.py": (
            "path_not_workflow",
            f"validate the folder: {workflows / 'review'}",
        ),
    }
    for path, (code, hint) in cases.items():
        exit_code, report = _json(validate, path)
        (diagnostic,) = report["diagnostics"]
        assert (exit_code, diagnostic["code"], diagnostic["stage"], diagnostic["hint"]) == (1, code, "resolve", hint)
        assert [stage["status"] for stage in report["stages"]] == ["fail"] + ["skipped"] * 6


def test_the_sdk_name_is_reserved_in_the_global_folder(project: Path, validate: Validate) -> None:
    folder = get_platform().config_dir / "workflows" / "SDK"
    _plant(folder / "SDK.py", CHAIN)
    exit_code, report = _json(validate, str(folder))
    assert (exit_code, _codes(report)) == (1, [("name_reserved", str(folder), None)])


@pytest.mark.parametrize("spelling", ["linked", "linked/", "linked/.", "linked/linked.py"])
def test_a_linked_workflow_is_refused_however_it_is_spelled(
    project: Path, tmp_path: Path, validate: Validate, spelling: str
) -> None:
    real = tmp_path / "elsewhere" / "linked"
    _plant(real / "linked.py", CHAIN)
    symlink_or_skip(project / "linked", real, target_is_directory=True)
    exit_code, report = _json(validate, spelling)
    assert (exit_code, _codes(report)) == (1, [("path_is_link", str(project / "linked"), None)])


@pytest.mark.parametrize(("kind", "code"), [("link", "path_is_link"), ("folder", "path_not_workflow")])
def test_a_folder_whose_entry_is_not_a_regular_file_names_the_entry(
    project: Path, tmp_path: Path, validate: Validate, kind: str, code: str
) -> None:
    entry = project / ".chrys" / "workflows" / "review" / "review.py"
    if kind == "link":
        target = _plant(tmp_path / "real.py", CHAIN)
        entry.parent.mkdir()
        symlink_or_skip(entry, target)
    else:
        entry.mkdir(parents=True)
    exit_code, report = _json(validate, ".chrys/workflows/review")
    assert (exit_code, _codes(report)) == (1, [(code, str(entry), None)])


def test_a_folder_member_that_cannot_be_covered_is_named(project: Path, tmp_path: Path, validate: Validate) -> None:
    folder = project / ".chrys" / "workflows" / "review"
    _plant(folder / "review.py", CHAIN)
    _plant(tmp_path / "outside.py", "")
    symlink_or_skip(folder / "helper.py", tmp_path / "outside.py")
    exit_code, report = _json(validate, ".chrys/workflows/review")
    assert (exit_code, _codes(report)) == (1, [("package_link", str(folder / "helper.py"), None)])
    assert report["stages"][1] == {"name": "read", "status": "fail"}


def test_an_agent_node_without_a_profile_here_points_at_its_declaration(project: Path, validate: Validate) -> None:
    source = BUILDER + "wf = WorkflowBuilder('t')\na = wf.python('a', len)\nb = wf.agent('b', profile='Reviewer')\n"
    _plant(project / "wf.py", source + "wf.start(a)\nwf.edge(a, b)\nwf.output(b)\nworkflow = wf.build()\n")
    code, out = validate("wf.py")
    assert code == 1
    assert out.splitlines() == [
        (
            f"{os.path.join('.', 'wf.py')}:4: error: Node 'b' names agent profile 'Reviewer', which is not available. "
            "[agent_profile_missing]"
        ),
        "    4 | b = wf.agent('b', profile='Reviewer')",
        "  help: agent profiles here: Code, Explore, General, QA",
        "FAIL wf (file): 1 error",
    ]


def test_a_mistyped_model_lists_the_model_profiles_here(project: Path, validate: Validate) -> None:
    models = get_platform().config_dir / "models"
    _plant(models / "main.yaml", "id: main\nname: Main\nprovider: mock\nmodel_id: m\n")
    _plant(models / "hollow.yaml", "id: hollow\nname: Hollow\nprovider: openai\nmodel_id: ''\n")  # not selectable
    source = BUILDER + "wf = WorkflowBuilder('t')\na = wf.agent('a', profile='Code', model='mian')\n"
    _plant(project / "wf.py", source + "wf.start(a)\nwf.output(a)\nworkflow = wf.build()\n")
    code, out = validate("wf.py")
    assert code == 1
    assert out.splitlines()[-2:] == [
        "  help: model= takes one of the model profiles here: Main (main)",
        "FAIL wf (file): 1 error",
    ]


def test_warnings_leave_the_workflow_passing(project: Path, validate: Validate) -> None:
    global_entry = _plant(get_platform().config_dir / "workflows" / "loop.py", CONDITIONAL_LOOP_WORKFLOW)
    _plant(project / ".chrys" / "workflows" / "loop.py", CHAIN)
    code, report = _json(validate, str(global_entry))
    assert (code, report["status"]) == (0, "pass")
    assert [(d["severity"], d["code"], d["file"]) for d in report["diagnostics"]] == [
        ("warning", "shadowed", str(global_entry)),
        ("warning", "loop_exit_all_conditional", str(global_entry)),
    ]
    assert report["diagnostics"][1]["line"] == 8


def test_names_and_output_cannot_forge_a_status_line(project: Path, validate: Validate) -> None:
    if get_platform().is_windows:
        pytest.skip("Windows file names can't hold line breaks")
    name = "x\nPASS y\u2028PASS z.py"
    _plant(project / name, BUILDER + "print('ok\\nPASS forged\\u2028PASS again')\nraise ValueError('a\\nPASS b')\n")
    code, out = validate(name)
    assert code == 1
    assert out.splitlines()[-1] == "FAIL x�PASS y�PASS z (file): 1 error"
    assert "    PASS b" in out.splitlines()


def test_a_surrogate_file_name_survives_in_json(project: Path, validate: Validate) -> None:
    name = "caf\udce9.py"
    try:
        _plant(project / name, CHAIN)
    except OSError, UnicodeEncodeError:
        pytest.skip("the file system takes no undecodable file names")
    code, out = validate(name, "--json")
    assert code == 0 and out.isascii()
    assert json.loads(out)["target"]["entry"] == str(project / name)


def test_a_load_that_never_ends_times_out(project: Path, validate: Validate, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_client, "LIMITS", replace(LIMITS, load_timeout=0.5))
    monkeypatch.setattr(validation_module, "LIMITS", replace(LIMITS, load_timeout=0.5))
    _plant(project / "wf.py", "import threading\nthreading.Event().wait()\n")
    code, report = _json(validate, "wf.py")
    assert (code, _codes(report)) == (1, [("load_timeout", str(project / "wf.py"), None)])
    assert "0.5 s" in report["diagnostics"][0]["message"]


def test_exit_codes_for_usage_interrupts_and_internal_errors(
    project: Path, validate: Validate, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exited:
        workflow_cli.main(["validate"])
    assert exited.value.code == 2
    assert capsys.readouterr().out == ""

    _plant(project / "chain.py", CHAIN)

    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(validation_module, "parse_environment_request", broken)
    code, report = _json(validate, "chain.py")
    assert (code, _codes(report)) == (1, [("internal_error", None, None)])
    assert report["diagnostics"][0]["stage"] == "metadata"
    assert report["diagnostics"][0]["message"] == "RuntimeError: boom"

    async def interrupted(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(workflow_cli, "validate_workflow_path", interrupted)
    assert workflow_cli.main(["validate", "chain.py"]) == workflow_cli.EXIT_INTERRUPTED
    assert "Interrupted" in capsys.readouterr().err

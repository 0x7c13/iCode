# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the top-level Chrys CLI dispatcher."""

from __future__ import annotations

import re
import sys
import tomllib

import pytest

from chrys import __version__
from chrys.app.cli import app as cli_app
from chrys.foundation.branding import APP_COMMAND, APP_DISPLAY_NAME
from tests.support.paths import REPO_ROOT


def test_chrys_run_dispatches_to_run_subcommand(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run_main(argv: list[str]) -> int:
        calls.append(argv)
        return 7

    import chrys.app.cli.run as run_module

    monkeypatch.setattr(run_module, "main", fake_run_main)
    monkeypatch.setattr(sys, "argv", ["chrys", "run", "hello", "--agent", "Code"])

    assert cli_app.main() == 7
    assert calls == [["hello", "--agent", "Code"]]


def test_pyapp_main_exits_with_dispatcher_return_code(monkeypatch) -> None:
    monkeypatch.setattr(cli_app, "main", lambda: 7)

    with pytest.raises(SystemExit) as exc_info:
        cli_app.pyapp_main()

    assert exc_info.value.code == 7


def test_chrys_acp_dispatches_to_acp_subcommand(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_acp_main(argv: list[str]) -> int:
        calls.append(argv)
        return 8

    import chrys.app.cli.acp as acp_module

    monkeypatch.setattr(acp_module, "main", fake_acp_main)
    monkeypatch.setattr(sys, "argv", ["chrys", "acp", "--agent", "Code"])

    assert cli_app.main() == 8
    assert calls == [["--agent", "Code"]]


def test_chrys_serve_dispatches_to_serve_subcommand(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_serve_main(argv: list[str]) -> int:
        calls.append(argv)
        return 9

    import chrys.app.cli.serve as serve_module

    monkeypatch.setattr(serve_module, "main", fake_serve_main)
    monkeypatch.setattr(sys, "argv", ["chrys", "serve", "--host", "0.0.0.0", "--port", "9000"])

    assert cli_app.main() == 9
    assert calls == [["--host", "0.0.0.0", "--port", "9000"]]


def test_chrys_agents_dispatches_to_agents_subcommand(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_agents_main(argv: list[str]) -> int:
        calls.append(argv)
        return 10

    import chrys.app.cli.profiles as profiles_module

    monkeypatch.setattr(profiles_module, "agents_main", fake_agents_main)
    monkeypatch.setattr(sys, "argv", ["chrys", "agents", "--json"])

    assert cli_app.main() == 10
    assert calls == [["--json"]]


def test_chrys_models_dispatches_to_models_subcommand(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_models_main(argv: list[str]) -> int:
        calls.append(argv)
        return 11

    import chrys.app.cli.profiles as profiles_module

    monkeypatch.setattr(profiles_module, "models_main", fake_models_main)
    monkeypatch.setattr(sys, "argv", ["chrys", "models", "list"])

    assert cli_app.main() == 11
    assert calls == [["list"]]


def test_chrys_workflow_dispatches_to_workflow_subcommand(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_workflow_main(argv: list[str]) -> int:
        calls.append(argv)
        return 12

    import chrys.app.cli.workflow as workflow_module

    monkeypatch.setattr(workflow_module, "main", fake_workflow_main)
    monkeypatch.setattr(sys, "argv", ["chrys", "workflow", "run", "demo-workflow", "--trust"])

    assert cli_app.main() == 12
    assert calls == [["run", "demo-workflow", "--trust"]]


def test_chrys_hidden_tui_subprocess_dispatches_to_tui(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_tui_main() -> None:
        calls.append(sys.argv[:])

    import chrys.app.tui.app as tui_app

    monkeypatch.setattr(tui_app, "main", fake_tui_main)
    monkeypatch.setattr(sys, "argv", ["chrys", "__tui_subprocess__"])

    assert cli_app.main() == 0
    assert calls == [["chrys"]]


def test_chrys_hidden_tui_subprocess_forwards_session_arg(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_tui_main() -> None:
        calls.append(sys.argv[:])

    import chrys.app.tui.app as tui_app

    monkeypatch.setattr(tui_app, "main", fake_tui_main)
    monkeypatch.setattr(sys, "argv", ["chrys", "__tui_subprocess__", "--session", "session-1"])

    assert cli_app.main() == 0
    assert calls == [["chrys", "--session", "session-1"]]


def test_chrys_hidden_tui_subprocess_forwards_tui_startup_args(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_tui_main() -> None:
        calls.append(sys.argv[:])

    import chrys.app.tui.app as tui_app

    monkeypatch.setattr(tui_app, "main", fake_tui_main)
    monkeypatch.setattr(
        sys,
        "argv",
        ["chrys", "__tui_subprocess__", "-s", "session-1", "-a", "Code", "-C", "/repo"],
    )

    assert cli_app.main() == 0
    assert calls == [["chrys", "-s", "session-1", "-a", "Code", "-C", "/repo"]]


def test_chrys_session_arg_dispatches_to_tui(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_tui_main() -> None:
        calls.append(sys.argv[:])

    import chrys.app.tui.app as tui_app

    monkeypatch.setattr(tui_app, "main", fake_tui_main)
    monkeypatch.setattr(sys, "argv", ["chrys", "--session", "session-1"])

    assert cli_app.main() == 0
    assert calls == [["chrys", "--session", "session-1"]]


def test_chrys_without_run_routes_args_to_tui(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_tui_main() -> None:
        calls.append(sys.argv[:])

    import chrys.app.tui.app as tui_app

    monkeypatch.setattr(tui_app, "main", fake_tui_main)
    monkeypatch.setattr(sys, "argv", ["chrys", "status"])

    assert cli_app.main() == 0
    assert calls == [["chrys", "status"]]


def test_chrys_without_args_dispatches_to_tui(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_tui_main() -> None:
        calls.append(sys.argv[:])

    import chrys.app.tui.app as tui_app

    monkeypatch.setattr(tui_app, "main", fake_tui_main)
    monkeypatch.setattr(sys, "argv", ["chrys"])

    assert cli_app.main() == 0
    assert calls == [["chrys"]]


def test_chrys_help_shows_top_level_modes(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["chrys", "--help"])

    assert cli_app.main() == 0
    out = capsys.readouterr()
    assert f"usage: {APP_COMMAND} " in out.out
    assert "Show this help message and exit" in out.out
    assert "show this help message and exit" not in out.out
    assert "--version" in out.out
    assert "--profile PROFILE" not in out.out
    assert "-s SESSION" in out.out
    assert "--session SESSION" in out.out
    assert "-a AGENT" in out.out
    assert "--agent AGENT" in out.out
    assert "agent profile id, name," in out.out
    assert "or display name" in out.out
    assert "-m MODEL" in out.out
    assert "--model MODEL" in out.out
    assert "active model profile id" in out.out
    assert "-C DIR" in out.out
    assert "--workdir DIR" in out.out
    assert "install" in out.out
    assert "agents" in out.out
    assert "models" in out.out
    assert "acp" in out.out
    assert "serve" in out.out
    assert "List available agent profiles" in out.out
    assert "List available model profiles" in out.out
    assert f"Host the {APP_DISPLAY_NAME} TUI in a browser" in out.out
    assert "Run an Agent Client Protocol stdio server" in out.out
    assert "Run an agent headlessly until the final response" in out.out
    assert "Start the HTTP server" not in out.out
    assert f"{APP_COMMAND} <command> --help" in out.out


def test_chrys_short_help_shows_top_level_modes(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["chrys", "-h"])

    assert cli_app.main() == 0
    out = capsys.readouterr()
    assert f"usage: {APP_COMMAND} " in out.out
    assert "Show this help message and exit" in out.out
    assert "show this help message and exit" not in out.out
    assert "--version" in out.out
    assert "run" in out.out
    assert "install" in out.out
    assert "agents" in out.out
    assert "models" in out.out
    assert "acp" in out.out
    assert "serve" in out.out
    assert "Run an Agent Client Protocol stdio server" in out.out
    assert "Start the HTTP server" not in out.out


@pytest.mark.parametrize("flag", ["-v", "--version"])
def test_chrys_version_prints_package_version(monkeypatch, capsys, flag: str) -> None:
    monkeypatch.setattr(sys, "argv", ["chrys", flag])

    assert cli_app.main() == 0
    out = capsys.readouterr()
    assert out.out == f"{__version__}\n"
    assert out.err == ""


def test_chrys_non_run_args_route_to_tui_unchanged(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_tui_main() -> None:
        calls.append(sys.argv[:])

    import chrys.app.tui.app as tui_app

    monkeypatch.setattr(tui_app, "main", fake_tui_main)
    monkeypatch.setattr(sys, "argv", ["chrys", "tui"])

    assert cli_app.main() == 0
    assert calls == [["chrys", "tui"]]


def test_chrys_install_dispatches_to_installer(monkeypatch) -> None:
    calls: list[bool] = []

    def fake_install_to_path() -> None:
        calls.append(True)

    monkeypatch.setattr("chrys.app.installer.install_to_path", fake_install_to_path)
    monkeypatch.setattr(sys, "argv", ["chrys", "install"])

    assert cli_app.main() == 0
    assert calls == [True]


def test_chrys_install_help_uses_capitalized_help_text(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["chrys", "install", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        cli_app.main()

    out = capsys.readouterr()
    assert exc_info.value.code == 0
    assert f"usage: {APP_COMMAND} install" in out.out
    assert "Show this help message and exit" in out.out
    assert "show this help message and exit" not in out.out


def test_chrys_install_rejects_additional_args(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["chrys", "install", "extra"])

    with pytest.raises(SystemExit) as exc_info:
        cli_app.main()

    out = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "install does not accept arguments" in out.err


def test_pyapp_build_scripts_use_cli_dispatcher() -> None:
    root = REPO_ROOT

    assert "PYAPP_EXEC_SPEC=chrys.app.cli.app:pyapp_main" in (root / "scripts" / "build.sh").read_text(encoding="utf-8")
    assert '$env:PYAPP_EXEC_SPEC = "chrys.app.cli.app:pyapp_main"' in (root / "scripts" / "build.ps1").read_text(
        encoding="utf-8"
    )
    assert 'PYAPP_EXEC_SPEC: "chrys.app.cli.app:pyapp_main"' in (root / ".github" / "workflows" / "cd.yml").read_text(
        encoding="utf-8"
    )


def test_pyapp_build_renames_runtime_python_on_process_name_sensitive_platforms() -> None:
    root = REPO_ROOT
    build_sh = (root / "scripts" / "build.sh").read_text(encoding="utf-8")
    build_ps1 = (root / "scripts" / "build.ps1").read_text(encoding="utf-8")
    cd_workflow = (root / ".github" / "workflows" / "cd.yml").read_text(encoding="utf-8")

    assert 'WHEEL_SOURCE="dist/chrys-${VERSION}-py3-none-any.whl"' in build_sh
    assert 'WHEEL="$(basename "$WHEEL_SOURCE")"' in build_sh
    assert "BUILD_USES_RUNTIME_ALIAS=true" in build_sh
    assert "Linux|Darwin|MINGW*|MSYS*|CYGWIN*" in build_sh
    assert "chrys-runtime.exe" in build_sh
    assert 'pub const CHRYS_RUNTIME_EXE: \\&str = "chrys-runtime";' in build_sh
    assert "chrys-runtimew.exe" in build_sh
    assert 'target_os = "macos"' in build_sh
    assert 'target_os = "linux"' in build_sh
    assert "ensure_runtime_aliases" in build_sh
    assert "runtime_python_path" in build_sh
    assert "source_pth_path" in build_sh
    assert "fs::hard_link" in build_sh
    assert "app::CHRYS_RUNTIME_EXE" in build_sh
    assert 'PYAPP_DISTRIBUTION_PYTHON_PATH="python/python.exe"' in build_sh
    assert "install_project\\(\\)\\?;.*ensure_runtime_aliases" in build_sh

    assert "Patching PyApp to run Chrys through renamed Python" in build_ps1
    assert '$WheelSource = Join-Path "dist" "chrys-$Version-py3-none-any.whl"' in build_ps1
    assert "$Wheel = $WheelFile.Name" in build_ps1
    assert "chrys-runtime.exe" in build_ps1
    assert 'pub const CHRYS_RUNTIME_EXE: &str = "chrys-runtime";' in build_ps1
    assert "chrys-runtimew.exe" in build_ps1
    assert 'target_os = "macos"' in build_ps1
    assert 'target_os = "linux"' in build_ps1
    assert "ensure_runtime_aliases" in build_ps1
    assert "ensure_windows_runtime_aliases" not in build_ps1
    assert "runtime_python_path" in build_ps1
    assert "source_pth_path" in build_ps1
    assert "fs::hard_link" in build_ps1
    assert "app::CHRYS_RUNTIME_EXE" in build_ps1
    assert "AppMacRuntimeAliasPatch" in build_ps1
    assert "AppLinuxRuntimeAliasPatch" in build_ps1
    assert "RuntimeAliasAfterInstall" in build_ps1

    assert (
        'if [[ "${{ matrix.platform }}" == "windows" || "${{ matrix.platform }}" == "macos" || '
        '"${{ matrix.platform }}" == "linux" ]]; then'
    ) in cd_workflow
    assert "chrys-runtime.exe" in cd_workflow
    assert 'pub const CHRYS_RUNTIME_EXE: \\&str = "chrys-runtime";' in cd_workflow
    assert "chrys-runtimew.exe" in cd_workflow
    assert 'target_os = "macos"' in cd_workflow
    assert 'target_os = "linux"' in cd_workflow
    assert "ensure_runtime_aliases" in cd_workflow
    assert "runtime_python_path" in cd_workflow
    assert "source_pth_path" in cd_workflow
    assert "fs::hard_link" in cd_workflow
    assert "app::CHRYS_RUNTIME_EXE" in cd_workflow
    assert "install_project\\(\\)\\?;.*ensure_runtime_aliases" in cd_workflow


def test_ci_and_build_paths_include_serve_dependencies() -> None:
    root = REPO_ROOT
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]

    assert "setproctitle==1.3.7" in pyproject["project"]["dependencies"]
    assert "textual-serve==1.1.3" in extras["tui"]
    assert "server" not in extras
    assert "chrys[tui,dev,doc_converter,observability]" in extras["all"]

    ci_workflow = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "uv sync --extra all" in ci_workflow

    build_sh = (root / "scripts" / "build.sh").read_text(encoding="utf-8")
    build_ps1 = (root / "scripts" / "build.ps1").read_text(encoding="utf-8")
    cd_workflow = (root / ".github" / "workflows" / "cd.yml").read_text(encoding="utf-8")

    assert "PYAPP_PROJECT_FEATURES=tui,observability,doc_converter" in build_sh
    assert '$env:PYAPP_PROJECT_FEATURES = "tui,observability,doc_converter"' in build_ps1
    # In CD the features are gated per matrix flavor: the offline flavor bundles
    # the extras into its distribution instead of pip-installing them.
    assert "'tui,observability,doc_converter'" in cd_workflow

    # The offline distribution must bundle the same extras the pip flavor
    # would have installed, or `chrys serve` breaks only in offline builds.
    offline_sh = (root / "scripts" / "build_offline_dist.sh").read_text(encoding="utf-8")
    offline_ps1 = (root / "scripts" / "build_offline_dist.ps1").read_text(encoding="utf-8")
    assert 'EXTRAS="tui,doc_converter,observability"' in offline_sh
    assert '[string]$Extras = "tui,doc_converter,observability"' in offline_ps1


def test_offline_wheel_overrides_match_the_lock() -> None:
    # build_offline_dist.sh refuses an override whose version differs from
    # uv.lock, but only once a Linux offline distribution is built; a pin bump
    # that forgets the rebuilt wheel should fail here instead.
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked: dict[str, set[str]] = {}
    for package in lock["package"]:
        locked.setdefault(package["name"], set()).add(package["version"])
    # The offline binaries ship the CPython version .python-version pins.
    python_version = re.fullmatch(r"3\.(\d+)\.\d+", (REPO_ROOT / ".python-version").read_text(encoding="utf-8").strip())
    assert python_version is not None
    abi = f"cp3{python_version.group(1)}"
    manifest = (REPO_ROOT / "scripts" / "offline_wheel_overrides.txt").read_text(encoding="utf-8")

    machines: dict[str, set[str]] = {}
    for raw in manifest.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        project, version, machine, sha256, url = line.split()
        assert locked.get(project) == {version}, line
        assert re.fullmatch(r"[0-9a-f]{64}", sha256), line
        # The main repository's CD downloads these, so each wheel must be
        # published there, not on a fork, in a release tagged for its version.
        assert url.startswith(f"https://github.com/openJiuwen-ai/iCode/releases/download/{project}-{version}-"), line
        filename = url.rsplit("/", 1)[1]
        assert filename.startswith(f"{project}-{version}-{abi}-{abi}-"), line
        assert filename.endswith(f"_{machine}.whl"), line
        machines.setdefault(project, set()).add(machine)

    # PyPI has no glibc 2.17 Pillow wheel for either machine CD builds the
    # Linux offline binaries on.
    assert machines.get("pillow") == {"x86_64", "aarch64"}

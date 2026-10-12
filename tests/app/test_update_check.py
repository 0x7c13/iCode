# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The once-a-day lookup of the newest release behind the welcome screen's update hint.

Every request is answered by an ``httpx.MockTransport``, so no test reaches a release host.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx
import pytest

from chrys import __version__
from chrys.app import update_check
from chrys.app.install_flavor import InstallFlavor
from chrys.app.update_check import CHECK_INTERVAL_SECONDS, UpdateCheck, UpdateNotice
from tests.support.waiting import wait_for

if TYPE_CHECKING:
    from pathlib import Path

_PYPI = "https://pypi.org/pypi/icode-tui/json"
_GITHUB = "https://github.com/openJiuwen-ai/iCode/releases/latest"
_NOW = 1_800_000_000.0


_SILENT = None
"""A host that takes the request and gives no answer until the test releases it."""


class _Hosts:
    """Release hosts answering from a table; a URL missing from it fails to connect."""

    def __init__(self, answers: dict[str, httpx.Response | None]) -> None:
        self.answers = answers
        self.asked: list[str] = []
        self.user_agents: list[str | None] = []
        self.threads: list[threading.Thread] = []
        self.release = threading.Event()
        self.transport = httpx.MockTransport(self._answer)

    def _answer(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.asked.append(url)
        self.user_agents.append(request.headers.get("user-agent"))
        self.threads.append(threading.current_thread())
        answer = self.answers.get(url)
        if url in self.answers and answer is None:
            self.release.wait(timeout=30)
        if answer is None:
            raise httpx.ConnectError("unreachable", request=request)
        return answer


def _pypi(version: str) -> httpx.Response:
    return httpx.Response(200, json={"info": {"version": version}})


def _github(tag: str) -> httpx.Response:
    return httpx.Response(302, headers={"Location": f"https://github.com/openJiuwen-ai/iCode/releases/tag/{tag}"})


def _check(tmp_path: Path, hosts: _Hosts, flavor: InstallFlavor = InstallFlavor.UV, **kwargs: object) -> UpdateCheck:
    return UpdateCheck(
        flavor=flavor,
        cache_path=tmp_path / "update-check.json",
        current_version="0.29.1",
        now=lambda: _NOW,
        transport=hosts.transport,
        **kwargs,  # type: ignore[arg-type]
    )


async def _run(check: UpdateCheck) -> list[UpdateNotice | None]:
    shown: list[UpdateNotice | None] = []
    await check.run(shown.append)
    return shown


def _write_cache(tmp_path: Path, *, checked_at: float, latest: str, source: str = "pypi") -> None:
    payload = {"checked_at": checked_at, "latest": latest, "source": source}
    (tmp_path / "update-check.json").write_text(json.dumps(payload), encoding="utf-8")


def _read_cache(tmp_path: Path) -> dict[str, object]:
    return json.loads((tmp_path / "update-check.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("text", "parsed"),
    [
        ("0.29.1", (0, 29, 1)),
        ("v1.2.3", (1, 2, 3)),
        ("0.30.0rc1", None),
        ("v0.30.0-beta", None),
        ("0.30", None),
        ("0.30.0.1", None),
        ("", None),
        (None, None),
        (30, None),
    ],
)
def test_only_plain_three_number_versions_count(text: object, parsed: tuple[int, int, int] | None) -> None:
    assert update_check.parse_version(text) == parsed


@dataclass(frozen=True)
class _Platform:
    is_windows: bool


@pytest.mark.parametrize(
    ("flavor", "windows", "command"),
    [
        (InstallFlavor.UV, False, "uv tool upgrade iCode-TUI"),
        (InstallFlavor.PIPX, False, "pipx upgrade iCode-TUI"),
        (InstallFlavor.PIP, True, "pip install -U iCode-TUI"),
        (
            InstallFlavor.OFFLINE,
            False,
            "curl -fsSL https://raw.githubusercontent.com/openJiuwen-ai/iCode/main/scripts/install.sh | sh",
        ),
        (
            InstallFlavor.OFFLINE,
            True,
            "irm https://raw.githubusercontent.com/openJiuwen-ai/iCode/main/scripts/install.ps1 | iex",
        ),
        (InstallFlavor.SOURCE, False, None),
    ],
)
def test_the_upgrade_command_matches_how_the_copy_was_installed(
    monkeypatch: pytest.MonkeyPatch, flavor: InstallFlavor, windows: bool, command: str | None
) -> None:
    monkeypatch.setattr(update_check, "get_platform", lambda: _Platform(is_windows=windows))

    assert update_check.upgrade_command(flavor) == command


async def test_a_uv_install_asks_pypi_first(tmp_path: Path) -> None:
    hosts = _Hosts({_PYPI: _pypi("0.30.0"), _GITHUB: _github("v0.31.0")})

    assert await _run(_check(tmp_path, hosts)) == [UpdateNotice("v0.30.0", "uv tool upgrade iCode-TUI")]
    assert hosts.asked == [_PYPI]
    assert _read_cache(tmp_path) == {"checked_at": _NOW, "latest": "0.30.0", "source": "pypi"}


async def test_github_answers_when_pypi_does_not(tmp_path: Path) -> None:
    hosts = _Hosts({_PYPI: httpx.Response(503), _GITHUB: _github("v0.30.0")})

    assert await _run(_check(tmp_path, hosts, InstallFlavor.PIPX)) == [
        UpdateNotice("v0.30.0", "pipx upgrade iCode-TUI")
    ]
    assert hosts.asked == [_PYPI, _GITHUB]
    assert _read_cache(tmp_path) == {"checked_at": _NOW, "latest": "0.30.0", "source": "github"}


async def test_every_request_names_chrys_and_its_project_page(tmp_path: Path) -> None:
    hosts = _Hosts({_PYPI: httpx.Response(503), _GITHUB: _github("v0.30.0")})

    await _run(_check(tmp_path, hosts))

    expected = f"Chrys/{__version__} (+https://github.com/openJiuwen-ai/iCode)"
    assert hosts.user_agents == [expected, expected]


@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(200, headers={"Location": "https://github.com/openJiuwen-ai/iCode/releases/tag/v0.30.0"}),
        httpx.Response(302, headers={"Location": "https://github.com/openJiuwen-ai/iCode/releases/download/v0.30.0"}),
    ],
    ids=["no-redirect", "not-a-release-page"],
)
async def test_github_tells_nothing_without_a_redirect_to_a_release(tmp_path: Path, reply: httpx.Response) -> None:
    hosts = _Hosts({_PYPI: httpx.Response(503), _GITHUB: reply})

    assert await _run(_check(tmp_path, hosts)) == []
    # No other host is asked after these two.
    assert hosts.asked == [_PYPI, _GITHUB]
    assert _read_cache(tmp_path) == {"checked_at": _NOW, "latest": "", "source": ""}


async def test_an_offline_package_never_asks_pypi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(update_check, "get_platform", lambda: _Platform(is_windows=False))
    hosts = _Hosts({_PYPI: _pypi("0.31.0"), _GITHUB: _github("v0.30.0")})

    shown = await _run(_check(tmp_path, hosts, InstallFlavor.OFFLINE))

    assert hosts.asked == [_GITHUB]
    assert shown == [UpdateNotice("v0.30.0", update_check._INSTALL_SH)]


async def test_a_pre_release_is_passed_over(tmp_path: Path) -> None:
    hosts = _Hosts({_PYPI: _pypi("0.30.0rc1"), _GITHUB: _github("v0.30.0")})

    assert await _run(_check(tmp_path, hosts)) == [UpdateNotice("v0.30.0", "uv tool upgrade iCode-TUI")]


@pytest.mark.parametrize("latest", ["0.29.1", "0.29.0"])
async def test_no_hint_unless_the_release_is_newer(tmp_path: Path, latest: str) -> None:
    hosts = _Hosts({_PYPI: _pypi(latest)})

    assert await _run(_check(tmp_path, hosts)) == [None]
    assert _read_cache(tmp_path)["latest"] == latest


async def test_a_new_answer_takes_back_a_hint_it_no_longer_backs(tmp_path: Path) -> None:
    _write_cache(tmp_path, checked_at=_NOW - CHECK_INTERVAL_SECONDS, latest="0.30.0")
    hosts = _Hosts({_PYPI: _pypi("0.29.1")})

    assert await _run(_check(tmp_path, hosts)) == [UpdateNotice("v0.30.0", "uv tool upgrade iCode-TUI"), None]


async def test_a_check_within_a_day_shows_the_saved_answer_without_asking(tmp_path: Path) -> None:
    _write_cache(tmp_path, checked_at=_NOW - CHECK_INTERVAL_SECONDS + 60, latest="0.30.0")
    hosts = _Hosts({})

    assert await _run(_check(tmp_path, hosts)) == [UpdateNotice("v0.30.0", "uv tool upgrade iCode-TUI")]
    assert hosts.asked == []


@pytest.mark.parametrize("age", [CHECK_INTERVAL_SECONDS, -60], ids=["a-day-old", "clock-set-back"])
async def test_an_old_answer_shows_at_once_and_is_then_replaced(tmp_path: Path, age: float) -> None:
    _write_cache(tmp_path, checked_at=_NOW - age, latest="0.30.0")
    hosts = _Hosts({_PYPI: _pypi("0.30.1")})

    assert [notice and notice.version for notice in await _run(_check(tmp_path, hosts))] == ["v0.30.0", "v0.30.1"]
    assert _read_cache(tmp_path) == {"checked_at": _NOW, "latest": "0.30.1", "source": "pypi"}


async def test_a_failed_check_keeps_the_last_answer_and_waits_a_day(tmp_path: Path) -> None:
    _write_cache(tmp_path, checked_at=_NOW - 2 * CHECK_INTERVAL_SECONDS, latest="0.30.0", source="github")

    shown = await _run(_check(tmp_path, _Hosts({})))

    assert shown == [UpdateNotice("v0.30.0", "uv tool upgrade iCode-TUI")]
    assert _read_cache(tmp_path) == {"checked_at": _NOW, "latest": "0.30.0", "source": "github"}


async def test_no_check_when_the_attempt_cannot_be_saved(tmp_path: Path) -> None:
    (tmp_path / "blocked").write_text("a file where the folder should be", encoding="utf-8")
    hosts = _Hosts({_PYPI: _pypi("0.30.0")})
    check = UpdateCheck(
        flavor=InstallFlavor.UV,
        cache_path=tmp_path / "blocked" / "update-check.json",
        current_version="0.29.1",
        now=lambda: _NOW,
        transport=hosts.transport,
    )

    assert await _run(check) == []
    # Otherwise every start would ask again.
    assert hosts.asked == []


async def test_the_first_check_makes_the_folder_it_saves_into(tmp_path: Path) -> None:
    hosts = _Hosts({_PYPI: _pypi("0.30.0")})
    cache_path = tmp_path / "new" / "update-check.json"
    check = UpdateCheck(
        flavor=InstallFlavor.UV,
        cache_path=cache_path,
        current_version="0.29.1",
        now=lambda: _NOW,
        transport=hosts.transport,
    )

    assert await _run(check) == [UpdateNotice("v0.30.0", "uv tool upgrade iCode-TUI")]
    assert json.loads(cache_path.read_text(encoding="utf-8"))["latest"] == "0.30.0"


async def test_a_check_cut_short_by_quitting_waits_a_day_and_holds_nothing_up(tmp_path: Path) -> None:
    _write_cache(tmp_path, checked_at=_NOW - 2 * CHECK_INTERVAL_SECONDS, latest="0.30.0")
    hosts = _Hosts({_PYPI: _SILENT})
    task = asyncio.create_task(_run(_check(tmp_path, hosts)))
    try:
        await wait_for(lambda: hosts.asked or task.done(), description="PyPI asked")
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert _read_cache(tmp_path) == {"checked_at": _NOW, "latest": "0.30.0", "source": "pypi"}
        # A lookup that never returns is left on a thread the app's exit does not wait for.
        assert [thread.daemon for thread in hosts.threads] == [True]
    finally:
        task.cancel()
        hosts.release.set()
        for thread in hosts.threads:
            await asyncio.to_thread(thread.join, 30)


async def test_an_offline_package_passes_over_what_another_install_saved_from_pypi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(update_check, "get_platform", lambda: _Platform(is_windows=False))
    _write_cache(tmp_path, checked_at=_NOW - 60, latest="0.31.0", source="pypi")
    hosts = _Hosts({_GITHUB: _github("v0.30.0")})

    shown = await _run(_check(tmp_path, hosts, InstallFlavor.OFFLINE))

    assert [notice and notice.version for notice in shown] == ["v0.30.0"]
    assert hosts.asked == [_GITHUB]


async def test_an_offline_package_keeps_to_a_failed_check_for_the_day(tmp_path: Path) -> None:
    _write_cache(tmp_path, checked_at=_NOW - 60, latest="", source="")
    hosts = _Hosts({})

    assert await _run(_check(tmp_path, hosts, InstallFlavor.OFFLINE)) == []
    assert hosts.asked == []


async def test_a_first_check_that_fails_is_remembered_too(tmp_path: Path) -> None:
    assert await _run(_check(tmp_path, _Hosts({}))) == []
    assert _read_cache(tmp_path) == {"checked_at": _NOW, "latest": "", "source": ""}


@pytest.mark.parametrize("content", ["not json", "[]", '{"checked_at": "soon", "latest": "0.30.0", "source": "pypi"}'])
async def test_a_damaged_answer_file_is_ignored(tmp_path: Path, content: str) -> None:
    (tmp_path / "update-check.json").write_text(content, encoding="utf-8")
    hosts = _Hosts({_PYPI: _pypi("0.30.0")})

    assert await _run(_check(tmp_path, hosts)) == [UpdateNotice("v0.30.0", "uv tool upgrade iCode-TUI")]


async def test_a_reply_of_the_wrong_shape_counts_as_no_answer(tmp_path: Path) -> None:
    hosts = _Hosts({_PYPI: httpx.Response(200, json={"info": None}), _GITHUB: _github("v0.30.0")})

    assert await _run(_check(tmp_path, hosts)) == [UpdateNotice("v0.30.0", "uv tool upgrade iCode-TUI")]


@pytest.mark.parametrize(
    ("enabled", "served", "ci", "flavor", "checks"),
    [
        (True, False, "", InstallFlavor.UV, True),
        (True, False, "false", InstallFlavor.OFFLINE, True),
        (False, False, "", InstallFlavor.UV, False),
        (True, True, "", InstallFlavor.UV, False),
        (True, False, "true", InstallFlavor.UV, False),
        (True, False, "1", InstallFlavor.PIP, False),
        (True, False, "", InstallFlavor.SOURCE, False),
    ],
    ids=["uv", "offline-ci-false", "setting-off", "icode-serve", "ci", "ci-1", "source-checkout"],
)
def test_only_an_installed_copy_in_a_terminal_checks(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, served: bool, ci: str, flavor: InstallFlavor, checks: bool
) -> None:
    monkeypatch.setenv("CI", ci)
    monkeypatch.setattr(update_check, "detect_install_flavor", lambda: flavor)

    check = update_check.for_this_copy(enabled=enabled, served=served)

    assert (check is not None) is checks
    if check is not None:
        assert check.flavor is flavor
        assert check.cache_path == update_check.get_platform().config_dir / "update-check.json"

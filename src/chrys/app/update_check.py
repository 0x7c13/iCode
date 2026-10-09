# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Find out whether a newer release is out, for the hint on the TUI's welcome screen.

At most once a day, when the TUI starts, this asks for the newest version number: PyPI, then
GitHub, for a copy installed with uv, pipx or pip, and GitHub alone for an offline package. The
requests carry nothing about the user or their work; their User-Agent names only Chrys, its version
and the project page. The answer is kept in ``<config_dir>/update-check.json``, so every start can
show the hint at once, and any failure leaves the hint as it was without telling anyone.

Headless runs, the ACP server, ``icode serve``, CI and source checkouts never check.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from chrys import DISTRIBUTION_NAME, __version__
from chrys.app.install_flavor import InstallFlavor, detect_install_flavor
from chrys.foundation.platform import get_platform
from chrys.foundation.platform.files import atomic_write_text

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import httpx

logger = logging.getLogger(__name__)

CHECK_INTERVAL_SECONDS = 24 * 60 * 60
CACHE_FILE_NAME = "update-check.json"
_TIMEOUT_SECONDS = 5.0

_PROJECT_URL = "https://github.com/openJiuwen-ai/iCode"
_PYPI_URL = f"https://pypi.org/pypi/{DISTRIBUTION_NAME.lower()}/json"
# Redirects to the newest release's page without the API and its hourly quota.
_GITHUB_LATEST_URL = f"{_PROJECT_URL}/releases/latest"
# PyPI and GitHub ask a client to name itself in the User-Agent, with a way to reach its makers;
# the name is the one LLM requests carry.
_USER_AGENT = f"Chrys/{__version__} (+{_PROJECT_URL})"

_INSTALL_SH = "curl -fsSL https://raw.githubusercontent.com/openJiuwen-ai/iCode/main/scripts/install.sh | sh"
_INSTALL_PS1 = "irm https://raw.githubusercontent.com/openJiuwen-ai/iCode/main/scripts/install.ps1 | iex"


@dataclass(frozen=True, slots=True)
class UpdateNotice:
    """A newer release, and the command that installs it the way this copy was installed."""

    version: str
    command: str


def parse_version(text: object) -> tuple[int, int, int] | None:
    """``X.Y.Z``, with or without a leading ``v``; anything else, such as a pre-release, is None."""
    if not isinstance(text, str):
        return None
    parts = text.strip().removeprefix("v").split(".")
    if len(parts) != 3 or not all(part.isascii() and part.isdigit() for part in parts):
        return None
    return int(parts[0]), int(parts[1]), int(parts[2])


def upgrade_command(flavor: InstallFlavor) -> str | None:
    """The command that upgrades a copy installed this way."""
    match flavor:
        case InstallFlavor.UV:
            return f"uv tool upgrade {DISTRIBUTION_NAME}"
        case InstallFlavor.PIPX:
            return f"pipx upgrade {DISTRIBUTION_NAME}"
        case InstallFlavor.PIP:
            return f"pip install -U {DISTRIBUTION_NAME}"
        case InstallFlavor.OFFLINE:
            return _INSTALL_PS1 if get_platform().is_windows else _INSTALL_SH
        case InstallFlavor.SOURCE:
            return None


def _from_pypi(client: httpx.Client) -> object:
    response = client.get(_PYPI_URL, follow_redirects=True)
    response.raise_for_status()
    return response.json()["info"]["version"]


def _from_github(client: httpx.Client) -> object:
    response = client.get(_GITHUB_LATEST_URL)
    # .../releases/tag/v0.29.1
    location = response.headers.get("location", "") if response.is_redirect else ""
    _, found, tag = location.rpartition("/releases/tag/")
    return tag if found else None


_SOURCES = {
    "pypi": _from_pypi,
    "github": _from_github,
}


def _sources_for(flavor: InstallFlavor) -> tuple[str, ...]:
    # PyPI gets a release before its offline packages are built, and keeps it if their build fails.
    return ("github",) if flavor is InstallFlavor.OFFLINE else ("pypi", "github")


def fetch_latest(flavor: InstallFlavor, *, transport: httpx.BaseTransport | None = None) -> tuple[str, str] | None:
    """The newest version and the host that told, or None when every host failed.

    Blocks; :func:`_fetch_in_background` runs it off the event loop.
    """
    import httpx

    with httpx.Client(
        timeout=httpx.Timeout(_TIMEOUT_SECONDS), transport=transport, headers={"User-Agent": _USER_AGENT}
    ) as client:
        for source in _sources_for(flavor):
            try:
                found = _SOURCES[source](client)
            except httpx.HTTPError, ValueError, KeyError, TypeError:
                logger.debug("Update check: no answer from %s", source, exc_info=True)
                continue
            version = parse_version(found)
            if version is not None:
                return ".".join(map(str, version)), source
    return None


async def _fetch_in_background(flavor: InstallFlavor, transport: httpx.BaseTransport | None) -> tuple[str, str] | None:
    """:func:`fetch_latest` on a daemon thread of its own.

    A host name lookup can hang where the network drops queries, and nothing can cancel it. On
    the event loop's worker threads it would hold up the app's exit until it gave up; a daemon
    thread is simply left behind.
    """
    loop = asyncio.get_running_loop()
    answer: asyncio.Future[tuple[str, str] | None] = loop.create_future()

    def deliver(found: tuple[str, str] | None) -> None:
        if not answer.done():  # Not given up on when the app quit.
            answer.set_result(found)

    def work() -> None:
        try:
            found = fetch_latest(flavor, transport=transport)
        except Exception:
            logger.debug("Update check failed", exc_info=True)
            found = None
        with contextlib.suppress(RuntimeError):  # The loop closed while the check ran.
            loop.call_soon_threadsafe(deliver, found)

    threading.Thread(target=work, name="icode-update-check", daemon=True).start()
    return await answer


@dataclass(frozen=True, slots=True)
class _Cache:
    checked_at: float
    latest: str
    source: str


def _read_cache(path: Path) -> _Cache | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        checked_at, latest, source = data["checked_at"], data["latest"], data["source"]
    except OSError, ValueError, KeyError, TypeError:
        return None
    if not isinstance(checked_at, int | float) or not isinstance(latest, str) or not isinstance(source, str):
        return None
    return _Cache(float(checked_at), latest, source)


def _write_cache(path: Path, cache: _Cache) -> bool:
    """Whether ``cache`` was saved."""
    payload = json.dumps({"checked_at": cache.checked_at, "latest": cache.latest, "source": cache.source})
    try:
        atomic_write_text(path, payload + "\n")
    except OSError:
        logger.debug("Update check: could not save %s", path, exc_info=True)
        return False
    return True


@dataclass(frozen=True, slots=True)
class UpdateCheck:
    """The check for one TUI run; :func:`for_this_copy` decides whether there is one."""

    flavor: InstallFlavor
    cache_path: Path
    current_version: str = __version__
    now: Callable[[], float] = time.time
    transport: httpx.BaseTransport | None = None

    def notice(self, latest: str) -> UpdateNotice | None:
        """The hint for ``latest``, if it is newer than the running version."""
        newest, running = parse_version(latest), parse_version(self.current_version)
        if newest is None or running is None or newest <= running:
            return None
        command = upgrade_command(self.flavor)
        return None if command is None else UpdateNotice(f"v{latest}", command)

    async def run(self, show: Callable[[UpdateNotice | None], None]) -> None:
        """Show what the last check found, then check again if that was a day or more ago.

        ``show`` gets None when a new answer takes back the notice the last one gave.
        """
        cache = await asyncio.to_thread(_read_cache, self.cache_path)
        if cache is not None and cache.source not in ("", *_sources_for(self.flavor)):
            # Saved by another install sharing the folder, from a host this one does not ask.
            cache = None
        if cache is not None and (notice := self.notice(cache.latest)) is not None:
            show(notice)
        now = self.now()
        # A clock set back past the last check counts as a day gone by.
        if cache is not None and 0 <= now - cache.checked_at < CHECK_INTERVAL_SECONDS:
            return
        # Saved before asking, so a check that fails or is cut short by quitting waits a day too,
        # keeping what the last one found.
        kept = _Cache(now, cache.latest, cache.source) if cache is not None else _Cache(now, "", "")
        if not await asyncio.to_thread(_write_cache, self.cache_path, kept):
            # Without a record of this attempt, every start would ask again.
            return
        found = await _fetch_in_background(self.flavor, self.transport)
        if found is not None:
            latest, source = found
            show(self.notice(latest))
            await asyncio.to_thread(_write_cache, self.cache_path, _Cache(now, latest, source))


def _running_in_ci() -> bool:
    return os.environ.get("CI", "").strip().lower() not in {"", "0", "false", "no"}


def for_this_copy(*, enabled: bool, served: bool) -> UpdateCheck | None:
    """The check for this TUI run, or None when the setting, CI or the install rules it out."""
    if not enabled or served or _running_in_ci():
        return None
    flavor = detect_install_flavor()
    if flavor is InstallFlavor.SOURCE:
        return None
    return UpdateCheck(flavor=flavor, cache_path=get_platform().config_dir / CACHE_FILE_NAME)

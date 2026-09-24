# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Native desktop notification drivers."""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from chrys.foundation.platform import get_platform
from chrys.foundation.platform.process import managed_subprocess

_COMMAND_TIMEOUT_SECONDS = 3.0
_MACOS_SOUND_NAME = "Glass"
_MACOS_SOUND_PATH = Path("/System/Library/Sounds/Glass.aiff")


@dataclass(frozen=True, slots=True)
class NotificationPayload:
    """A native notification request."""

    title: str
    body: str
    sound: bool = True
    desktop: bool = True


@dataclass(frozen=True, slots=True)
class NotificationDeliveryResult:
    """Best-effort delivery result."""

    desktop_sent: bool = False
    sound_sent: bool = False


class NotificationDriver(Protocol):
    """Async desktop notification driver."""

    async def send(self, payload: NotificationPayload) -> NotificationDeliveryResult:
        """Send a notification."""


async def _run_command(argv: list[str], *, timeout: float = _COMMAND_TIMEOUT_SECONDS) -> bool:
    """Run a fixed argv command, suppressing output and returning success.

    Spawns via ``managed_subprocess`` — a bare ``create_subprocess_exec``
    would flash a visible console window on Windows for every PowerShell
    notification command.
    """
    if not argv or not argv[0]:
        return False
    try:
        async with managed_subprocess(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        ) as proc:
            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout)
            except TimeoutError:
                return False
    except OSError:
        return False
    return proc.returncode == 0


def _applescript_string(value: str) -> str:
    """Return a quoted AppleScript string literal."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


class MacOSNotificationDriver:
    """macOS native command driver."""

    async def send(self, payload: NotificationPayload) -> NotificationDeliveryResult:
        desktop_sent = False
        sound_sent = False
        if payload.desktop:
            notifier = shutil.which("terminal-notifier")
            if notifier:
                argv = [notifier, "-title", payload.title, "-message", payload.body]
                if payload.sound:
                    argv.extend(["-sound", "default"])
                desktop_sent = await _run_command(argv)
                sound_sent = desktop_sent and payload.sound
            else:
                script = (
                    f"display notification {_applescript_string(payload.body)} "
                    f"with title {_applescript_string(payload.title)}"
                )
                if payload.sound:
                    script += f" sound name {_applescript_string(_MACOS_SOUND_NAME)}"
                desktop_sent = await _run_command(["/usr/bin/osascript", "-e", script])
                sound_sent = desktop_sent and payload.sound
        elif payload.sound:
            sound_sent = await self._play_sound()
        return NotificationDeliveryResult(desktop_sent=desktop_sent, sound_sent=sound_sent)

    async def _play_sound(self) -> bool:
        if _MACOS_SOUND_PATH.is_file() and await _run_command(["/usr/bin/afplay", str(_MACOS_SOUND_PATH)]):
            return True
        return await _run_command(["/usr/bin/osascript", "-e", "beep 1"])


class LinuxNotificationDriver:
    """Linux native command driver."""

    async def send(self, payload: NotificationPayload) -> NotificationDeliveryResult:
        desktop_sent = False
        sound_sent = False
        notify_send = shutil.which("notify-send")
        if payload.desktop and notify_send:
            desktop_sent = await _run_command([notify_send, payload.title, payload.body])
        if payload.sound:
            sound_sent = await self._play_sound()
        return NotificationDeliveryResult(desktop_sent=desktop_sent, sound_sent=sound_sent)

    async def _play_sound(self) -> bool:
        canberra = shutil.which("canberra-gtk-play")
        if canberra and await _run_command([canberra, "-i", "message-new-instant"]):
            return True
        paplay = shutil.which("paplay")
        sound_path = Path("/usr/share/sounds/freedesktop/stereo/message.oga")
        if paplay and sound_path.is_file():
            return await _run_command([paplay, str(sound_path)])
        return False


def _powershell_string(value: str) -> str:
    """Return a single-quoted PowerShell string literal."""
    return "'" + value.replace("'", "''") + "'"


class WindowsNotificationDriver:
    """Windows native command driver."""

    async def send(self, payload: NotificationPayload) -> NotificationDeliveryResult:
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if not powershell:
            return NotificationDeliveryResult()
        sound_line = "[System.Media.SystemSounds]::Asterisk.Play();" if payload.sound else ""
        script = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "Add-Type -AssemblyName System.Drawing;"
            f"{sound_line}"
            "$n=New-Object System.Windows.Forms.NotifyIcon;"
            "$n.Icon=[System.Drawing.SystemIcons]::Information;"
            f"$n.BalloonTipTitle={_powershell_string(payload.title)};"
            f"$n.BalloonTipText={_powershell_string(payload.body)};"
            f"$n.Visible={'$true' if payload.desktop else '$false'};"
            "if($n.Visible){$n.ShowBalloonTip(5000);Start-Sleep -Milliseconds 1000};"
            "$n.Dispose();"
        )
        argv = [powershell, "-NoProfile", "-NonInteractive", "-Command", script]
        ok = await _run_command(argv, timeout=5.0)
        return NotificationDeliveryResult(desktop_sent=ok and payload.desktop, sound_sent=ok and payload.sound)


class NullNotificationDriver:
    """No-op notification driver."""

    async def send(self, payload: NotificationPayload) -> NotificationDeliveryResult:
        return NotificationDeliveryResult()


def default_notification_driver() -> NotificationDriver:
    """Return the native driver for the current platform."""
    platform = get_platform()
    if platform.is_macos:
        return MacOSNotificationDriver()
    if platform.is_linux:
        return LinuxNotificationDriver()
    if platform.is_windows:
        return WindowsNotificationDriver()
    return NullNotificationDriver()

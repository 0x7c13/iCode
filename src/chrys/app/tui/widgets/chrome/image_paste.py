# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Convert pasted or dropped image paths into chat ``@file`` mentions."""

from __future__ import annotations

import os
import shlex
import subprocess
import time
import uuid
from binascii import Error as BinasciiError
from binascii import unhexlify
from collections.abc import Sequence
from contextlib import suppress
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import url2pathname

from chrys.app.tui.terminal.title import running_under_textual_web_driver
from chrys.foundation.platform import get_platform, safe_getcwd
from chrys.foundation.text.images import IMAGE_MIME_BY_EXTENSION
from chrys.foundation.text.mentions import format_file_mention

if TYPE_CHECKING:
    from PIL import Image

    from chrys.service.state.store import StateStore

CLIPBOARD_IMAGE_SESSION_SUBDIR = Path("attachments") / "clipboard"
_CLIPBOARD_IMAGE_KEEP_COUNT = 100
_CLIPBOARD_IMAGE_KEEP_BYTES = 512 * 1024 * 1024
_WINDOWS_CLIPBOARD_GRAB_ATTEMPTS = 5
_WINDOWS_CLIPBOARD_GRAB_DELAY_SECONDS = 0.02
_MACOS_CLIPBOARD_IMAGE_CLASSES = ("PNGf", "TIFF", "JPEG", "GIFf")
# The formats those classes carry; Pillow decodes no other from the clipboard.
_MACOS_CLIPBOARD_IMAGE_FORMATS = ("PNG", "TIFF", "JPEG", "GIF")
_MACOS_CLASS_OPEN = chr(0x00AB)
_MACOS_CLASS_CLOSE = chr(0x00BB)


def convert_pasted_image_paths_to_mentions(text: str, cwd: Path) -> str:
    """Return ``@`` mentions when *text* is a paste/drop payload of image paths.

    Terminal drag-and-drop generally arrives as bracketed paste text. File
    copy/paste can arrive the same way depending on the terminal and platform.
    Only path-only payloads are rewritten; normal prose is left untouched.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    paths = _parse_image_path_payload(normalized, cwd=cwd)
    if not paths:
        return text
    return " ".join(format_file_mention(path) for path in paths) + " "


def clipboard_image_dir_for_session(state_store: StateStore | None, session_id: str | None) -> Path | None:
    """Return the session attachment directory for saved clipboard images."""
    if state_store is None or not session_id:
        return None
    return state_store.session_dir(session_id) / CLIPBOARD_IMAGE_SESSION_SUBDIR


def save_clipboard_image_to_file(directory: Path | None) -> Path | None:
    """Save a platform clipboard image into *directory* and return its path.

    Screenshot and image-copy workflows on macOS and Windows can put bitmap
    data on the clipboard instead of a file path. Textual's terminal paste
    events only carry text, so the chat input uses this helper from its
    explicit paste action to bridge bitmap clipboard contents back into
    Chrys' existing ``@file`` image attachment flow. Browser-host sessions
    must not read the server machine's clipboard because it is unrelated to
    the browser client's clipboard and may belong to a different user.
    """
    if directory is None or running_under_textual_web_driver():
        return None
    info = get_platform()
    if not (info.is_macos or info.is_windows):
        return None
    try:
        clipboard_payload = _grab_clipboard()
        return _path_from_clipboard_payload(clipboard_payload, directory=directory)
    except OSError, RuntimeError, ValueError:
        return None


def _grab_clipboard() -> object:
    if get_platform().is_macos:
        return _grab_macos_clipboard_image()
    return _grab_windows_clipboard()


def _grab_windows_clipboard() -> object:
    """Read the Windows clipboard with retries for transient clipboard-owner races."""
    from PIL import ImageGrab

    for attempt in range(_WINDOWS_CLIPBOARD_GRAB_ATTEMPTS):
        try:
            return ImageGrab.grabclipboard()
        except OSError:
            if attempt == _WINDOWS_CLIPBOARD_GRAB_ATTEMPTS - 1:
                raise
            time.sleep(_WINDOWS_CLIPBOARD_GRAB_DELAY_SECONDS)

    return None


def _grab_macos_clipboard_image() -> object | None:
    from PIL import Image, UnidentifiedImageError

    for class_code in _MACOS_CLIPBOARD_IMAGE_CLASSES:
        data = _read_macos_clipboard_class(class_code)
        if data is None:
            continue
        try:
            with Image.open(BytesIO(data), formats=_MACOS_CLIPBOARD_IMAGE_FORMATS) as image:
                image.load()
                return image.copy()
        except UnidentifiedImageError, OSError, ValueError:
            continue
    return None


def _read_macos_clipboard_class(class_code: str) -> bytes | None:
    try:
        result = subprocess.run(  # noqa: S603 - fixed osascript path and fixed AppleScript class codes.
            [
                "/usr/bin/osascript",
                "-e",
                f"get the clipboard as {_MACOS_CLASS_OPEN}class {class_code}{_MACOS_CLASS_CLOSE}",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=2,
        )
    except subprocess.SubprocessError, FileNotFoundError, OSError:
        return None
    if result.returncode != 0:
        return None
    return _decode_macos_clipboard_data(result.stdout)


def _decode_macos_clipboard_data(stdout: bytes) -> bytes | None:
    prefix = b"\xc2\xabdata "
    suffix = b"\xc2\xbb"
    start = stdout.find(prefix)
    if start < 0:
        return None
    hex_start = start + len(prefix) + 4
    end = stdout.find(suffix, hex_start)
    if end < 0:
        return None
    try:
        return unhexlify(stdout[hex_start:end].strip())
    except BinasciiError:
        return None


def _path_from_clipboard_payload(payload: object, *, directory: Path) -> Path | None:
    from PIL import Image

    if isinstance(payload, Image.Image):
        return _save_clipboard_image(payload, directory=directory)
    if isinstance(payload, list):
        return _first_supported_clipboard_file_path(payload)
    return None


def _save_clipboard_image(image: Image.Image, *, directory: Path) -> Path | None:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    path = directory / f"clipboard-image-{uuid.uuid4().hex}.png"
    converted: Image.Image | None = None
    try:
        image_to_save = image
        if image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA"}:
            converted = image.convert("RGBA")
            image_to_save = converted
        image_to_save.save(path, format="PNG")
        saved = path.resolve(strict=True)
        _prune_clipboard_image_dir(directory, keep=saved)
        return saved
    except OSError, RuntimeError, ValueError:
        return None
    finally:
        if converted is not None:
            converted.close()


def _prune_clipboard_image_dir(directory: Path, *, keep: Path) -> None:
    """Bound session/temp clipboard-image buildup while keeping recent pasted files available."""
    entries: list[tuple[int, int, Path]] = []
    try:
        candidates = list(directory.glob("clipboard-image-*.png"))
    except OSError:
        return
    for path in candidates:
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append((stat.st_mtime_ns, stat.st_size, path))
    entries.sort(key=lambda entry: entry[0], reverse=True)

    total_size = 0
    for index, (_mtime_ns, size, path) in enumerate(entries):
        total_size += size
        if path == keep or (index < _CLIPBOARD_IMAGE_KEEP_COUNT and total_size <= _CLIPBOARD_IMAGE_KEEP_BYTES):
            continue
        with suppress(OSError):
            path.unlink()


def _first_supported_clipboard_file_path(paths: Sequence[object]) -> Path | None:
    cwd = Path(safe_getcwd())
    for candidate in paths:
        if not isinstance(candidate, str):
            continue
        path = _resolve_supported_image_path(candidate, cwd=cwd)
        if path is not None:
            return path
    return None


def _parse_image_path_payload(text: str, *, cwd: Path) -> list[Path]:
    stripped = text.strip()
    if not stripped:
        return []

    for candidates in _candidate_path_groups(stripped):
        paths = _resolve_supported_image_group(candidates, cwd=cwd)
        if paths:
            return paths
    return []


def _candidate_path_groups(text: str) -> list[list[str]]:
    groups: list[list[str]] = [[text]]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        groups.append(lines)

    split_tokens = text.split()
    if len(split_tokens) > 1:
        groups.append(split_tokens)

    if os.name != "nt":
        shlex_tokens = _shlex_split(text)
        if shlex_tokens and shlex_tokens not in groups:
            groups.append(shlex_tokens)

    return groups


def _shlex_split(text: str) -> list[str]:
    try:
        return shlex.split(text)
    except ValueError:
        return []


def _resolve_supported_image_group(candidates: list[str], *, cwd: Path) -> list[Path]:
    paths: list[Path] = []
    for candidate in candidates:
        path = _resolve_supported_image_path(candidate, cwd=cwd)
        if path is None:
            return []
        paths.append(path)
    return paths


def _resolve_supported_image_path(candidate: str, *, cwd: Path) -> Path | None:
    quoted = _candidate_is_quoted(candidate)
    path_text = _path_text_from_candidate(candidate)
    if not _path_text_has_supported_image_suffix(path_text) or not _path_text_looks_like_single_path(
        path_text,
        quoted=quoted,
    ):
        return None
    try:
        path = Path(path_text).expanduser()
    except RuntimeError:
        return None
    if path.suffix.lower() not in IMAGE_MIME_BY_EXTENSION:
        return None
    if not path.is_absolute():
        path = cwd / path
    try:
        resolved = path.resolve(strict=True)
    except OSError, ValueError:
        return None
    if not resolved.is_file():
        return None
    return resolved


def _path_text_has_supported_image_suffix(path_text: str) -> bool:
    if not path_text:
        return False
    try:
        suffix = Path(path_text).suffix.lower()
    except ValueError:
        return False
    return suffix in IMAGE_MIME_BY_EXTENSION


def _path_text_looks_like_single_path(path_text: str, *, quoted: bool) -> bool:
    if not any(char.isspace() for char in path_text):
        return True
    try:
        expanded_path = Path(path_text).expanduser()
    except RuntimeError, ValueError:
        return False
    return (
        quoted or expanded_path.is_absolute() or path_text.startswith("~") or _path_text_is_windows_absolute(path_text)
    )


def _path_text_is_windows_absolute(path_text: str) -> bool:
    return (len(path_text) >= 3 and path_text[1:3] in (":\\", ":/")) or path_text.startswith(("\\\\", "//"))


def _candidate_is_quoted(candidate: str) -> bool:
    text = candidate.strip()
    return len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"')


def _path_text_from_candidate(candidate: str) -> str:
    text = candidate.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1]
    try:
        parsed = urlparse(text)
    except ValueError:
        return text
    if parsed.scheme == "file":
        path = parsed.path
        if parsed.netloc and parsed.netloc != "localhost":
            path = f"//{parsed.netloc}{path}"
        try:
            return url2pathname(path)
        except URLError, ValueError:
            return ""
    return text

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Runtime executable path helpers.

PyApp-packaged Chrys runs from an embedded Python distribution.  On some
platforms that distribution's executable directories appear early on PATH,
causing child shells and subprocess tools to resolve ``python`` / ``uv`` to
Chrys's private runtime instead of the user's system install.  These helpers
demote those runtime directories only for packaged runs while preserving them
as a fallback.
"""

from __future__ import annotations

import os
import shutil
import sys
import sysconfig
from collections.abc import Mapping, MutableMapping
from pathlib import Path

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_WINDOWS_DEFAULT_PATHEXT = (".COM", ".EXE", ".BAT", ".CMD")
_PYTHON_HOME_OVERRIDE_KEYS = frozenset({"PYTHONHOME"})
_PYTHON_RUNTIME_OVERRIDE_KEYS = frozenset({"PYTHONHOME", "PYTHONPATH"})


def is_frozen_runtime(env: Mapping[str, str] | None = None) -> bool:
    """Return whether this process is running from a packaged PyApp runtime."""
    source = os.environ if env is None else env
    forced = source.get("CHRYS_FORCE_FROZEN", "").strip().lower()
    return forced in _TRUE_VALUES or bool(source.get("PYAPP"))


def normalize_path(path: str | os.PathLike[str]) -> str:
    """Return an absolute path for comparison/display, or ``""`` for empty inputs."""
    raw = os.fspath(path)
    if not raw:
        return ""
    return os.path.abspath(os.path.expanduser(raw))


def same_path(left: str | os.PathLike[str], right: str | os.PathLike[str]) -> bool:
    """Compare paths using platform normalization."""
    left_norm = normalize_path(left)
    right_norm = normalize_path(right)
    if not left_norm or not right_norm:
        return left_norm == right_norm
    return os.path.normcase(os.path.realpath(left_norm)) == os.path.normcase(os.path.realpath(right_norm))


def runtime_executable_dirs() -> tuple[str, ...]:
    """Return executable/search directories associated with Chrys's Python runtime."""
    dirs: list[str] = []

    executable = normalize_path(sys.executable)
    if executable:
        _append_unique_path(dirs, str(Path(executable).parent))

    for raw_prefix in (sys.prefix, sys.exec_prefix):
        if raw_prefix:
            _append_unique_path(dirs, raw_prefix)

    scripts = sysconfig.get_path("scripts")
    if scripts:
        _append_unique_path(dirs, scripts)

    script_dir_name = "Scripts" if sys.platform == "win32" else "bin"
    for raw_prefix in (sys.prefix, sys.exec_prefix):
        if raw_prefix:
            _append_unique_path(dirs, str(Path(raw_prefix) / script_dir_name))
            if sys.platform == "win32":
                _append_unique_path(dirs, str(Path(raw_prefix) / "Library" / "bin"))

    return tuple(dirs)


def path_env_key(env: Mapping[str, str]) -> str | None:
    """Return the actual PATH key casing present in *env*."""
    if sys.platform == "win32":
        # Plain dicts can contain both Path and PATH after merges; mirror
        # Windows' effective last-writer-wins behavior for that degenerate case.
        keys = _path_env_keys(env)
        return keys[-1] if keys else None
    if "PATH" in env:
        return "PATH"
    keys = _path_env_keys(env)
    if keys:
        return keys[0]
    return None


def strip_python_runtime_overrides(
    env: MutableMapping[str, str],
    *,
    strip_pythonpath: bool = True,
) -> MutableMapping[str, str]:
    """Remove inherited Python runtime path overrides from a child environment.

    This is intentionally not gated on :func:`is_frozen_runtime`: a normal
    Chrys process may spawn a PyApp or standalone Python command whose CPython
    bootstrap would otherwise honor the parent's ``PYTHONHOME`` / ``PYTHONPATH``
    before the child application can defend itself.

    ``strip_pythonpath=False`` keeps user module search paths for managed
    commands that do not have their own explicit environment override.
    """
    blocked_keys = _PYTHON_RUNTIME_OVERRIDE_KEYS if strip_pythonpath else _PYTHON_HOME_OVERRIDE_KEYS
    for key in list(env):
        if key.upper() in blocked_keys:
            del env[key]
    return env


def reorder_path_demoting_runtime(env: MutableMapping[str, str]) -> MutableMapping[str, str]:
    """Move runtime PATH entries behind user/system entries for packaged runs.

    The operation is a stable partition: non-runtime entries keep their
    original relative order at the front; runtime entries keep their original
    relative order at the back.  Duplicate entries are preserved.
    """
    if not is_frozen_runtime(env):
        return env

    key = path_env_key(env)
    if key is None:
        return env
    _drop_duplicate_path_keys(env, keep=key)

    entries = _path_entries(env[key])
    if not entries:
        env[key] = ""
        return env

    runtime_dirs = runtime_executable_dirs()
    if not runtime_dirs:
        return env

    normal_entries: list[str] = []
    runtime_entries: list[str] = []
    for entry in entries:
        if _is_runtime_dir(entry, runtime_dirs):
            runtime_entries.append(entry)
        else:
            normal_entries.append(entry)

    env[key] = os.pathsep.join([*normal_entries, *runtime_entries])
    return env


def which_excluding_runtime(
    name: str,
    *,
    env: Mapping[str, str] | None = None,
    frozen_only: bool = True,
    fallback_to_runtime: bool = True,
    exclude_paths: tuple[str, ...] = (),
) -> str:
    """Resolve *name* from PATH, preferring non-runtime directories.

    By default the runtime-dir preference only activates for packaged PyApp
    runs.  ``fallback_to_runtime`` keeps Chrys's embedded tools available when
    the system does not provide the requested executable.
    """
    source = os.environ if env is None else env
    key = path_env_key(source)
    if key is None:
        return normalize_path(shutil.which(name) or "") if env is None else ""

    entries = _path_entries(source[key])
    if not entries:
        return ""

    should_exclude = not frozen_only or is_frozen_runtime(source)
    if not should_exclude:
        return _which_on_path_entries(name, entries, env=source, exclude_paths=exclude_paths)

    runtime_dirs = runtime_executable_dirs()
    non_runtime_entries = [entry for entry in entries if not _is_runtime_dir(entry, runtime_dirs)]
    found = _which_on_path_entries(name, non_runtime_entries, env=source, exclude_paths=exclude_paths)
    if found or not fallback_to_runtime:
        return found
    return _which_on_path_entries(name, entries, env=source, exclude_paths=exclude_paths)


def _append_unique_path(paths: list[str], new_path: str | os.PathLike[str]) -> None:
    normalized = normalize_path(new_path)
    if not normalized:
        return
    if any(same_path(normalized, existing) for existing in paths):
        return
    paths.append(normalized)


def _path_entries(path_value: str) -> list[str]:
    return [entry for entry in path_value.split(os.pathsep) if entry]


def _path_env_keys(env: Mapping[str, str]) -> list[str]:
    return [key for key in env if key.upper() == "PATH"]


def _drop_duplicate_path_keys(env: MutableMapping[str, str], *, keep: str) -> None:
    if sys.platform != "win32":
        return
    for key in list(env):
        if key != keep and key.upper() == "PATH":
            del env[key]


def _is_runtime_dir(path: str, runtime_dirs: tuple[str, ...]) -> bool:
    return any(same_path(path, runtime_dir) for runtime_dir in runtime_dirs)


def _which_on_path_entries(
    name: str,
    entries: list[str],
    *,
    env: Mapping[str, str],
    exclude_paths: tuple[str, ...],
) -> str:
    for raw_dir in entries:
        directory = normalize_path(raw_dir)
        if not directory:
            continue
        path = _which_in_directory(name, directory, env=env)
        if not path:
            continue
        if any(same_path(path, excluded_path) for excluded_path in exclude_paths if excluded_path):
            continue
        return path
    return ""


def _which_in_directory(name: str, directory: str, *, env: Mapping[str, str]) -> str:
    if sys.platform != "win32":
        return normalize_path(shutil.which(name, path=directory) or "")

    directory_entries = _windows_directory_entries(Path(directory))
    for candidate_name in _windows_candidate_names(name, env):
        path = directory_entries.get(candidate_name.casefold()) or normalize_path(Path(directory) / candidate_name)
        if _access_check(path):
            return path
    return ""


def _windows_directory_entries(directory: Path) -> dict[str, str]:
    """Return child paths keyed case-insensitively, preserving on-disk casing."""
    try:
        return {child.name.casefold(): normalize_path(child) for child in directory.iterdir()}
    except OSError:
        return {}


def _windows_candidate_names(name: str, env: Mapping[str, str]) -> tuple[str, ...]:
    pathext = _windows_pathext(env)
    suffix = Path(name).suffix
    names: list[str] = []
    if suffix and any(suffix.upper() == ext.upper() for ext in pathext):
        names.append(name)
    names.extend(f"{name}{ext}" for ext in pathext)
    return tuple(dict.fromkeys(names))


def _windows_pathext(env: Mapping[str, str]) -> tuple[str, ...]:
    raw = _case_insensitive_env_value(env, "PATHEXT")
    if raw is None:
        return _WINDOWS_DEFAULT_PATHEXT
    exts = tuple(ext.rstrip(".") for ext in raw.split(";") if ext)
    return exts or _WINDOWS_DEFAULT_PATHEXT


def _case_insensitive_env_value(env: Mapping[str, str], key: str) -> str | None:
    for env_key, value in env.items():
        if env_key.upper() == key.upper():
            return value
    return None


def _access_check(path: str) -> bool:
    return os.path.exists(path) and os.access(path, os.F_OK | os.X_OK) and not os.path.isdir(path)

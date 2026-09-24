# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path

import pytest

from chrys.foundation.platform import get_platform
from chrys.foundation.platform.files import SecureFileError
from chrys.foundation.trajectory.fingerprint import (
    DOMAIN_TOOL_ARGUMENTS,
    DOMAIN_TOOL_CONTENT,
    FINGERPRINT_KEY_BYTES,
    canonical_json_bytes,
    fingerprint_json,
    fingerprint_text,
    keyed_fingerprint,
)
from chrys.foundation.trajectory.keys import (
    TRAJECTORY_KEY_FILE_NAME,
    ensure_owner_only_directory,
    load_or_create_fingerprint_key,
)


def test_load_or_create_key_is_stable_owner_only_and_32_bytes(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    key = load_or_create_fingerprint_key(config_dir)
    assert len(key) == FINGERPRINT_KEY_BYTES
    key_path = config_dir / TRAJECTORY_KEY_FILE_NAME
    assert key_path.read_bytes() == key
    if not get_platform().is_windows:
        assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    assert load_or_create_fingerprint_key(config_dir) == key
    # No temp files linger.
    assert [p.name for p in config_dir.iterdir() if p.name.endswith(".tmp")] == []


def test_existing_key_is_never_overwritten(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    first = load_or_create_fingerprint_key(config_dir)
    second = load_or_create_fingerprint_key(config_dir)
    assert first == second


def test_invalid_existing_key_is_rejected_not_repaired(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    key_path = config_dir / TRAJECTORY_KEY_FILE_NAME
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT, 0o600)
    os.write(fd, b"short")
    os.close(fd)
    with pytest.raises(SecureFileError):
        load_or_create_fingerprint_key(config_dir)
    assert key_path.read_bytes() == b"short"


async def test_concurrent_creators_agree_on_one_key(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    keys = await asyncio.gather(*(asyncio.to_thread(load_or_create_fingerprint_key, config_dir) for _ in range(8)))
    assert len(set(keys)) == 1


def test_ensure_owner_only_directory_creates_and_tightens(tmp_path: Path) -> None:
    target = tmp_path / "session" / "trajectory"
    ensure_owner_only_directory(target)
    assert target.is_dir()
    if get_platform().is_windows:
        return
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    target.chmod(0o755)
    ensure_owner_only_directory(target)
    assert stat.S_IMODE(target.stat().st_mode) == 0o700


def test_ensure_owner_only_directory_rejects_a_file(tmp_path: Path) -> None:
    target = tmp_path / "not-a-dir"
    target.write_bytes(b"")
    with pytest.raises(OSError):
        ensure_owner_only_directory(target)


def test_fingerprints_are_keyed_and_domain_separated() -> None:
    key_a = b"a" * 32
    key_b = b"b" * 32
    data = b"same-input"
    assert keyed_fingerprint(key_a, DOMAIN_TOOL_ARGUMENTS, data) != keyed_fingerprint(
        key_b, DOMAIN_TOOL_ARGUMENTS, data
    )
    assert keyed_fingerprint(key_a, DOMAIN_TOOL_ARGUMENTS, data) != keyed_fingerprint(key_a, DOMAIN_TOOL_CONTENT, data)
    assert keyed_fingerprint(key_a, DOMAIN_TOOL_ARGUMENTS, data) == keyed_fingerprint(
        key_a, DOMAIN_TOOL_ARGUMENTS, data
    )
    assert len(keyed_fingerprint(key_a, DOMAIN_TOOL_ARGUMENTS, data)) == 64
    with pytest.raises(ValueError):
        keyed_fingerprint(b"", DOMAIN_TOOL_ARGUMENTS, data)
    with pytest.raises(ValueError):
        keyed_fingerprint(key_a, "bad\x00domain", data)


def test_canonical_json_is_order_independent_and_injective_for_surrogates() -> None:
    assert canonical_json_bytes({"b": 1, "a": [1, 2]}) == canonical_json_bytes({"a": [1, 2], "b": 1})
    assert canonical_json_bytes({"s": "\udcff"}) != canonical_json_bytes({"s": "\\udcff"})
    key = b"k" * 32
    assert fingerprint_json(key, DOMAIN_TOOL_ARGUMENTS, {"x": 1}) == fingerprint_json(
        key, DOMAIN_TOOL_ARGUMENTS, {"x": 1}
    )
    assert fingerprint_text(key, DOMAIN_TOOL_CONTENT, "hi\udcff") == fingerprint_text(
        key, DOMAIN_TOOL_CONTENT, "hi\udcff"
    )
    # ...and the escape a user typed is not the surrogate it spells.
    assert fingerprint_text(key, DOMAIN_TOOL_CONTENT, "hi\udcff") != fingerprint_text(
        key, DOMAIN_TOOL_CONTENT, "hi\\udcff"
    )

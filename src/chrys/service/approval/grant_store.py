# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Bounded owner-only JSON grants, re-read under the shared file lock before writes."""

from __future__ import annotations

import errno
import json
import logging
from pathlib import Path
from typing import Any

from chrys.foundation.config.settings import resolve_sessions_dir
from chrys.foundation.platform.files import atomic_write_owner_only_text, secure_open_owner_only_binary
from chrys.foundation.util.lock import FileLock
from chrys.foundation.util.session_ids import session_short_id

logger = logging.getLogger(__name__)
GRANTS_FILE = "approval-grants.json"
MAX_GRANTS = 1000
MAX_BYTES = 4 * 1024 * 1024


def session_grants_path(config_dir: Path, session_id: str) -> Path:
    short_id = session_short_id(session_id)
    if short_id in {"", ".", ".."}:
        raise ValueError("A session ID is required")
    return resolve_sessions_dir(config_dir, create=False) / short_id / GRANTS_FILE


class ApprovalGrantStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def lock_path(self) -> Path:
        return self.path.with_name(self.path.name + ".lock")

    def _read(self) -> list[dict[str, Any]]:
        try:
            with secure_open_owner_only_binary(self.path) as handle:
                raw = handle.read(MAX_BYTES + 1)
        except OSError as exc:
            if exc.errno in (errno.ENOENT, errno.ENOTDIR):
                return []
            raise
        if len(raw) > MAX_BYTES:
            raise ValueError("Grant store is too large")
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("rules"), list):
            raise ValueError("Unknown grant schema")
        rules = data["rules"]
        if len(rules) > MAX_GRANTS or any(not isinstance(rule, dict) for rule in rules):
            raise ValueError("Invalid grant records")
        return rules

    def load(self) -> list[dict[str, Any]]:
        try:
            return self._read()
        except OSError, ValueError, RecursionError:
            logger.warning("Approval grants could not be read; no grants reused")
            return []

    def _update(
        self, additions: list[dict[str, Any]], *, remove: str = "", clear: bool = False, project: str | None = None
    ) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(self.lock_path, timeout=1):
                rules = self._read()
                kept = [
                    rule
                    for rule in rules
                    if not (
                        (clear and (project is None or rule.get("project") == project))
                        or (remove and rule.get("id") == remove)
                    )
                ]
                if remove and len(kept) == len(rules):
                    return False
                # Identical grants replace themselves rather than consuming the bound.
                for new in additions:
                    kept = [
                        old
                        for old in kept
                        if any(old.get(key) != new.get(key) for key in ("scope", "scope_id", "prefix", "key"))
                    ]
                    kept.append(new)
                payload = json.dumps({"version": 1, "rules": kept}, ensure_ascii=True, allow_nan=False, indent=2) + "\n"
                if len(kept) > MAX_GRANTS or len(payload.encode("utf-8")) > MAX_BYTES:
                    return False
                atomic_write_owner_only_text(self.path, payload)
            return True
        except OSError, ValueError, TypeError, RecursionError:
            logger.warning("Approval grants could not be updated")
            return False

    def add_many(self, rules: list[dict[str, Any]]) -> bool:
        return self._update(rules)

    def revoke(self, rule_id: str) -> bool:
        return bool(rule_id) and self._update([], remove=rule_id)

    def clear(self, *, project: str | None = None) -> bool:
        return self._update([], clear=True, project=project)

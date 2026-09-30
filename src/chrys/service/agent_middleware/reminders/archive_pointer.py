# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The ``archive`` catalog: a pointer to the records compaction archived from previous turns.

The record count and catalog path are captured when a turn is prepared
fresh and stay fixed for that turn; a restored retry reuses the persisted
count (``CATALOG_POINTER_RECORD_COUNT_STATE_KEY``).  A recorded pointer is
re-rendered at this session's catalog path (``retarget``), so a forked or
moved session names its own catalog.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from chrys.service.context.compaction.spill import SpillQuota

logger = logging.getLogger(__name__)

_STATE_INTEGER_MAX = (1 << 63) - 1
CATALOG_POINTER_RECORD_COUNT_STATE_KEY = "last_words_catalog_pointer_record_count"

_CATALOG_POINTER_SUFFIX = " (contains each record's relative path)."
# The whole pointer sentence, so a reminder that merely quotes part of it
# (a todo item, a hook note) is never taken for the pointer.
_CATALOG_POINTER = re.compile(
    r"(Earlier context compaction archived [1-9][0-9]* records? from previous turns; catalog: )"
    r".+" + re.escape(_CATALOG_POINTER_SUFFIX),
    re.DOTALL,
)


def _catalog_pointer_text(record_count: int, path: str) -> str:
    record_label = "record" if record_count == 1 else "records"
    return (
        f"Earlier context compaction archived {record_count} {record_label} from previous turns; "
        f"catalog: {path}{_CATALOG_POINTER_SUFFIX}"
    )


class ArchivePointerSource:
    """The archive pointer one middleware offers for its session."""

    name = "archive"
    """The catalog's record name."""

    def __init__(
        self,
        *,
        session_root: Path | None,
        file_read_available: bool,
        spill_quota: SpillQuota | None,
        enabled: bool,
    ) -> None:
        self._session_root = session_root
        self._session_catalog: str | None = None
        self._file_read_available = file_read_available
        self._spill_quota = spill_quota
        self._enabled = enabled
        self._record_count = 0
        self._path: str | None = None
        self._snapshot_valid = False
        self._restored_record_count: int | None = None

    def current_text(self) -> str | None:
        """The pointer this turn offers; None without archived records or a readable catalog."""
        return _catalog_pointer_text(self._record_count, self._path) if self._record_count and self._path else None

    @property
    def restored_record_count(self) -> int | None:
        """The count a restored retry stashed, until the next prepared turn discards it."""
        return self._restored_record_count

    def restore_record_count(self, value: object) -> None:
        """Stash a validated turn-start catalog count for a restored retry."""
        if value is None:
            self._restored_record_count = None
            return
        if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > _STATE_INTEGER_MAX:
            logger.debug("Dropping malformed dropped-record catalog pointer count")
            self._restored_record_count = None
            return
        self._restored_record_count = value

    def discard_restored(self) -> None:
        """Drop the stashed count: it is single-shot."""
        self._restored_record_count = None

    def record_count_state(self) -> int | None:
        """Return the persisted turn-start catalog count, including zero."""
        if self._snapshot_valid:
            return self._record_count
        return self._restored_record_count

    def capture_live(self, *, excluded_relative_paths: set[str] | None = None) -> None:
        """Capture the catalog count/path at a real turn boundary."""
        if not self._enabled or self._session_root is None or self._spill_quota is None:
            self.capture(0)
            return
        record_count = self._spill_quota.live_record_count(
            excluded_relative_paths=excluded_relative_paths or (),
        )
        self.capture(record_count)

    def capture(self, record_count: int) -> None:
        """Install one stable turn-start count and derive its read affordance."""
        self._record_count = record_count
        self._path = None
        self._snapshot_valid = True
        if (
            record_count
            and self._enabled
            and self._file_read_available
            and self._session_root is not None
            and self._spill_quota is not None
            and self._spill_quota.storage_available
        ):
            from chrys.service.context.compaction.spill import catalog_path_for_read

            if catalog_path_for_read(self._session_root) is not None:
                self._path = self._session_catalog_path()

    def retarget(self, text: str) -> str | None:
        """*text* naming this session's catalog, when it is a whole pointer sentence.

        The pointer names the catalog by absolute path; a session copied
        elsewhere (fork, a move to a new session root) points it at its own
        catalog, which changes the prefix once.  None when *text* is not a
        pointer or there is no session root.
        """
        pointer = _CATALOG_POINTER.fullmatch(text)
        current = self._session_catalog_path() if pointer is not None else None
        if pointer is None or current is None:
            return None
        return f"{pointer.group(1)}{current}{_CATALOG_POINTER_SUFFIX}"

    def _session_catalog_path(self) -> str | None:
        """This session's compaction catalog, by the absolute path reminders name it with.

        Resolved once: the session root is fixed for the middleware's life,
        and every call re-renders each recorded pointer with it.
        """
        if self._session_root is None:
            return None
        if self._session_catalog is None:
            from chrys.service.context.compaction.spill import CATALOG_RELATIVE_PATH

            self._session_catalog = (self._session_root.resolve(strict=False) / CATALOG_RELATIVE_PATH).as_posix()
        return self._session_catalog

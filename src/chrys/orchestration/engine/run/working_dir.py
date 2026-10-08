# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Refuse new work while the session working directory no longer exists.

The directory can be deleted or moved outside the app at any time. Nothing
watches it: a turn already running finishes on its own (tools report the
missing directory to the model), and the next fresh turn or retry is refused
here so the frontend can ask the user for another directory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from chrys.foundation.events.types import Error
from chrys.foundation.i18n import DisplayPath, msg
from chrys.foundation.platform.files import surrogate_safe_text

if TYPE_CHECKING:
    from chrys.foundation.events.bus import EventBus
    from chrys.orchestration.engine.state.active_session import ActiveSession

WORKING_DIR_MISSING_CODE = "working_dir_missing"

_WORKING_DIR_MISSING = msg(
    "engine.working_dir_missing",
    fallback="The working directory no longer exists: {path}",
)


async def refuse_while_working_dir_missing(bus: EventBus, session: ActiveSession) -> bool:
    """Publish an :class:`Error` and return True when the primary cwd is gone."""
    workspace = session.workspace
    missing = workspace.missing_primary() if workspace is not None else None
    if missing is None:
        return False
    await publish_working_dir_missing(bus, session.session_id, missing)
    return True


async def publish_working_dir_missing(bus: EventBus, session_id: str | None, path: str) -> None:
    """Publish the :class:`Error` that tells frontends *path* is gone."""
    await bus.publish(
        Error(
            code=WORKING_DIR_MISSING_CODE,
            message=f"Working directory no longer exists: {surrogate_safe_text(path)}",
            display_message=_WORKING_DIR_MISSING.bind(path=DisplayPath(path)),
            session_id=session_id,
        )
    )

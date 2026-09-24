# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Trajectory prelude of a forked session.

A fork copies the conversation but never the parent's ``trajectory/`` (the
parent's writer may be live, and the fork's events must start at sequence 1
with their own runtime). Instead the fork opens its own log with a complete
mini runtime — ``session.started`` then ``session.forked`` pointing back at
the origin and the parent sequence the fork branched from — and closes it
again, so a later restore of the fork resumes a cleanly closed runtime.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from chrys.foundation.trajectory.event_types import EventType, RuntimeFinishReason
from chrys.foundation.trajectory.writer import EmitResult
from chrys.service.trajectory.session import SessionStartInfo, SessionTrajectory

logger = logging.getLogger(__name__)

_ABANDONED_PRELUDES: set[asyncio.Task[bool]] = set()
"""Preludes whose caller was cancelled, kept referenced until they close themselves."""


async def record_fork(
    *,
    fork_session_id: str,
    fork_session_dir: Path,
    fork_write_lock_path: Path | None,
    origin_session_id: str,
    forked_at_sequence: int,
    session_start_info: SessionStartInfo | None,
) -> bool:
    """Write the fork's opening runtime; ``True`` when every marker was written."""
    trajectory = SessionTrajectory(
        session_id=fork_session_id,
        session_dir=fork_session_dir,
        write_lock_path=fork_write_lock_path,
        session_start_info=(lambda: session_start_info) if session_start_info is not None else None,
        # The fork directory already holds session.json: without this the
        # prelude would read as "feature introduced on a legacy session".
        persisted_state_probe=lambda _path: False,
    )
    # The first emit activates the log — on a worker thread, which keeps
    # running after the caller stops waiting for it. A cancellation landing
    # there would leave that worker holding the fork's descriptor and its
    # writer lease with nobody left to close them, so the write and the close
    # are one task that always finishes, and cancellation only detaches the
    # caller from it.
    prelude = asyncio.ensure_future(_write_prelude(trajectory, origin_session_id, forked_at_sequence))
    try:
        return await asyncio.shield(prelude)
    except asyncio.CancelledError:
        _ABANDONED_PRELUDES.add(prelude)
        prelude.add_done_callback(_ABANDONED_PRELUDES.discard)
        raise


async def _write_prelude(trajectory: SessionTrajectory, origin_session_id: str, forked_at_sequence: int) -> bool:
    try:
        result = await trajectory.emit(
            trajectory.main_actor_draft(
                EventType.SESSION_FORKED,
                payload={"origin_session_id": origin_session_id, "forked_at_sequence": forked_at_sequence},
            )
        )
    except Exception:
        logger.warning("Failed to record fork prelude for session %s", trajectory.session_id, exc_info=True)
        result = EmitResult.DEGRADED
    closed = await trajectory.close(reason=RuntimeFinishReason.SESSION_SWITCH)
    return result is EmitResult.WRITTEN and closed

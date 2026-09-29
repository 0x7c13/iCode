# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A real turn — fresh or retried — records its launch's surface on the saved session."""

from __future__ import annotations

from chrys.foundation.events.types import SessionNew, SessionRestore
from chrys.foundation.models.session_surface import SessionSurface
from chrys.service.llm.mock import MockResponse
from tests.support.pipeline_helpers import error_on_nth


async def test_a_turn_records_the_launch_surface(make_pipeline_ctx) -> None:
    ctx = await make_pipeline_ctx([MockResponse(text="done")], surface=SessionSurface.CLI)

    await ctx.send_message("hello")

    meta = await ctx.store.load_session_meta(ctx.session_id)
    assert meta is not None and meta.message_count >= 1
    assert meta.last_surface is SessionSurface.CLI


async def test_a_retried_turn_records_the_launch_surface(make_pipeline_ctx) -> None:
    ctx = await make_pipeline_ctx([MockResponse(text="retried")], surface=SessionSurface.ACP)
    restore_client = error_on_nth(ctx, 1)
    await ctx.send_message("hello")
    assert ctx.engine.current.loaded.bindings.state.run_failed
    restore_client()
    # What reopening the failed session does: the mark is retired, and the
    # recorded surface is whichever launch worked in the session last.
    ctx.engine.session.adopt_restore_identity(session_id=ctx.session_id, recovered_from_sidecar=False)
    await ctx.store.save_session(
        ctx.session_id, await ctx.store.load_session(ctx.session_id), last_surface=SessionSurface.TUI
    )

    await ctx.send_retry()

    meta = await ctx.store.load_session_meta(ctx.session_id)
    assert meta is not None and meta.last_surface is SessionSurface.ACP


async def test_coming_back_to_a_session_records_nothing_until_the_next_turn(make_pipeline_ctx) -> None:
    """Another launch may work in a session while this one is elsewhere; saving it on return is not a turn."""
    ctx = await make_pipeline_ctx([MockResponse(text="first"), MockResponse(text="again")], surface=SessionSurface.TUI)
    await ctx.send_message("hello")
    session_a = ctx.engine.session.session_id
    assert session_a is not None
    assert (await ctx.store.load_session_meta(session_a)).last_surface is SessionSurface.TUI

    await ctx.bus.publish(SessionNew())
    assert ctx.engine.session.session_id != session_a
    await ctx.store.save_session(session_a, await ctx.store.load_session(session_a), last_surface=SessionSurface.CLI)
    await ctx.engine.on_session_restore(SessionRestore(session_id=session_a))
    assert ctx.engine.session.session_id == session_a

    assert await ctx.engine.writer.save_current_session() is True
    assert (await ctx.store.load_session_meta(session_a)).last_surface is SessionSurface.CLI

    await ctx.send_message("again")
    assert (await ctx.store.load_session_meta(session_a)).last_surface is SessionSurface.TUI

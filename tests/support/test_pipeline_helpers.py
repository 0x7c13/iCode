# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the shared pipeline test helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import create_autospec

import pytest

from chrys.foundation.events.types import ApprovalRequest, SessionRestore
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.models.workspace import Workspace
from chrys.service.llm.mock import MockResponse
from chrys.service.mutations.workspace_changes import WorkspaceChangeTracker
from tests.support.pipeline_helpers import create_test_engine, extract_final_messages


@pytest.mark.asyncio
@pytest.mark.parametrize("custom_workspace", [False, True], ids=["default-workspace", "explicit-workspace"])
async def test_mock_turn_is_independent_of_ambient_workspace_scans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, custom_workspace: bool
) -> None:
    ambient = tmp_path / "ambient"
    ambient.mkdir()
    monkeypatch.chdir(ambient)
    monkeypatch.setenv("CHRYS_WORKSPACE_CHANGE_NOTICE", "1")
    workdir = tmp_path / "custom" if custom_workspace else tmp_path
    workdir.mkdir(exist_ok=True)
    compute_notice = create_autospec(
        WorkspaceChangeTracker.compute_turn_notice, side_effect=WorkspaceChangeTracker.compute_turn_notice
    )
    capture_baseline = create_autospec(
        WorkspaceChangeTracker.capture_baseline, side_effect=WorkspaceChangeTracker.capture_baseline
    )
    monkeypatch.setattr(WorkspaceChangeTracker, "compute_turn_notice", compute_notice)
    monkeypatch.setattr(WorkspaceChangeTracker, "capture_baseline", capture_baseline)
    ctx = await create_test_engine(
        [MockResponse(text="done"), MockResponse(text="restored")],
        tmp_path,
        workspace=Workspace.from_cwd(str(workdir)) if custom_workspace else None,
    )
    try:
        await ctx.send_message("hello")
        await ctx.bus.publish(SessionRestore(session_id=ctx.session_id), raise_handler_errors=True)
        await ctx.send_message("again")
        saved = await ctx.get_session_messages()
        assert [
            message["role"]
            for message in saved
            if not message.get("additional_properties", {}).get(HistoryMarkerKind.KEY)
        ] == ["user", "assistant", "user", "assistant"]
        assert extract_final_messages(ctx.events) == ["done", "restored"]
        assert ctx.mock_client.call_count == 2
        assert ctx.engine.session.workspace is not None
        assert ctx.engine.session.workspace.primary_cwd == str(workdir)
        compute_notice.assert_not_called()
        capture_baseline.assert_not_called()
    finally:
        await ctx.cleanup()


class TestPendingApprovalDetection:
    """wait_for_idle's blocked-on-approval discriminator trusts the request.

    The production policy decides whether approval is needed (name overrides,
    kind overrides like "shell", qualified "kind.tool" keys, sensitive-access
    triggers) — a published ApprovalRequest already encodes that decision, so
    the helper must not re-derive the mode from the profile config: a
    kind-based override would resolve to "auto" by tool name and wrongly wait
    the full timeout on a run that only the test can unblock.
    """

    def _ctx_with_events(self, events):
        from tests.support.pipeline_helpers import PipelineTestContext

        return PipelineTestContext(
            engine=None, bus=None, events=list(events), mock_client=None, store=None, session_id=""
        )

    def test_unresponded_request_pends_regardless_of_override_shape(self):
        from tests.support.pipeline_helpers import _pending_user_approval

        # tool_name deliberately matches no override key — as with a
        # kind-based override ("shell") or a sensitive-access trigger.
        ctx = self._ctx_with_events([ApprovalRequest(request_id="r1", tool_name="run_command", tool_kind="shell")])
        assert _pending_user_approval(ctx) is True

    def test_judging_request_defers_to_the_verdict(self):
        from chrys.foundation.events.types import ApprovalReviewed
        from tests.support.pipeline_helpers import _pending_user_approval

        request = ApprovalRequest(request_id="r1", tool_name="run_command", tool_kind="shell", judging=True)

        # No verdict yet: the judge may still auto-approve (resolving the
        # middleware future without any ApprovalResponse), so keep waiting.
        assert _pending_user_approval(self._ctx_with_events([request])) is False

        # Approved verdict: the judge fulfilled the future — run progresses.
        approved = self._ctx_with_events([request, ApprovalReviewed(request_id="r1", approved=True)])
        assert _pending_user_approval(approved) is False

        # Flagged verdict (incl. judge errors): _run_judge leaves the future
        # unresolved for the user, so the request pends on the test again.
        flagged = self._ctx_with_events([request, ApprovalReviewed(request_id="r1", approved=False)])
        assert _pending_user_approval(flagged) is True

        # A verdict for a different request does not re-block this one.
        other = self._ctx_with_events([request, ApprovalReviewed(request_id="r9", approved=False)])
        assert _pending_user_approval(other) is False

    def test_responded_and_interrupted_requests_do_not_pend(self):
        from chrys.foundation.events.types import ApprovalResponse, UserInterrupt
        from tests.support.pipeline_helpers import _pending_user_approval

        responded = self._ctx_with_events(
            [
                ApprovalRequest(request_id="r1", tool_name="guarded_echo"),
                ApprovalResponse(request_id="r1", approved=True),
            ]
        )
        assert _pending_user_approval(responded) is False

        interrupted = self._ctx_with_events(
            [ApprovalRequest(request_id="r2", tool_name="guarded_echo"), UserInterrupt()]
        )
        assert _pending_user_approval(interrupted) is False

        # A request published after the interrupt (e.g. the resumed run) pends.
        resumed = self._ctx_with_events([UserInterrupt(), ApprovalRequest(request_id="r3", tool_name="guarded_echo")])
        assert _pending_user_approval(resumed) is True

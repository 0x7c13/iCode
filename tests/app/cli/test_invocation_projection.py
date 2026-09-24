# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""CLI output consumes only the owning turn's terminal text."""

from __future__ import annotations

import json

import pytest

from chrys.app.cli.run import _write_result
from chrys.foundation.events.types import InvocationMessage
from chrys.orchestration.session_host import ChrysSessionHost
from tests.support.invocation_events import ORIGINS, PHASES, projection_event


@pytest.mark.parametrize("as_json", (False, True))
async def test_cli_child_final_never_replaces_main_result(as_json: bool, capsys: pytest.CaptureFixture[str]) -> None:
    class ProjectedHost(ChrysSessionHost):
        @property
        def session_id(self):
            return "s1"

        async def iter_run_events(self, message):
            yield InvocationMessage(origin=ORIGINS[0], text="owned final", is_final=True)
            for origin in ORIGINS[1:]:
                for phase in PHASES:
                    yield projection_event(origin, phase)

    host = ProjectedHost.__new__(ProjectedHost)
    result = await host.run_until_final("prompt")
    assert result.text == "owned final"
    _write_result(result, as_json=as_json, duration=1)
    captured = capsys.readouterr()
    assert captured.err == ""
    if as_json:
        assert json.loads(captured.out) == {"session_id": "s1", "result": "owned final", "duration": 1}
    else:
        assert captured.out == "owned final\n"

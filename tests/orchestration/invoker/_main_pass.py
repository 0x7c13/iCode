# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Drive the main shell's request ports in tests that do not create a TurnRunner."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from chrys.kernel import Message
from chrys.orchestration.engine.run.bindings import TurnBindings


async def fresh_pass(executor: TurnBindings, contents: list[Any], created_at: datetime | str | None = None) -> None:
    executor.inputs.begin_invocation()
    request = executor.inputs.fresh_request(contents, created_at)
    executor.record_outcome(await executor.backend.run(request))


async def continuation_pass(executor: TurnBindings, messages: list[Message]) -> None:
    executor.record_outcome(await executor.backend.run(executor.inputs.continuation_request(messages)))


async def retry_pass(
    executor: TurnBindings, additional_text: str = "", created_at: datetime | str | None = None
) -> None:
    async with executor.inputs.retry_request(additional_text, created_at) as request:
        if request is not None:
            executor.record_outcome(await executor.backend.run(request))

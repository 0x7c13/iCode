# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Interactive retry fixture, installed only into temporary test projects.

Run from the TUI: select Test workflow playground and press Start. The flaky
branch fails twice, exhausts its automatic retry budget, and waits for Retry
on the graph or in its node details. The third attempt succeeds; the other branch is reused.
Each new workflow run gets a fresh worker, so the scenario resets every time.
All work is simulated in memory; no model, network, or project edits are used.
"""

from __future__ import annotations

import asyncio

from chrys.workflows import NodeContext, Retry, SourceValue, WorkflowBuilder, WorkflowValue

# Seconds per simulated step, slow enough for a person to watch each state. The TUI tests
# install a copy with a shorter pace: they need the order of events, not the waiting.
PACE = 0.8

_calls = {"prepare": 0, "healthy_check": 0, "flaky_check": 0}


def prepare(value: WorkflowValue, ctx: NodeContext) -> WorkflowValue:
    _calls["prepare"] += 1
    ctx.emit("Two parallel checks: one succeeds; the other waits for manual retry after two intentional failures.")
    return WorkflowValue(text=value.text.strip() or "Retry playground", data={"prepare_calls": _calls["prepare"]})


async def healthy_check(value: WorkflowValue, ctx: NodeContext) -> WorkflowValue:
    _calls["healthy_check"] += 1
    ctx.emit("Healthy branch: simulating a successful check.")
    await asyncio.sleep(PACE * 0.75)
    return WorkflowValue(text="Healthy branch passed.", data={"healthy_check_calls": _calls["healthy_check"]})


async def flaky_check(value: WorkflowValue, ctx: NodeContext) -> WorkflowValue:
    _calls["flaky_check"] += 1
    attempt = _calls["flaky_check"]
    ctx.emit(f"Attempt {attempt}: the first two attempts intentionally fail; attempt 3 will pass.")
    await asyncio.sleep(PACE)
    if attempt <= 2:
        raise RuntimeError(
            f"Intentional playground failure ({attempt}/2). "
            "After automatic retries are exhausted, click Retry on flaky_check or select it and press r. "
            "The next attempt will pass without rerunning the healthy branch."
        )
    return WorkflowValue(text="Flaky branch passed on manual retry.", data={"flaky_check_calls": attempt})


def combine_checks(sources: list[SourceValue]) -> WorkflowValue:
    return WorkflowValue(text="\n".join(source.value.text for source in sources))


def summary(value: WorkflowValue, ctx: NodeContext) -> WorkflowValue:
    ctx.emit("Both checks passed. The workflow continued from the retried node.")
    return WorkflowValue(
        text=(
            "# Retry playground complete\n\n"
            f"{value.text}\n\n"
            f"prepare: {_calls['prepare']} execution\n"
            f"healthy_check: {_calls['healthy_check']} execution\n"
            f"flaky_check: {_calls['flaky_check']} attempts (2 automatic + 1 manual)\n\n"
            "The completed branch was retained. Start a new run to repeat the demo. "
            "Cancel ends the whole run and does not offer resume."
        ),
        data=dict(_calls),
    )


wf = WorkflowBuilder(
    "Test workflow playground",
    description="Two intentional failures → awaiting retry → click Retry or press r → successful join",
)
prepared = wf.python("prepare", prepare)
healthy = wf.python("healthy_check", healthy_check)
flaky = wf.python("flaky_check", flaky_check, retry=Retry(max_attempts=2, backoff=PACE))
report = wf.python("summary", summary)
wf.start(prepared)
wf.edge(prepared, healthy)
wf.edge(prepared, flaky)
wf.join([healthy, flaky], report, combine=combine_checks)
wf.output(report)
workflow = wf.build()

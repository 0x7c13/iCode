# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Code refactor pipeline: edit with Code, check with QA, repeat until the check passes.

The input describes the requested refactor. Each iteration gives Code the
original request and the previous QA feedback. QA uses the project's checks
and reviews the diff, ending with exactly ``CHECK: PASS`` or ``CHECK: FAIL``.
The check's final line decides whether to exit; three unsuccessful iterations
fail the run as loop_exhausted. A side edge carries the request past the agents
so it survives their text-only outputs. Changes use the ordinary agent tools
and approvals; this workflow does not roll changes back on failure.
"""

from __future__ import annotations

from chrys.workflows import BuilderScope, NodeContext, NodeHandle, SourceValue, WorkflowBuilder, WorkflowValue


def prepare(value: WorkflowValue) -> WorkflowValue:
    text = value.text
    return WorkflowValue(text=text, data={"request": text, "iteration": 0})


def dispatch(value: WorkflowValue, ctx: NodeContext) -> WorkflowValue:
    data = value.data if isinstance(value.data, dict) else {}
    request = data.get("request", value.text)
    previous = data.get("iteration", 0)
    prompt = "Implement this refactor using the project tools and conventions:\n" + request
    if previous:
        prompt += "\n\nAddress the previous check feedback:\n" + value.text
    ctx.emit(f"Refactor iteration {previous + 1}")
    return WorkflowValue(text=prompt, data={"request": request, "iteration": previous + 1})


def carry_request(sources: list[SourceValue]) -> WorkflowValue:
    by_node = {source.node_id: source.value for source in sources}
    return WorkflowValue(text=by_node["check"].text, data=by_node["dispatch"].data)


def finish_check(value: WorkflowValue, ctx: NodeContext) -> WorkflowValue:
    ctx.emit("Check passed." if passed(value) else "Check needs another iteration.")
    return value


def passed(value: WorkflowValue) -> bool:
    return bool(value.text.strip()) and value.text.strip().splitlines()[-1] == "CHECK: PASS"


def body(scope: BuilderScope) -> tuple[NodeHandle, NodeHandle]:
    entry = scope.python("dispatch", dispatch)
    edit = scope.agent(
        "refactor", profile="Code", instructions_suffix="Make the requested refactor and describe the changes."
    )
    check = scope.agent(
        "check",
        profile="QA",
        instructions_suffix=(
            "Check this refactor: inspect the diff and run the relevant project checks. "
            "Describe failures for the next iteration. Finish with exactly CHECK: PASS or CHECK: FAIL."
        ),
    )
    exit_node = scope.python("finish_check", finish_check)
    scope.chain(entry, edit, check)
    scope.join([entry, check], exit_node, combine=carry_request)
    return entry, exit_node


wf = WorkflowBuilder("Code refactor pipeline", description="Code edits and QA checks, up to three iterations")
start = wf.python("prepare", prepare)
refine = wf.loop("refine", body, until=passed, max_iterations=3, on_exhausted="fail")
wf.start(start)
wf.chain(start, refine)
wf.output(refine)
workflow = wf.build()

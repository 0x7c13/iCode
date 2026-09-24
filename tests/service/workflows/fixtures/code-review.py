# Code Review — the golden Workflow Mode SDK example (fully annotated; teaching is the contract).
# Standard library only, default interpreter mode.
from chrys.workflows import (
    BuilderScope,
    NodeContext,
    NodeHandle,
    Retry,
    SourceValue,
    Workflow,
    WorkflowBuilder,
    WorkflowValue,
)

wf = WorkflowBuilder(
    "代码评审",
    description="并行三维评审 → 汇总裁决，不过关自动打回重评（最多 3 轮），耗尽转整改建议",
)


def prepare_context(value: WorkflowValue) -> str:
    """Entry (fn(value) ABI): turn the InputBar text into a review task."""
    text = value.text
    target = text.strip() or "HEAD"
    return f"评审对象: {target}\n请基于当前工作区的 git diff 完成你负责的维度评审。"


def dispatch(value: WorkflowValue) -> str:
    """Loop-body entry: forward unchanged (shared upstream of the fan-out)."""
    text = value.text
    return text


def merge_reviews(sources: list[SourceValue]) -> WorkflowValue:
    """join combine (pure ABI): three reviews become one document; skipped sources are absent."""
    parts = [f"## {s.node_id}\n{s.value.text}" for s in sources]
    return WorkflowValue(
        text="\n\n".join(parts),
        data={"review_count": len(sources)},
    )


def summarize(value: WorkflowValue, ctx: NodeContext) -> str:
    """join target (fn(value, ctx) ABI): data stops at the agent boundary, so consume it here."""
    data = value.data
    count = data.get("review_count", 0) if isinstance(data, dict) else 0
    ctx.emit(f"已合并 {count} 份评审，交给裁决")
    return f"（本轮共 {count} 维评审）\n\n{value.text}"


def format_report(value: WorkflowValue) -> str:
    """Final formatting (fn(value) ABI) once the verdict passed."""
    text = value.text
    return f"# 代码评审报告（裁决：通过）\n\n{text}"


def passed(value: WorkflowValue) -> bool:
    """Predicate ABI shared by the loop until and the switch case."""
    return "VERDICT: PASS" in value.text


def review_round(scope: BuilderScope) -> tuple[NodeHandle, NodeHandle]:
    entry: NodeHandle = scope.python("分发", dispatch)
    correctness: NodeHandle = scope.agent("正确性评审", profile="QA")
    security: NodeHandle = scope.agent(
        "安全评审",
        profile="QA",
        model="glm-4.7",
        retry=Retry(max_attempts=4, backoff=10),
    )
    perf: NodeHandle = scope.agent("性能评审", profile="QA", model="deepseek-r1", timeout=600)
    summary: NodeHandle = scope.python("汇总", summarize)
    verdict: NodeHandle = scope.agent(
        "裁决",
        profile="QA",
        instructions_suffix="最后一行只输出 VERDICT: PASS 或 VERDICT: FAIL，并列出未过关项。",
    )
    scope.edge(entry, correctness)
    scope.edge(entry, security)
    scope.edge(entry, perf)
    scope.join([correctness, security, perf], summary, combine=merge_reviews)
    scope.chain(summary, verdict)
    return entry, verdict


prepare: NodeHandle = wf.python("准备上下文", prepare_context)

review: NodeHandle = wf.loop(
    "评审轮",
    body=review_round,
    until=passed,
    max_iterations=3,
    on_exhausted="continue",
)

report: NodeHandle = wf.python("排版报告", format_report)
advice: NodeHandle = wf.agent(
    "整改建议",
    profile="QA",
    instructions_suffix="评审未通过。汇总全部未过关项，输出可执行的整改清单。",
)

wf.start(prepare)
wf.chain(prepare, review)
wf.switch(
    review,
    cases=[(passed, report)],
    default=advice,
)
wf.output(report)
wf.output(advice)

workflow: Workflow = wf.build()

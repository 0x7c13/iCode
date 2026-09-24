# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Research synthesis: two independent research perspectives, collected into one Python report.

The input is the research question. Evidence and counterarguments are researched
in parallel. The join preserves their declared order and names on the data
channel; a Python node formats both contributions into a synthesis report.
"""

from __future__ import annotations

from chrys.workflows import NodeContext, SourceValue, WorkflowBuilder, WorkflowValue


def question(value: WorkflowValue) -> str:
    text = value.text
    return "Research this question, citing sources and distinguishing evidence from inference:\n" + text


def collect(sources: list[SourceValue]) -> WorkflowValue:
    return WorkflowValue(
        text="Research collected",
        data=[{"perspective": source.node_id, "text": source.value.text} for source in sources],
    )


def synthesize(value: WorkflowValue, ctx: NodeContext) -> WorkflowValue:
    """Keep both perspectives visible instead of letting one overwrite the other."""
    contributions = value.data if isinstance(value.data, list) else []
    ctx.emit(f"Synthesizing {len(contributions)} research perspectives.")
    sections = [f"## {item['perspective']}\n{item['text']}" for item in contributions]
    return WorkflowValue(text="# Research synthesis\n\n" + "\n\n".join(sections), data=value.data)


wf = WorkflowBuilder("Research synthesis", description="Parallel evidence and counterarguments, synthesized in Python")
prepare = wf.python("question", question)
evidence = wf.agent(
    "evidence", profile="General", instructions_suffix="Find supporting evidence and cite your sources."
)
counterarguments = wf.agent(
    "counterarguments",
    profile="General",
    instructions_suffix="Research counterarguments, uncertainties and missing evidence.",
)
report = wf.python("synthesis", synthesize)
wf.start(prepare)
wf.edge(prepare, evidence)
wf.edge(prepare, counterarguments)
wf.join([evidence, counterarguments], report, combine=collect)
wf.output(report)
workflow = wf.build()

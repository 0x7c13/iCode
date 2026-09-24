# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Release readiness: assess the release, then choose a release report or a remediation plan.

The input names the release target. A QA agent checks tests, compatibility and
release documentation. Exactly a final line of ``READY: YES`` selects the
Python release report; every other verdict selects a second QA agent's remediation
path. Only the selected branch contributes an output. Nothing is published.
"""

from __future__ import annotations

from chrys.workflows import WorkflowBuilder, WorkflowValue


def ready(value: WorkflowValue) -> bool:
    return bool(value.text.strip()) and value.text.strip().splitlines()[-1] == "READY: YES"


def release_report(value: WorkflowValue) -> str:
    text = value.text
    return "# Release readiness\n\n" + text


wf = WorkflowBuilder("Release readiness", description="QA assessment switches to a ready report or a remediation plan")
assessment = wf.agent(
    "assessment",
    profile="QA",
    instructions_suffix=(
        "Assess the named release target's tests, compatibility and release documentation. "
        "Cite blockers. Finish with exactly READY: YES or READY: NO on its own line."
    ),
)
report = wf.python("release_report", release_report)
remediation = wf.agent(
    "remediation",
    profile="QA",
    instructions_suffix="Turn this release assessment into an ordered remediation plan.",
)
wf.start(assessment)
wf.switch(assessment, cases=[(ready, report)], default=remediation)
wf.output(report)
wf.output(remediation)
workflow = wf.build()

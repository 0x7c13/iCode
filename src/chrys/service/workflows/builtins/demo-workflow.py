# Demo workflow: a runnable example of every workflow feature, meant to be copied and adapted.

# /// script
# requires-python = ">=3.9"
# ///

r"""Workflow Demo · Project Tour: every workflow feature in one read-only, runnable example.

WHAT IT DOES
    It writes a guided tour of the project in the current workspace for someone new to it. Python
    nodes take a bounded inventory of the tree, agent nodes read the code, and you review the draft
    before it becomes the output. Copy this file to `.chrys/workflows/` and edit it to start your own.

HOW TO RUN IT
    In the TUI, open Workflows, pick this one and press Start. The input is optional free text naming
    what you want to learn ("how does authentication work?"). The run asks you two things: how deep to
    go, and whether the draft is good.

    `icode workflow run` is headless: nobody can answer there, so `ctx.ask` fails the node. This demo
    therefore defines its own input lines that settle the questions up front:

        icode workflow run demo-workflow --input $'interactive: false\ndepth: deep\nhow are errors handled?'

        interactive: false      never ask; the tour is labelled "not reviewed by a human"
        depth: quick | deep     skip the depth question (unattended runs default to quick)

    `$'...'` is bash and zsh quoting. In PowerShell, write each line break as `n inside double quotes:

        icode workflow run demo-workflow --input "interactive: false`ndepth: deep`nhow are errors handled?"

    These lines are a convention of this file (see `read_request`), not a workflow feature. The file
    deliberately does not catch the failure of `ctx.ask` to fall back on a default answer: that would
    turn every real fault into "carry on as if the user had agreed".

READ-ONLY SCOPE
    Nothing in the project is modified. The Python nodes only list directories and read the head of a
    README; they never import or run project code, run tests, install anything or write a file. The
    walk skips VCS, dependency, cache and virtual-environment directories, never enters a symlinked
    directory, and is bounded in depth and entry count. iCode itself still records the run (session,
    transcripts, outputs) as it does for every workflow.

    Every agent uses the builtin QA profile, which is read-only by its instructions and has no
    file-writing tools, and each node repeats the rule in `instructions_suffix`. Be precise about what
    that guarantees: QA still carries the `shell` tool (behind an approval the user cannot switch off),
    so the read-only promise rests on instructions and approval, not on the absence of a tool. A
    workflow that needs a tool-level guarantee has to name a profile without `shell`.

THE GRAPH
    read_request -> scan_tree -> read_key_files --[must_ask_depth]--> choose_depth (ask) --+
                                       |                                                   v
                                       +-----------[depth_is_known]--------------------> plan

    plan --switch--+-- wants_deep --> fan_out --+--> architecture --+
                   |                            +--> entry_points --+--join--> merge_findings --+
                   |                            +--> conventions ---+                           |
                   +-- default ----> hand_over ----> quick_scan --------------------------------+
    plan ---------------- side edge: carries the run state past the agents --------------------+
                                                                                                |
                                                                          join(combine=carry_state)
                                                                                                v
    refine (loop, at most MAX_ROUNDS):   open_round --> brief --> write_tour --+
                                              |                                +--join(combine=keep_round_state)
                                              +---------- side edge -----------+          |
                                                                                          v
                                                                                     review (ask)
                                                                                                |
                                   render_tour (output) <---------------------------------------+
                  next_steps (output) <---- tour_text <--------------[wants_deep]---------------+

WHERE EACH FEATURE IS USED
    WorkflowBuilder, build(), Workflow ...... bottom of the file; the module-level `workflow` is what iCode loads
    wf.python, sync  fn(value) ............... read_request, read_key_files, merge_findings, brief, tour_text,
                                              render_tour
    wf.python, async fn(value) ............... scan_tree
    wf.python, sync  fn(value, ctx) .......... plan, fan_out, hand_over, open_round
    wf.python, async fn(value, ctx) .......... choose_depth, review
    returning a WorkflowValue(text, data) .... read_request, plan, open_round, review
    returning a plain str ................... fan_out, hand_over, merge_findings, brief, tour_text, render_tour
    ctx.emit ................................ plan, fan_out, hand_over, choose_depth, open_round, review
    await ctx.ask(Question), Answer ......... choose_depth (one choice between two Options)
    await ctx.ask(str) ...................... review (free text)
    wf.agent, instructions_suffix ........... all six agents: the three readers, quick_scan, write_tour, next_steps
    timeout= ................................ conventions (agent), choose_depth and review (None)
    Retry on a Python node .................. read_key_files
    Retry on an agent node .................. write_tour
    model= (commented example) .............. architecture
    wf.chain ................................ read_request -> scan_tree -> read_key_files, and inside the loop
    wf.edge, plain .......................... choose_depth -> plan, the fan-out, refine -> render_tour
    wf.edge(when=...) ....................... the bypass around choose_depth; refine -> tour_text
    wf.switch(cases, default) ............... plan -> fan_out | hand_over
    wf.join, default combine ................ the three readers into merge_findings
    wf.join(combine=...), SourceValue ....... carry_state (into the loop), keep_round_state (inside it)
    wf.loop, BuilderScope, NodeHandle ....... refine / refine_round
    until=, max_iterations=, on_exhausted= .. refine
    wf.start, wf.output (more than one) ..... bottom of the file
    script metadata block ................... top of the file (`requires-python`)

RULES WORTH REMEMBERING
    * A node body is `fn(value)` or `fn(value, ctx)`, sync or async, and returns a WorkflowValue or a
      str. A sync body runs on a worker thread, so blocking calls are fine in it. An async body runs on
      the worker's event loop, so it must not block (see `scan_tree`), and only an async body can
      `await ctx.ask`.
    * Edge predicates, switch cases, a loop's `until` and a join's `combine` must be plain sync
      functions of one argument. There is no async form of any of them; `build()` rejects one.
    * `value.text` reaches everyone. `value.data` (JSON only) reaches Python nodes and predicates, and
      is lost in two places: an agent reads and writes text only, and the default join concatenates
      text only. State that must outlive either travels on a side edge into a `join(combine=...)`,
      which is what `carry_state` and `keep_round_state` do.
    * Handing `data` to an agent is not an error, but the run raises a notice about it, because it is
      usually a mistake. So the node in front of an agent returns a plain str, the prompt and nothing
      else, and the state leaves on the side edge from the node before that one. This file never
      triggers the notice.
    * A node runs once all its in-edges are settled. Sources that were skipped, or whose condition was
      false, are simply absent. With one source present the value passes through unchanged, data
      included; with none present the node is skipped, and so is everything that depends only on it.
    * A timeout is wall-clock per attempt and includes the time spent waiting for a person, so a node
      that asks needs `timeout=None`. Python nodes default to 300 seconds, agents to no deadline.
    * An agent node retries up to 3 attempts by default, a Python node runs once. A retry runs the
      whole body again, so give `Retry` only to a body that is safe to repeat.

ENVIRONMENT
    The block at the top of this file is standard inline script metadata. `requires-python` is checked
    against the interpreter that runs the file. iCode does not install `dependencies`; a workflow that
    needs third-party packages points `[tool.chrys] python` at an interpreter or virtual environment
    of its own, relative to the workflow file. A workspace copy could start like this (indented here
    so that it stays an example):

        # /// script
        # requires-python = ">=3.11"
        # dependencies = ["pyyaml"]
        #
        # [tool.chrys]
        # python = "../../.venv"
        # ///
"""

import asyncio
import os
from typing import Any

from chrys.workflows import (
    BuilderScope,
    NodeContext,
    NodeHandle,
    Option,
    Question,
    Retry,
    SourceValue,
    Workflow,
    WorkflowBuilder,
    WorkflowValue,
)

wf = WorkflowBuilder(
    "Workflow Demo · Project Tour",
    description="A read-only guided tour of this project: quick or deep, reviewed by you for up to 2 rounds",
)

# ---------------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------------

QUICK = "quick"
DEEP = "deep"
MAX_ROUNDS = 2  # The draft is written once and revised at most once.

# Node names that code refers to by id: a custom combine finds its sources by `SourceValue.node_id`.
PLAN_NODE = "plan"
ROUND_NODE = "open_round"
WRITER_NODE = "write_tour"

# Bounds of the inventory. Depth and entry count bound the walk whatever the file system looks like.
MAX_DEPTH = 3
MAX_ENTRIES = 250
README_BYTES = 6000
README_LINES = 60
SKIPPED_DIRECTORIES = frozenset(
    {
        ".git", ".hg", ".svn",
        "node_modules", "bower_components", "vendor", "site-packages",
        ".venv", "venv", "env",
        "__pycache__", ".tox", ".nox", ".cache", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".gradle",
        ".idea", ".vscode",
        "build", "dist", "target",
    }
)  # fmt: skip
README_NAMES = ("readme.md", "readme.rst", "readme.txt", "readme")
PROJECT_FILES = (
    "pyproject.toml", "setup.py", "requirements.txt", "package.json", "tsconfig.json", "Cargo.toml", "go.mod",
    "pom.xml", "build.gradle", "CMakeLists.txt", "Makefile", "Dockerfile",
)  # fmt: skip

TRUE_WORDS = ("true", "yes", "on", "1")
FALSE_WORDS = ("false", "no", "off", "0")
ACCEPT_WORDS = ("ok", "okay", "yes", "y", "accept", "accepted", "approve", "approved", "lgtm", "looks good")

DEFAULT_FOCUS = "a general orientation for someone who is new to this project"

# The state every Python node and predicate can rely on. It rides on `WorkflowValue.data`, so it has to
# be plain JSON: dicts with string keys, lists, strings, numbers, booleans and None.
DEFAULT_STATE = {
    "focus": DEFAULT_FOCUS,
    "interactive": True,  # False: an unattended run that must never call ctx.ask
    "depth": None,  # QUICK, DEEP, or None while it is still undecided
    "round": 0,  # refine rounds started so far
    "reviewed": False,  # a person looked at the draft
    "accepted": False,  # that person accepted it
    "feedback": "",  # what they asked to change, when they did not
}

READ_ONLY_RULES = (
    " This is a read-only tour: only read and search the workspace. Never create, edit, move or delete"
    " anything, never run a command that changes something, and do not run tests, builds or installers."
    " Cite files as path:line."
)

READER_TASK = (
    "Help a newcomer understand the project in the current workspace.\n"
    "What they want to learn: {focus}\n\n"
    "A bounded inventory of the project, to get you started (read further yourself):\n\n{inventory}"
)
FAN_OUT_NOTE = (
    "\n\nYou are one of three readers working in parallel. Cover only the dimension named in your"
    " instructions; the other two are covered by someone else."
)
WRITE_TASK = (
    "Write a guided tour of this project for a newcomer.\n"
    "What they want to learn: {focus}\n\n"
    "Notes from reading the project:\n\n{findings}"
)
REVISE_TASK = (
    "Revise the project tour below. The reader reviewed it and asked for this change:\n\n{feedback}\n\n"
    "Return the complete revised tour, not a list of edits.\n\n--- current tour ---\n{draft}"
)
# A question is shown as Markdown in a dialog that stays open until it is answered, so the person
# cannot look anything up in the run meanwhile: put what they need to decide into the question and
# its option descriptions.
DEPTH_QUESTION = Question(
    "How deep should the tour go?",
    options=[
        Option(QUICK, "One reader skims the project (the default)"),
        Option(
            DEEP,
            "Three readers study architecture, entry points and conventions in parallel, and the run adds"
            " a list of next steps",
        ),
    ],
)
REVIEW_QUESTION = (
    "{draft}\n\n---\n\n"
    "**Review, round {round} of {rounds}.** Reply `ok` to accept this tour, or describe what to change."
)


# ---------------------------------------------------------------------------------------------------
# Helpers (plain functions; only the functions handed to the builder below become nodes)
# ---------------------------------------------------------------------------------------------------


def _state(value: WorkflowValue) -> dict[str, Any]:
    """The run state a value carries on its data channel, with defaults for whatever is missing."""
    state: dict[str, Any] = dict(DEFAULT_STATE)
    if isinstance(value.data, dict):
        state.update(value.data)
    return state


def _list_directory(directory: str, depth: int, lines: list[str], counts: dict[str, int]) -> None:
    """Append an indented listing of *directory* to *lines*; stops at MAX_DEPTH and MAX_ENTRIES."""
    try:
        with os.scandir(directory) as scanner:
            entries = sorted(scanner, key=lambda entry: entry.name)
    except OSError:
        return  # An unreadable directory is left out; it is no reason to fail the tour.
    for entry in entries:
        if len(lines) >= MAX_ENTRIES:
            return
        indent = "  " * depth
        # follow_symlinks=False: a link to a directory is listed like a file and never entered.
        if entry.is_dir(follow_symlinks=False):
            is_virtualenv = os.path.isfile(os.path.join(entry.path, "pyvenv.cfg"))
            if entry.name in SKIPPED_DIRECTORIES or is_virtualenv:
                continue
            lines.append(f"{indent}{entry.name}/")
            if depth + 1 < MAX_DEPTH:
                _list_directory(entry.path, depth + 1, lines, counts)
        else:
            lines.append(f"{indent}{entry.name}")
            suffix = os.path.splitext(entry.name)[1].lower() or "(no extension)"
            counts[suffix] = counts.get(suffix, 0) + 1


def _inventory(root: str) -> str:
    """The tree listing and a file-type tally of *root*, as text."""
    lines: list[str] = []
    counts: dict[str, int] = {}
    _list_directory(root, 0, lines, counts)
    most_common = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8]
    tally = ", ".join(f"{suffix} {count}" for suffix, count in most_common) or "no files"
    limit = f"; stopped at {MAX_ENTRIES} entries" if len(lines) >= MAX_ENTRIES else ""
    return (
        f"Project: {os.path.basename(root) or root}\n"
        f"File types: {tally}\n\n"
        f"Tree ({MAX_DEPTH} levels; VCS, dependency, cache and virtual-environment directories skipped{limit}):\n"
        + "\n".join(lines)
    )


# ---------------------------------------------------------------------------------------------------
# Python node bodies, in graph order
# ---------------------------------------------------------------------------------------------------


def read_request(value: WorkflowValue) -> WorkflowValue:
    """Start node. SYNC `fn(value)`, returning a WorkflowValue with `data`.

    The run's input text arrives as `value.text` (its `data` is None). This body separates the two
    option lines described in the module docstring from the free text, and from here on the options
    travel as structured state on `data` while `text` stays readable prose.
    """
    lines = value.text.splitlines()
    if len(lines) == 1 and "\\n" in lines[0]:
        # The bash command pasted into PowerShell arrives as one line with literal `\n`s, after the `$` of
        # `$'...'`. Read as written, its options would be ignored without a word.
        first_key = lines[0].partition(":")[0].strip().lstrip("$").strip().lower()
        if first_key in ("interactive", "depth"):
            raise ValueError(
                "The input is one line with a literal \\n in it: $'...' is bash quoting, and PowerShell passes "
                "it on as written. In PowerShell, write each line break as `n inside double quotes: --input "
                '"interactive: false`ndepth: deep`nhow are errors handled?"'
            )
    options: dict[str, str] = {}
    focus_lines: list[str] = []
    for line in lines:
        key, separator, raw = line.partition(":")
        name = key.strip().lower()
        if separator and name in ("interactive", "depth"):
            options[name] = raw.strip().lower()
        else:
            focus_lines.append(line)

    # A value this demo cannot interpret is an error, not a reason to guess: raising fails the node,
    # and the message is what the user sees.
    switch = options.get("interactive", "true")
    if switch not in TRUE_WORDS + FALSE_WORDS:
        raise ValueError(f"'interactive: {switch}' is not understood; write true or false.")
    depth = options.get("depth") or None
    if depth not in (None, QUICK, DEEP):
        raise ValueError(f"'depth: {depth}' is not understood; write {QUICK} or {DEEP}.")

    interactive = switch in TRUE_WORDS
    if not interactive and depth is None:
        depth = QUICK  # Nobody to ask, nothing requested: the cheaper tour.
    focus = "\n".join(focus_lines).strip() or DEFAULT_FOCUS
    return WorkflowValue(text=focus, data=dict(DEFAULT_STATE, focus=focus, interactive=interactive, depth=depth))


async def scan_tree(value: WorkflowValue) -> WorkflowValue:
    """ASYNC `fn(value)`.

    An async body runs on the worker's own event loop, next to the machinery that talks to iCode, so
    blocking in it stalls everything else in the run. The directory walk blocks, hence `to_thread`.
    `read_key_files` below does the same kind of file access as a SYNC body and needs no such care:
    the worker already runs sync bodies on a thread. Prefer sync unless the body awaits something.

    The worker's current directory is the run's workspace, so `os.getcwd()` is the project root.
    """
    inventory = await asyncio.to_thread(_inventory, os.getcwd())
    # `data` is passed along untouched. Returning only a str here would silently drop it.
    return WorkflowValue(text=inventory, data=value.data)


def read_key_files(value: WorkflowValue) -> WorkflowValue:
    """SYNC `fn(value)` with a Retry policy (see where the node is declared).

    Adds the project files found at the top level and the head of the README. A file can change or
    vanish between the listing and the read; the OSError is not caught here, so it fails the attempt
    and the node's Retry runs the body once more. That is only sound because this body is safe to
    repeat: it reads, and nothing else.
    """
    names = {name.lower(): name for name in os.listdir(".")}
    present = [name for name in PROJECT_FILES if name.lower() in names]
    sections = [value.text, "Project files at the top level: " + (", ".join(present) or "none of the usual ones")]
    for candidate in README_NAMES:
        readme = names.get(candidate)
        if readme is None or not os.path.isfile(readme):
            continue
        with open(readme, "rb") as handle:
            head = handle.read(README_BYTES).decode("utf-8", errors="replace")
        excerpt = "\n".join(head.splitlines()[:README_LINES])
        sections.append(f"{readme} (first {README_LINES} lines at most):\n{excerpt}")
        break
    return WorkflowValue(text="\n\n".join(sections), data=value.data)


def must_ask_depth(value: WorkflowValue) -> bool:
    """Edge predicate. Predicates are SYNC, take the source's value, and should be cheap and pure."""
    state = _state(value)
    return bool(state["interactive"]) and state["depth"] is None


def depth_is_known(value: WorkflowValue) -> bool:
    """The complement of `must_ask_depth`: exactly one of the two edges out of read_key_files fires."""
    return not must_ask_depth(value)


async def choose_depth(value: WorkflowValue, ctx: NodeContext) -> WorkflowValue:
    """ASYNC `fn(value, ctx)`: the only shape that can `await ctx.ask`.

    `ctx.ask` suspends this body until the person driving the run answers. Asked a `Question`, it
    returns an `Answer`: `answer.choice` is the option label they picked (clicking it and typing it
    exactly are the same answer), or None when they typed something else or nothing; `answer.text`
    holds what they typed besides. A list of up to five Questions shows them in one dialog, one tab each
    labelled by its `header`, and returns a tuple of Answers. The node is declared with `timeout=None`
    because a timeout counts the waiting too.

    In an unattended run this node is never activated: the conditional edges route around it, which
    shows up as "skipped" in the graph. That is one of two ways to keep `ctx.ask` out of a headless
    run; `review` shows the other.
    """
    answer = await ctx.ask(DEPTH_QUESTION)
    depth = DEEP if answer.choice == DEEP else QUICK  # Anything that is not "deep" is the default.
    ctx.emit(f"Depth chosen: {depth}.")
    return WorkflowValue(text=value.text, data=dict(_state(value), depth=depth))


def plan(value: WorkflowValue, ctx: NodeContext) -> WorkflowValue:
    """SYNC `fn(value, ctx)`. `ctx.emit` works in every body; only `ctx.ask` needs async.

    This node has two in-edges, from choose_depth and from the bypass, and exactly one of them carries
    a value in any run. A single present source passes through unchanged, `data` included, so the body
    does not care which way the value came.

    Its value goes two ways: the switch judges it and hands it to one branch, and a side edge takes
    the same value, state and all, straight to the join in front of the loop. Emitted lines appear
    live under the node and are kept with the run.
    """
    state = _state(value)
    depth = state["depth"] or QUICK
    readers = "three readers in parallel" if depth == DEEP else "one reader"
    asked = "asked interactively" if state["interactive"] else "unattended run"
    ctx.emit(f"Tour depth: {depth} ({readers}; {asked}).")
    return WorkflowValue(text=value.text, data=dict(state, depth=depth))


def wants_deep(value: WorkflowValue) -> bool:
    """Used twice: as the switch case after `plan`, and as the condition on the edge out of the loop.

    It reads `data`, so it only works where the state has been carried along. Both call sites are
    downstream of Python nodes or a custom combine, never directly downstream of an agent.
    """
    return _state(value)["depth"] == DEEP


def _reader_task(value: WorkflowValue) -> str:
    return READER_TASK.format(focus=_state(value)["focus"], inventory=value.text)


def fan_out(value: WorkflowValue, ctx: NodeContext) -> str:
    """SYNC `fn(value, ctx)` returning a plain str, which is shorthand for `WorkflowValue(text=...)`.

    The shorthand carries no `data`, and that is the point: this is the node in front of agents, so
    it returns their prompt and nothing else. An agent's prompt is the `text` of its input. The state
    is not lost, it left `plan` on the side edge.

    A switch picks ONE target, so fanning out to three agents needs this node in between: the switch
    routes to it, and three plain edges leave it.
    """
    ctx.emit("Deep tour: three readers start in parallel.")
    return _reader_task(value) + FAN_OUT_NOTE


def hand_over(value: WorkflowValue, ctx: NodeContext) -> str:
    """The quick branch of the switch: the same prompt for a single reader, again as a plain str."""
    ctx.emit("Quick tour: one reader skims the project.")
    return _reader_task(value)


def merge_findings(value: WorkflowValue) -> str:
    """Target of the DEFAULT join over the three readers.

    Without `combine=`, a join concatenates its present sources in declared order as
    "## <node id>" sections of text, and the result has no `data`. A reader that failed its way to
    "skipped" would simply be missing from the text.
    """
    return "Notes from three readers, one section each.\n\n" + value.text


def carry_state(sources: list[SourceValue]) -> WorkflowValue:
    """CUSTOM combine of the join in front of the loop. SYNC, one argument, at most 30 seconds.

    `sources` holds only the sources that produced a value, in declared order. Each is a SourceValue:
    `node_id`, `activation_id` and `value`. Here that is always `plan` (the side edge) plus whichever
    branch of the switch ran; the other branch was skipped and is absent.

    This is the reunion the side edge exists for: the findings are text that came through agents, and
    the state comes from `plan`, where it last existed.
    """
    by_node = {source.node_id: source.value for source in sources}
    findings = [source.value.text for source in sources if source.node_id != PLAN_NODE]
    return WorkflowValue(text="\n\n".join(findings), data=_state(by_node[PLAN_NODE]))


def open_round(value: WorkflowValue, ctx: NodeContext) -> WorkflowValue:
    """Loop-body ENTRY. Round 1 receives the loop's input; every later round the previous EXIT value.

    So in round 1 `value.text` is the findings, and in round 2 it is the draft that `review` returned,
    with the reviewer's feedback on `data`. The round counter lives on `data` as well: a loop keeps no
    variables between rounds other than the value it hands from exit to entry.
    """
    state = _state(value)
    round_number = state["round"] + 1
    ctx.emit("Round 1: writing the tour." if round_number == 1 else f"Round {round_number}: revising the tour.")
    return WorkflowValue(text=value.text, data=dict(state, round=round_number))


def brief(value: WorkflowValue) -> str:
    """The node in front of the writer: it turns the round's value into a prompt, a plain str.

    The same split as `plan` and `fan_out` above. `open_round` owns the state and sends it around the
    agent; this node only writes the prompt.
    """
    state = _state(value)
    if state["round"] == 1:
        return WRITE_TASK.format(focus=state["focus"], findings=value.text)
    return REVISE_TASK.format(feedback=state["feedback"], draft=value.text)


def keep_round_state(sources: list[SourceValue]) -> WorkflowValue:
    """CUSTOM combine inside the loop: the writer's text, with the state `open_round` sent around it."""
    by_node = {source.node_id: source.value for source in sources}
    return WorkflowValue(text=by_node[WRITER_NODE].text, data=_state(by_node[ROUND_NODE]))


async def review(value: WorkflowValue, ctx: NodeContext) -> WorkflowValue:
    """Loop-body EXIT, and the second `ctx.ask`. Its return value is what `until` judges.

    The other way to keep `ctx.ask` out of an unattended run: branch inside the body. Routing around
    the node, as choose_depth does, is not an option for a loop's exit, which must produce a value in
    every round (a round without one fails the run).

    An unattended run is recorded as NOT reviewed. It does not pretend that somebody approved.
    """
    state = _state(value)
    if not state["interactive"]:
        ctx.emit("Unattended run: nobody reviews the draft.")
        return WorkflowValue(text=value.text, data=dict(state, reviewed=False, accepted=False))

    question = REVIEW_QUESTION.format(draft=value.text, round=state["round"], rounds=MAX_ROUNDS)
    answer = (await ctx.ask(question)).strip()
    accepted = answer.lower().rstrip(".!") in ACCEPT_WORDS
    ctx.emit("Accepted." if accepted else "Changes requested.")
    # Only an explicit word of acceptance counts; an empty answer is not one.
    feedback = "" if accepted else answer or "(no details given)"
    return WorkflowValue(text=value.text, data=dict(state, reviewed=True, accepted=accepted, feedback=feedback))


def tour_settled(value: WorkflowValue) -> bool:
    """The loop's `until`: SYNC, judged on the exit value after every round. True ends the loop.

    An unattended run is settled after one round, because another round has nothing new to go on.
    """
    state = _state(value)
    return bool(state["accepted"]) or not state["interactive"]


def tour_text(value: WorkflowValue) -> str:
    """The node in front of next_steps: the tour as a plain str, the state left behind.

    The condition on the edge into this node reads `data`, so the loop's value has to carry it; the
    agent after this node must not receive it. One line of Python separates the two.
    """
    return value.text


def render_tour(value: WorkflowValue) -> str:
    """Output node. Its value appears on the Output tab and is what `icode workflow run` prints.

    The status line says honestly how the draft got here, including the case where the loop ran out
    of rounds and handed over a draft that nobody accepted.
    """
    state = _state(value)
    if state["accepted"]:
        status = f"accepted by the reader in round {state['round']}"
    elif not state["reviewed"]:
        status = "NOT reviewed by a human (unattended run)"
    else:
        status = f"DRAFT, not accepted after {state['round']} rounds. Still open: {state['feedback']}"
    return f"# Project tour\n\nFocus: {state['focus']}\nDepth: {state['depth']}\nStatus: {status}\n\n{value.text}"


# ---------------------------------------------------------------------------------------------------
# The refine loop
# ---------------------------------------------------------------------------------------------------


def refine_round(scope: BuilderScope) -> tuple[NodeHandle, NodeHandle]:
    """Loop body. Called ONCE, at build time, to declare the nodes of a round; it is not run per round.

    `scope` offers python/agent/edge/switch/join/chain, but no loop (loops do not nest), no start and
    no output. Edges cannot cross the loop boundary: a body node connects only to body nodes, and the
    outside connects to the loop node as a whole. Return the (entry, exit) pair.
    """
    entry: NodeHandle = scope.python(ROUND_NODE, open_round)
    prompt: NodeHandle = scope.python("brief", brief)
    writer: NodeHandle = scope.agent(
        WRITER_NODE,
        profile="QA",
        # Agents already retry up to 3 attempts, and only failures the backend marks as transient.
        # Retry replaces that default: 2 attempts, and a 5 second pause before the second.
        retry=Retry(max_attempts=2, backoff=5),
        instructions_suffix=(
            "You are writing a guided tour of this project for a newcomer. Keep it under 500 words: what the"
            " project is, how it is laid out, where execution starts, and what to read first. When asked to"
            " revise, return the whole tour again." + READ_ONLY_RULES
        ),
    )
    # timeout=None: this node waits for a person, and a timeout would count that wait.
    exit_: NodeHandle = scope.python("review", review, timeout=None)

    scope.chain(entry, prompt, writer)
    # The side edge again: `open_round` feeds the join directly, so the round's state reaches
    # `review` although the prompt and the writer in between pass text only.
    scope.join([entry, writer], exit_, combine=keep_round_state)
    return entry, exit_


# ---------------------------------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------------------------------

request: NodeHandle = wf.python("read_request", read_request)
tree: NodeHandle = wf.python("scan_tree", scan_tree)
# A Python node runs once unless told otherwise. Retry counts the first execution: 2 attempts means
# one retry, after an exception or a timeout, 0.5 seconds later.
key_files: NodeHandle = wf.python("read_key_files", read_key_files, retry=Retry(max_attempts=2, backoff=0.5))
depth_choice: NodeHandle = wf.python("choose_depth", choose_depth, timeout=None)
tour_plan: NodeHandle = wf.python(PLAN_NODE, plan)

dispatch: NodeHandle = wf.python("fan_out", fan_out)
architecture: NodeHandle = wf.agent(
    "architecture",
    # `profile` names an agent profile: a builtin one, or one of your own. An unknown name rejects the
    # workflow when it is opened, before anything runs.
    profile="QA",
    # `model` pins this node to one model profile; without it the node uses the run's active model.
    # A name that does not resolve rejects the workflow just like an unknown profile, which is why
    # this shipped file names none. In your own copy:
    #     model="my-fast-model",
    # `instructions_suffix` is appended to the profile's own instructions, for this node only.
    instructions_suffix=(
        "Your dimension is architecture: the main components, what each is responsible for, and how they"
        " depend on each other." + READ_ONLY_RULES
    ),
)
entry_points: NodeHandle = wf.agent(
    "entry_points",
    profile="QA",
    instructions_suffix=(
        "Your dimension is entry points: where execution starts (commands, mains, servers, handlers) and the"
        " path a typical request or command takes through the code." + READ_ONLY_RULES
    ),
)
conventions: NodeHandle = wf.agent(
    "conventions",
    profile="QA",
    # Agents have no deadline by default. This one gets 10 minutes per attempt.
    timeout=600,
    instructions_suffix=(
        "Your dimension is conventions: how the code is organised and named, how it is tested, and the rules"
        " a contributor is expected to follow." + READ_ONLY_RULES
    ),
)
merged: NodeHandle = wf.python("merge_findings", merge_findings)
solo: NodeHandle = wf.python("hand_over", hand_over)
quick: NodeHandle = wf.agent(
    "quick_scan",
    profile="QA",
    instructions_suffix=(
        "Skim the project and note what a newcomer needs first: its purpose, its layout, where execution"
        " starts, and how it is tested. Be brief." + READ_ONLY_RULES
    ),
)

refine: NodeHandle = wf.loop(
    "refine",
    body=refine_round,
    until=tour_settled,
    max_iterations=MAX_ROUNDS,
    # "continue" (the default): when the rounds run out without `until` turning true, the last exit
    # value becomes the loop's output all the same, and render_tour labels it as a draft.
    # The alternative is on_exhausted="fail": the run then ends as failed (loop_exhausted) and no
    # output is produced. Choose it when an unaccepted result must never look like a result.
    on_exhausted="continue",
)

tour: NodeHandle = wf.python("render_tour", render_tour)
draft_text: NodeHandle = wf.python("tour_text", tour_text)
steps: NodeHandle = wf.agent(
    "next_steps",
    profile="QA",
    instructions_suffix=(
        "The message is a finished tour of this project. Suggest what the newcomer should do next: up to five"
        " files to read, each with a one-line reason, and up to three small, safe first tasks. Only suggest;"
        " do not carry out any of it." + READ_ONLY_RULES
    ),
)

# ---------------------------------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------------------------------

wf.start(request)  # Exactly one start node; it receives the run's input text.

# chain(a, b, c) is edge(a, b) plus edge(b, c).
wf.chain(request, tree, key_files)

# Conditional edges. Each `when` is judged on its own against the source's value, so any number of
# them may fire: none, one or all. These two are complements, which makes them an either/or that
# routes around the asking node when there is nothing to ask.
wf.edge(key_files, depth_choice, when=must_ask_depth)
wf.edge(key_files, tour_plan, when=depth_is_known)
wf.edge(depth_choice, tour_plan)

# A switch is the other either/or: cases are tried in order, the first true one wins, and `default`
# (required) takes the rest. Exactly one target runs; the others are skipped. One switch per node.
wf.switch(tour_plan, cases=[(wants_deep, dispatch)], default=solo)
wf.edge(solo, quick)

# Fan-out is nothing special: several plain edges out of one node, and the targets run in parallel.
wf.edge(dispatch, architecture)
wf.edge(dispatch, entry_points)
wf.edge(dispatch, conventions)

# Fan-in with the default combine. A join synthesises a node of its own, "join:merge_findings", in
# front of the target, and the target may have no other in-edge.
wf.join([architecture, entry_points, conventions], merged)

# Fan-in with a custom combine, straight into the loop. `tour_plan` is the side edge that carries
# the state; of `merged` and `quick` only the branch the switch chose is present.
wf.join([tour_plan, merged, quick], refine, combine=carry_state)

# The loop node stands for its whole body: its value is the last exit value.
wf.edge(refine, tour)
wf.edge(refine, draft_text, when=wants_deep)  # Fires next to the plain edge above, not instead of it.
wf.edge(draft_text, steps)

# Every output a run produced is reported, in this order. An output that was skipped is left out,
# so a quick tour has one output and a deep tour two. Outputs must be top-level nodes.
wf.output(tour)
wf.output(steps)

# iCode loads the module-level name `workflow`. build() validates the whole graph (names, cycles,
# reachability, callable shapes) and freezes the builder; a mistake surfaces here, when the file is
# opened, not in the middle of a run.
workflow: Workflow = wf.build()

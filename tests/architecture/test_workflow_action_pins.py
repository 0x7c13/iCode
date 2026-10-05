# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Supply-chain guards for GitHub Action pins and workflow token permissions.

A mutable ref (``@v6``, ``@stable``) hands whoever can move that ref arbitrary
code execution inside our workflows, two of which hold ``contents: write``.
Pinning is only durable if a newly added step cannot quietly reintroduce one.

The parsed document is the authority on which ``uses:`` keys exist — a line
regex alone silently skips anything it fails to anticipate, and a guard that
cannot see a violation is worse than no guard. The line scan only supplies
line numbers and trailing comments, which YAML discards, and the two must
agree or the guard fails closed.

Token permissions follow the same rule: every workflow must declare its
baseline, and every write grant must appear in an explicit reviewed allowlist.
A newly added workflow or job therefore cannot inherit or acquire write scope
without changing this guard in the same review.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.support.paths import REPO_ROOT

WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

# A `uses:` value is a local path (`./…`), a container image (`docker://…`), or
# `owner/repo[/subpath]@ref`. Only the local path is exempt: it lives in this
# repo and is reviewed with the change. An image tag moves under you exactly
# like a git tag, so a container reference is pinned by digest instead.
_LOCAL_PREFIX = "./"
_DOCKER_PREFIX = "docker://"
_SHA_PINNED = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")
_DIGEST_PINNED = re.compile(r"^docker://\S+@sha256:[0-9a-f]{64}$")
_VERSION_COMMENT = re.compile(r"^v?\d[\w.+-]*$")
# Deliberately permissive: it must match every `uses:` line the parser can see,
# including `uses :`, quoted values, and multi-word trailing comments.
_USES_LINE = re.compile(r"^\s*(?:-\s+)?uses\s*:\s*(.+?)\s*$")

_WORKFLOW_WRITE_GRANTS = frozenset(
    {
        ("tag-release.yml", "actions"),
        ("tag-release.yml", "contents"),
    }
)
_JOB_WRITE_GRANTS = frozenset(
    {
        ("cd.yml", "release", "contents"),
        ("pillow-manylinux2014.yml", "publish", "contents"),
    }
)


def workflow_files(directory: Path) -> list[Path]:
    """GitHub honours both extensions; scanning one of them is a silent hole."""

    return sorted([*directory.glob("*.yml"), *directory.glob("*.yaml")])


def parsed_uses(text: str) -> list[str]:
    """Every `uses:` value anywhere in the document — the authoritative set."""

    found: list[str] = []
    pending: list[Any] = [yaml.safe_load(text)]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "uses" and isinstance(value, str):
                    found.append(value)
                pending.append(value)
        elif isinstance(node, list):
            pending.extend(node)
    return found


def scanned_uses(text: str) -> list[tuple[int, str, str]]:
    """`(line number, ref, trailing comment)` per `uses:` line."""

    scanned: list[tuple[int, str, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        match = _USES_LINE.match(line)
        if match is None:
            continue
        ref, _, comment = match.group(1).partition("#")
        scanned.append((number, ref.strip().strip("\"'"), comment.strip()))
    return scanned


def unpinned(path: Path) -> list[str]:
    """Return one message per `uses:` that is not SHA-pinned and annotated."""

    text = path.read_text(encoding="utf-8")
    scanned = scanned_uses(text)
    problems: list[str] = []

    # Fail closed: anything the parser sees and the line scan misses would
    # otherwise be checked by nobody at all.
    parsed = sorted(parsed_uses(text))
    if sorted(ref for _, ref, _ in scanned) != parsed:
        problems.append(
            f"{path.name}: the line scan cannot classify every `uses:` key "
            f"(scanned {sorted(ref for _, ref, _ in scanned)}, parsed {parsed}) — fix the matcher"
        )

    for number, ref, comment in scanned:
        if ref.startswith(_LOCAL_PREFIX):
            continue
        where = f"{path.name}:{number}"
        if ref.startswith(_DOCKER_PREFIX):
            if not _DIGEST_PINNED.match(ref):
                problems.append(f"{where}: {ref} is not pinned to an image digest (`@sha256:<64 hex>`)")
                continue
        elif not _SHA_PINNED.match(ref):
            problems.append(f"{where}: {ref} is not pinned to a full 40-character commit SHA")
            continue
        if not _VERSION_COMMENT.match(comment):
            problems.append(f"{where}: {ref} needs a trailing `# <version>` comment, found {comment or '(none)'!r}")
    return problems


def test_every_third_party_action_is_sha_pinned() -> None:
    workflows = workflow_files(WORKFLOW_DIR)
    # Fail closed: a renamed directory or extension must not pass vacuously.
    assert workflows, f"no workflows found under {WORKFLOW_DIR}"
    assert sum(len(scanned_uses(path.read_text(encoding="utf-8"))) for path in workflows) > 0, (
        "no `uses:` steps found — is the matcher stale?"
    )

    problems = [problem for path in workflows for problem in unpinned(path)]
    assert not problems, "unpinned GitHub Actions:\n" + "\n".join(problems)


_PINNED = "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803"


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        ("      - uses: actions/checkout@v6", "not pinned"),
        ("      - uses: dtolnay/rust-toolchain@stable", "not pinned"),
        (f"        uses: {_PINNED[:20]} # v6.1.0", "not pinned"),
        # A chatty comment must not buy a floating ref its way past the scan.
        ("      - uses: owner/action@main # latest approved action", "not pinned"),
        ("      - uses : owner/action@main", "not pinned"),
        ('      - uses: "owner/action@main"', "not pinned"),
        (f"      - uses: {_PINNED}", "needs a trailing"),
        (f"      - uses: {_PINNED} # pin me later", "needs a trailing"),
        # Flow style is legal YAML the line scan cannot see — fail closed.
        (f"      - {{uses: {_PINNED}}}", "cannot classify"),
        (f"      - uses: {_PINNED} # v6.1.0", None),
        (f'      - uses: "{_PINNED}" # v6.1.0', None),
        (f"      - uses : {_PINNED} # v6.1.0", None),
        ("      - uses: ./.github/workflows/ci.yml", None),
        # An image tag is mutable too — digest or nothing.
        ("      - uses: docker://alpine:3.20", "not pinned to an image digest"),
        (f"      - uses: docker://alpine@sha256:{'a' * 64}", "needs a trailing"),
        (f"      - uses: docker://alpine@sha256:{'a' * 64} # 3.20", None),
        (f"      - uses: docker://ghcr.io/owner/img:3.20@sha256:{'b' * 64} # 3.20", None),
    ],
)
def test_guard_detects_the_violations_it_claims_to(tmp_path: Path, step: str, expected: str | None) -> None:
    workflow = tmp_path / "probe.yml"
    workflow.write_text(f"jobs:\n  build:\n    steps:\n{step}\n", encoding="utf-8")

    problems = unpinned(workflow)

    if expected is None:
        assert problems == []
    else:
        assert len(problems) == 1 and expected in problems[0]


def test_guard_scans_both_workflow_extensions(tmp_path: Path) -> None:
    for name in ("a.yml", "b.yaml"):
        (tmp_path / name).write_text("jobs:\n  b:\n    steps:\n      - uses: owner/evil@main\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("uses: owner/evil@main\n", encoding="utf-8")

    assert [path.name for path in workflow_files(tmp_path)] == ["a.yml", "b.yaml"]
    assert len([problem for path in workflow_files(tmp_path) for problem in unpinned(path)]) == 2


def _permission_write_scopes(permissions: object, *, where: str) -> tuple[set[str], list[str]]:
    if not isinstance(permissions, dict):
        return set(), [f"{where}: permissions must be an explicit scope mapping"]
    problems: list[str] = []
    writes: set[str] = set()
    for scope, level in permissions.items():
        if not isinstance(scope, str):
            problems.append(f"{where}: permission scopes must be strings")
            continue
        if level not in {"none", "read", "write"}:
            problems.append(f"{where}: permission level for {scope!r} must be none/read/write")
            continue
        if level == "write":
            writes.add(scope)
    return writes, problems


def permission_problems(
    directory: Path,
    *,
    allowed_workflow_writes: frozenset[tuple[str, str]] = _WORKFLOW_WRITE_GRANTS,
    allowed_job_writes: frozenset[tuple[str, str, str]] = _JOB_WRITE_GRANTS,
) -> list[str]:
    """Return every workflow permission-policy violation under *directory*."""
    workflows = workflow_files(directory)
    if not workflows:
        return [f"no workflows found under {directory}"]

    problems: list[str] = []
    workflow_grants: set[tuple[str, str]] = set()
    job_grants: set[tuple[str, str, str]] = set()
    for path in workflows:
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            problems.append(f"{path.name}: could not parse workflow: {error}")
            continue
        if not isinstance(document, dict):
            problems.append(f"{path.name}: workflow document must be a mapping")
            continue
        if "permissions" not in document:
            problems.append(f"{path.name}: workflow-level permissions must be explicit")
        else:
            writes, scope_problems = _permission_write_scopes(document["permissions"], where=path.name)
            problems.extend(scope_problems)
            workflow_grants.update((path.name, scope) for scope in writes)

        jobs = document.get("jobs")
        if not isinstance(jobs, dict):
            problems.append(f"{path.name}: jobs must be a mapping")
            continue
        for job_name, job in jobs.items():
            if not isinstance(job_name, str) or not isinstance(job, dict):
                problems.append(f"{path.name}: invalid job definition")
                continue
            if "permissions" not in job:
                continue
            writes, scope_problems = _permission_write_scopes(
                job["permissions"],
                where=f"{path.name}:{job_name}",
            )
            problems.extend(scope_problems)
            job_grants.update((path.name, job_name, scope) for scope in writes)

    for grant in sorted(workflow_grants - allowed_workflow_writes):
        problems.append(f"unallowlisted workflow write grant: {grant[0]}:{grant[1]}")
    for grant in sorted(allowed_workflow_writes - workflow_grants):
        problems.append(f"allowlisted workflow write grant is missing: {grant[0]}:{grant[1]}")
    for grant in sorted(job_grants - allowed_job_writes):
        problems.append(f"unallowlisted job write grant: {grant[0]}:{grant[1]}:{grant[2]}")
    for grant in sorted(allowed_job_writes - job_grants):
        problems.append(f"allowlisted job write grant is missing: {grant[0]}:{grant[1]}:{grant[2]}")
    return problems


def test_every_workflow_declares_permissions_and_only_allowlisted_scopes_can_write() -> None:
    assert permission_problems(WORKFLOW_DIR) == []


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        ("jobs:\n  build:\n    runs-on: ubuntu-latest\n", "workflow-level permissions must be explicit"),
        ("permissions: write-all\njobs:\n  build:\n    runs-on: ubuntu-latest\n", "explicit scope mapping"),
        ("permissions: read-all\njobs:\n  build:\n    runs-on: ubuntu-latest\n", "explicit scope mapping"),
        (
            "permissions:\n  contents: admin\njobs:\n  build:\n    runs-on: ubuntu-latest\n",
            "must be none/read/write",
        ),
        (
            "permissions:\n  contents: write\njobs:\n  build:\n    runs-on: ubuntu-latest\n",
            "unallowlisted workflow write grant",
        ),
        (
            (
                "permissions:\n  contents: read\njobs:\n  build:\n    runs-on: ubuntu-latest\n"
                "    permissions:\n      contents: write\n"
            ),
            "unallowlisted job write grant",
        ),
    ],
    ids=["missing", "write-all", "read-all", "unknown-level", "workflow-write", "job-write"],
)
def test_permission_guard_detects_the_violations_it_claims_to(
    tmp_path: Path,
    document: str,
    expected: str,
) -> None:
    (tmp_path / "probe.yml").write_text(document, encoding="utf-8")

    problems = permission_problems(
        tmp_path,
        allowed_workflow_writes=frozenset(),
        allowed_job_writes=frozenset(),
    )

    assert any(expected in problem for problem in problems), problems


def test_permission_guard_rejects_stale_write_allowlist_entries(tmp_path: Path) -> None:
    (tmp_path / "probe.yml").write_text(
        "permissions:\n  contents: read\njobs:\n  build:\n    runs-on: ubuntu-latest\n",
        encoding="utf-8",
    )

    problems = permission_problems(
        tmp_path,
        allowed_workflow_writes=frozenset({("probe.yml", "contents")}),
        allowed_job_writes=frozenset({("probe.yml", "build", "contents")}),
    )

    assert "allowlisted workflow write grant is missing: probe.yml:contents" in problems
    assert "allowlisted job write grant is missing: probe.yml:build:contents" in problems

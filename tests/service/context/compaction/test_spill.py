# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the dark Phase-4 spill quota and path skeleton."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from uuid import UUID

import pytest

from chrys.foundation.platform import get_platform
from chrys.foundation.tool_execution_stamp import EXECUTION_STAMP_KEY, build_execution_stamp
from chrys.foundation.tool_result_metadata import TOOL_ERROR_KIND_METADATA_KEY
from chrys.kernel import Content, Message
from chrys.service.context.compaction import spill as spill_mod
from chrys.service.context.compaction.scoped import ScopedGroup
from chrys.service.context.compaction.spill import (
    CATALOG_RELATIVE_PATH,
    CatalogRecord,
    SpillQuota,
    build_record_filename,
    catalog_live_records,
    dropped_turn_relative_path,
    reconcile_spill_storage,
    sub_agent_dropped_turn_relative_path,
    write_spill_batch,
)


def _tool_group(
    group_id: str,
    calls: list[tuple[str, str, dict[str, object], object, dict[str, object] | None]],
) -> ScopedGroup:
    call_contents = [
        Content.from_function_call(call_id, tool, arguments=arguments)
        for call_id, tool, arguments, _result, _metadata in calls
    ]
    result_contents = [
        Content.from_function_result(call_id, result=result, additional_properties=metadata)
        for call_id, _tool, _arguments, result, metadata in calls
    ]
    return ScopedGroup(
        group_id,
        "tool_call",
        (Message("assistant", call_contents), Message("tool", result_contents)),
        True,
    )


def _assistant_group(group_id: str, text: str) -> ScopedGroup:
    return ScopedGroup(group_id, "assistant", (Message("assistant", [Content.from_text(text)]),), True)


def test_spill_quota_reserve_commit_release_and_limit() -> None:
    quota = SpillQuota(limit_bytes=100)
    quota.initialize(20)

    assert quota.spent_bytes == 20
    assert quota.try_reserve(70) is True
    assert quota.try_reserve(11) is False

    quota.commit(70, 60)
    assert quota.spent_bytes == 80
    assert quota.try_reserve(20) is True
    assert quota.try_reserve(1) is False

    quota.release(20)
    assert quota.spent_bytes == 80
    assert quota.try_reserve(20) is True
    quota.commit(20, 20)
    assert quota.spent_bytes == 100
    assert quota.try_reserve(1) is False


def test_spill_quota_rejects_overcommit_without_consuming_reservation() -> None:
    quota = SpillQuota(limit_bytes=10)
    assert quota.try_reserve(10) is True

    with pytest.raises(ValueError, match="cannot exceed"):
        quota.commit(10, 11)

    quota.release(10)
    assert quota.try_reserve(10) is True


def test_spill_quota_initialize_sets_catalog_baseline() -> None:
    quota = SpillQuota(limit_bytes=100)
    old_path = "compactions/dropped/turn001/001_old_00000000000000000000000000000001.md"
    missing_path = "compactions/dropped/turn001/002_missing_00000000000000000000000000000002.md"
    quota.initialize(75, {old_path}, live_relative_paths=(old_path, missing_path))

    assert quota.spent_bytes == 75
    assert quota.is_record_available(old_path)
    assert not quota.is_record_available("compactions/dropped/turn001/missing.md")
    assert quota.live_record_count() == 2
    assert quota.live_record_count(excluded_relative_paths={missing_path}) == 1
    assert quota.try_reserve(25) is True
    new_path = "compactions/dropped/turn002/001_new_00000000000000000000000000000003.md"
    quota.commit(25, 20, relative_path=new_path)
    assert quota.live_record_count() == 3
    quota.reclaim(20, relative_path=new_path)
    assert quota.live_record_count() == 2
    assert quota.reclaim_record(missing_path) == 0
    assert quota.live_record_count() == 1


def test_spill_quota_disable_storage_fails_closed_until_reinitialized() -> None:
    quota = SpillQuota(limit_bytes=100)
    old_path = "compactions/dropped/turn001/001_old_00000000000000000000000000000001.md"
    quota.initialize(75, {old_path}, committed_bytes_by_path={old_path: 75})

    quota.disable_storage()

    assert not quota.storage_available
    assert quota.spent_bytes == 0
    assert not quota.is_record_available(old_path)
    assert quota.live_record_count() == 0
    assert quota.try_reserve(1) is False

    quota.initialize(20)
    assert quota.storage_available
    assert quota.try_reserve(80) is True


def test_write_spill_batch_preserves_disk_when_storage_is_unavailable(tmp_path: Path) -> None:
    retained = tmp_path / dropped_turn_relative_path(1) / "retained.md"
    retained.parent.mkdir(parents=True)
    retained.write_text("retained", encoding="utf-8")
    quota = SpillQuota()
    quota.disable_storage()

    result = write_spill_batch(
        tmp_path,
        quota,
        [_tool_group("g", [("c", "read_file", {"path": "x"}, "value", None)])],
        record_dir=dropped_turn_relative_path(2),
        absolute_turn=2,
        round_number=1,
        session_id="s",
    )

    assert not result.unexpected_io_failure
    assert result.entries[0].no_record_reason == "storage unavailable"
    assert retained.read_text(encoding="utf-8") == "retained"
    assert not (tmp_path / CATALOG_RELATIVE_PATH).exists()


def test_spill_quota_reclaim_restores_headroom_with_outstanding_reservations() -> None:
    quota = SpillQuota(limit_bytes=100)
    quota.initialize(80)
    assert quota.try_reserve(20) is True

    quota.reclaim(30)

    assert quota.spent_bytes == 50
    assert quota.try_reserve(30) is True
    assert quota.try_reserve(1) is False
    quota.release(20)
    quota.commit(30, 30)
    assert quota.spent_bytes == 80


def test_spill_quota_rejects_reclaim_beyond_committed_bytes() -> None:
    quota = SpillQuota(limit_bytes=100)
    quota.initialize(25)

    with pytest.raises(RuntimeError, match="exceeds committed"):
        quota.reclaim(26)

    assert quota.spent_bytes == 25


def test_spill_quota_concurrent_reservations_never_exceed_limit() -> None:
    quota = SpillQuota(limit_bytes=100)
    worker_count = 20
    barrier = Barrier(worker_count + 1)

    def _reserve_and_commit() -> bool:
        barrier.wait()
        if not quota.try_reserve(10):
            return False
        quota.commit(10, 10)
        return True

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_reserve_and_commit) for _ in range(worker_count)]
        barrier.wait()
        results = [future.result() for future in futures]

    assert results.count(True) == 10
    assert quota.spent_bytes == 100


def test_spill_catalog_and_record_path_shapes() -> None:
    assert Path("compactions/dropped/catalog.jsonl") == CATALOG_RELATIVE_PATH
    assert dropped_turn_relative_path(12) == Path("compactions/dropped/turn012")
    assert sub_agent_dropped_turn_relative_path("Explore/Agent", "inv:1", 2) == Path(
        "compactions/sub_agents/Explore_Agent/inv_1/dropped/turn002"
    )


def test_record_filename_reserves_full_collision_free_affixes() -> None:
    first_id = "a" * 8
    second_id = "b" * 8
    long_tool_name = "读取/文件:" * 100

    first = build_record_filename(7, long_tool_name, first_id)
    second = build_record_filename(7, long_tool_name, second_id)

    assert first.startswith("007_")
    assert first.endswith(f"_{first_id}.md")
    assert second.endswith(f"_{second_id}.md")
    assert first != second
    assert len(first.encode("utf-8")) <= 255
    assert len(second.encode("utf-8")) <= 255
    assert "/" not in first and ":" not in first


def test_record_filename_rejects_short_or_noncanonical_record_id() -> None:
    with pytest.raises(ValueError, match="8 lowercase hexadecimal"):
        build_record_filename(1, "read_file", "ABC")
    # The old full-uuid32 form is rejected too — the length is exact.
    with pytest.raises(ValueError, match="8 lowercase hexadecimal"):
        build_record_filename(1, "read_file", "a" * 32)


def test_spill_record_layout_parallel_calls_effective_args_and_image(tmp_path: Path) -> None:
    image_uri = "data:image/png;base64,aGVsbG8="
    image = Content.from_uri(image_uri, media_type="image/png")
    metadata = {
        EXECUTION_STAMP_KEY: build_execution_stamp(
            {"path": "effective.txt", "token": "do-not-render"},
            outcome="error",
            error_kind="read_failed",
        )
    }
    group = _tool_group(
        "parallel",
        [
            ("c1", "read_file", {"path": "requested.txt"}, [image], metadata),
            ("c2", "execute", {"command": "pwd"}, "ok", None),
        ],
    )
    quota = SpillQuota()

    result = write_spill_batch(
        tmp_path,
        quota,
        [group],
        record_dir=dropped_turn_relative_path(12),
        absolute_turn=12,
        round_number=2,
        session_id="session-1",
    )

    assert not result.unexpected_io_failure
    assert len(result.entries) == 2
    assert {entry.record_id for entry in result.entries} == {result.entries[0].record_id}
    assert result.entries[0].display_argument == 'path="effective.txt"'
    assert result.entries[0].outcome == "error(read_failed)"
    assert result.entries[1].outcome == "unknown"
    record_path = tmp_path / result.entries[0].relative_path
    record = record_path.read_text(encoding="utf-8")
    assert record.index("## Call: read_file") < record.index("## Call: execute")
    assert "Arguments (requested)" in record
    assert "Arguments (effective)" in record
    assert "requested.txt" in record and "effective.txt" in record
    assert image_uri not in record
    assert "[image: image/png, 5, sha256:2cf24dba5fb0a30e]" in record
    assert record.endswith("<!-- end of record -->\n")
    catalog = (tmp_path / CATALOG_RELATIVE_PATH).read_text(encoding="utf-8").splitlines()
    assert len(catalog) == 1
    assert json.loads(catalog[0])["relative_path"] == result.entries[0].relative_path
    manifest = record_path.with_name("manifest.md").read_text(encoding="utf-8")
    assert record_path.name in manifest


def test_spill_record_escapes_surrogates_as_valid_utf8(tmp_path: Path) -> None:
    surrogate_path = "artifact-\udcff.txt"
    group = _tool_group(
        "surrogate",
        [("c1", "read_file", {"path": surrogate_path}, f"read {surrogate_path}", None)],
    )

    result = write_spill_batch(
        tmp_path,
        SpillQuota(),
        [group],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="session-1",
    )

    record = (tmp_path / result.entries[0].relative_path).read_text(encoding="utf-8")
    assert "artifact-\\udcff.txt" in record
    assert record.endswith("<!-- end of record -->\n")


def test_spill_sanitizes_surrogate_in_tool_name_before_path_and_catalog_use(tmp_path: Path) -> None:
    tool_name = "tool-\ud800"
    group = _tool_group(
        "surrogate-tool",
        [("c1", tool_name, {"value": "safe"}, "result", None)],
    )

    result = write_spill_batch(
        tmp_path,
        SpillQuota(),
        [group],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="session-1",
    )

    assert not result.unexpected_io_failure
    assert result.entries[0].tool == r"tool-\ud800"
    record_path = tmp_path / result.entries[0].relative_path
    record_path.read_bytes().decode("utf-8", errors="strict")
    catalog_line = (tmp_path / CATALOG_RELATIVE_PATH).read_bytes().decode("utf-8", errors="strict")
    assert json.loads(catalog_line)["tool"] == r"tool-\ud800"


def test_spill_fsyncs_records_catalog_and_directory_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fsynced_files: set[tuple[int, int]] = set()
    fsynced_directories: list[Path] = []

    def remember_fsync(descriptor: int) -> None:
        stat = spill_mod.os.fstat(descriptor)
        fsynced_files.add((stat.st_dev, stat.st_ino))

    monkeypatch.setattr(spill_mod.os, "fsync", remember_fsync)
    monkeypatch.setattr(spill_mod, "fsync_directory", fsynced_directories.append)

    result = write_spill_batch(
        tmp_path,
        SpillQuota(),
        [_tool_group("durable", [("c1", "read_file", {"path": "a.txt"}, "contents", None)])],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
    )

    assert not result.unexpected_io_failure
    record_path = tmp_path / result.entries[0].relative_path
    catalog_path = tmp_path / CATALOG_RELATIVE_PATH
    record_stat = record_path.stat()
    catalog_stat = catalog_path.stat()
    assert (record_stat.st_dev, record_stat.st_ino) in fsynced_files
    assert (catalog_stat.st_dev, catalog_stat.st_ino) in fsynced_files
    assert record_path.parent in fsynced_directories
    assert catalog_path.parent in fsynced_directories
    assert tmp_path in fsynced_directories


def test_spill_record_preserves_hosted_call_inputs_and_pairs_image_results(tmp_path: Path) -> None:
    first_image = Content.from_uri("data:image/png;base64,Zmlyc3Q=", media_type="image/png")
    second_image = Content.from_uri("data:image/png;base64,c2Vjb25k", media_type="image/png")
    group = ScopedGroup(
        "hosted",
        "tool_call",
        (
            Message(
                "assistant",
                [
                    Content.from_shell_tool_call(
                        call_id="shell-1",
                        commands=["printf HOSTED_COMMAND"],
                        timeout_ms=1_234,
                    ),
                    Content.from_code_interpreter_tool_call(
                        call_id="code-1",
                        inputs=[Content.from_text("print('HOSTED_CODE')")],
                    ),
                    Content.from_image_generation_tool_call(image_id="image-1"),
                    Content.from_image_generation_tool_call(image_id="image-2"),
                ],
            ),
            Message(
                "tool",
                [
                    Content.from_shell_tool_result(
                        call_id="shell-1",
                        outputs=[Content.from_shell_command_output(stdout="shell output", exit_code=0)],
                    ),
                    Content.from_code_interpreter_tool_result(
                        call_id="code-1",
                        outputs=[Content.from_text("code output")],
                    ),
                    Content.from_image_generation_tool_result(image_id="image-1", outputs=first_image),
                    Content.from_image_generation_tool_result(image_id="image-2", outputs=second_image),
                ],
            ),
        ),
        True,
    )

    result = write_spill_batch(
        tmp_path,
        SpillQuota(),
        [group],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
    )

    assert not result.unexpected_io_failure
    assert len(result.entries) == 4
    assert result.entries[0].display_argument.startswith("commands=")
    assert result.entries[1].display_argument.startswith("inputs=")
    assert result.entries[2].display_argument == 'image_id="image-1"'
    record = (tmp_path / result.entries[0].relative_path).read_text(encoding="utf-8")
    assert "HOSTED_COMMAND" in record
    assert "HOSTED_CODE" in record
    assert '"timeout_ms": 1234' in record
    assert '"image_id": "image-1"' in record
    assert '"image_id": "image-2"' in record
    assert record.count(hashlib.sha256(b"first").hexdigest()[:16]) == 1
    assert record.count(hashlib.sha256(b"second").hexdigest()[:16]) == 1


def test_spill_record_preserves_shell_status_and_sibling_hosted_files(tmp_path: Path) -> None:
    group = ScopedGroup(
        "hosted-shell",
        "tool_call",
        (
            Message(
                "assistant",
                [
                    Content.from_shell_tool_call(call_id="shell-failed", commands=["false"]),
                    Content.from_shell_tool_call(call_id="shell-timeout", commands=["sleep 60"]),
                ],
            ),
            Message(
                "tool",
                [
                    Content.from_hosted_file("artifact-file-id", name="artifact.txt"),
                    Content.from_shell_tool_result(
                        call_id="shell-failed",
                        outputs=[Content.from_shell_command_output(stdout="", stderr="", exit_code=7, timed_out=False)],
                    ),
                    Content.from_shell_tool_result(
                        call_id="shell-timeout",
                        outputs=[Content.from_shell_command_output(stdout="", stderr="", timed_out=True)],
                    ),
                ],
            ),
        ),
        True,
    )

    result = write_spill_batch(
        tmp_path,
        SpillQuota(),
        [group],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
    )

    assert not result.unexpected_io_failure
    assert [entry.outcome for entry in result.entries] == ["error(exit_code)", "error(timeout)"]
    record = (tmp_path / result.entries[0].relative_path).read_text(encoding="utf-8")
    assert "[exit_code]\n7" in record
    assert "[timed_out]\nfalse" in record
    assert "[timed_out]\ntrue" in record
    assert "## Additional group content" in record
    assert "artifact-file-id" in record
    assert "artifact.txt" in record


def test_spill_records_preserve_reasoning_and_tool_group_assistant_context(tmp_path: Path) -> None:
    reasoning = ScopedGroup(
        "reasoning",
        "assistant",
        (
            Message(
                "assistant",
                [Content.from_text_reasoning(text="NEGATIVE_INVARIANT", protected_data="signed-state")],
            ),
        ),
        True,
    )
    tool_group = ScopedGroup(
        "tool-with-context",
        "tool_call",
        (
            Message(
                "assistant",
                [
                    Content.from_text_reasoning(text="inspect before calling"),
                    Content.from_text("visible preface"),
                    Content.from_function_call("c1", "read_file", arguments={"path": "a.txt"}),
                ],
            ),
            Message("tool", [Content.from_function_result("c1", result="contents")]),
        ),
        True,
    )

    result = write_spill_batch(
        tmp_path,
        SpillQuota(),
        [reasoning, tool_group],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
    )

    reasoning_record = (tmp_path / result.entries[0].relative_path).read_text(encoding="utf-8")
    tool_record = (tmp_path / result.entries[1].relative_path).read_text(encoding="utf-8")
    assert "[reasoning]\nNEGATIVE_INVARIANT" in reasoning_record
    assert "[protected reasoning data]\nsigned-state" in reasoning_record
    assert result.entries[0].size_chars > 0
    assert "## Assistant context" in tool_record
    assert "[reasoning]\ninspect before calling" in tool_record
    assert "visible preface" in tool_record
    assert "## Call: read_file" in tool_record


def test_spill_records_retain_format_stamped_reasoning(tmp_path: Path) -> None:
    """Dialect-stamped reasoning (GLM/DeepSeek ``reasoning_content``) keeps the
    same owner-only local retention as other reasoning: streamed text spills as
    ``[reasoning]`` and protected JSON as ``[protected reasoning data]``."""
    group = ScopedGroup(
        "reasoning",
        "assistant",
        (
            Message(
                "assistant",
                [
                    Content.from_text_reasoning(
                        text="GLM streamed thinking",
                        additional_properties={"openai_reasoning_format": "reasoning_content"},
                    ),
                    Content.from_text_reasoning(
                        protected_data='"GLM protected thinking"',
                        additional_properties={"openai_reasoning_format": "reasoning_content"},
                    ),
                ],
            ),
        ),
        True,
    )

    result = write_spill_batch(
        tmp_path,
        SpillQuota(),
        [group],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
    )

    record = (tmp_path / result.entries[0].relative_path).read_text(encoding="utf-8")
    assert "[reasoning]\nGLM streamed thinking" in record
    assert '[protected reasoning data]\n"GLM protected thinking"' in record


def test_spill_record_is_created_owner_only(tmp_path: Path) -> None:
    if get_platform().is_windows:
        pytest.skip("POSIX permission bits are not authoritative on Windows")
    previous_umask = os.umask(0)
    try:
        result = write_spill_batch(
            tmp_path,
            SpillQuota(),
            [_assistant_group("private", "sensitive context")],
            record_dir=dropped_turn_relative_path(1),
            absolute_turn=1,
            round_number=1,
            session_id="s",
        )
    finally:
        os.umask(previous_umask)

    record_path = tmp_path / result.entries[0].relative_path
    assert record_path.stat().st_mode & 0o777 == 0o600


def test_spill_batch_rebuilds_manifest_projection_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rebuild_calls: list[Path] = []
    rebuild = spill_mod._rebuild_manifest_projections

    def rebuild_spy(root: Path, **kwargs: object) -> None:
        rebuild_calls.append(root)
        rebuild(root, **kwargs)

    monkeypatch.setattr(spill_mod, "_rebuild_manifest_projections", rebuild_spy)
    result = write_spill_batch(
        tmp_path,
        SpillQuota(),
        [_assistant_group(f"g{index}", f"text {index}") for index in range(3)],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
    )

    assert rebuild_calls == [tmp_path.resolve()]
    manifest = (tmp_path / result.entries[0].relative_path).with_name("manifest.md").read_text(encoding="utf-8")
    assert all(Path(entry.relative_path).name in manifest for entry in result.entries)


def test_spill_batch_updates_only_dirty_projection_without_rescanning_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quota = SpillQuota()
    first = write_spill_batch(
        tmp_path,
        quota,
        [_assistant_group("first", "first")],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
    )
    first_manifest = (tmp_path / first.entries[0].relative_path).with_name("manifest.md")
    first_projection = first_manifest.read_text(encoding="utf-8")
    projection_writes: list[Path] = []
    validated_record_paths: list[str] = []
    atomic_write_text = spill_mod.atomic_write_text
    record_path_within_root = spill_mod._record_path_within_root

    def remember_projection(path: Path, payload: str, *, encoding: str = "utf-8") -> None:
        projection_writes.append(path)
        atomic_write_text(path, payload, encoding=encoding)

    def remember_validation(root: Path, relative_path: str | Path) -> bool:
        validated_record_paths.append(Path(relative_path).as_posix())
        return record_path_within_root(root, relative_path)

    monkeypatch.setattr(spill_mod, "_read_live_catalog", lambda _root: pytest.fail("batch rescanned catalog"))
    monkeypatch.setattr(spill_mod, "atomic_write_text", remember_projection)
    monkeypatch.setattr(spill_mod, "_record_path_within_root", remember_validation)

    second = write_spill_batch(
        tmp_path,
        quota,
        [_assistant_group("second", "second")],
        record_dir=dropped_turn_relative_path(2),
        absolute_turn=2,
        round_number=1,
        session_id="s",
    )

    second_manifest = (tmp_path / second.entries[0].relative_path).with_name("manifest.md")
    assert projection_writes == [second_manifest]
    assert set(validated_record_paths) == {second.entries[0].relative_path}
    assert first_manifest.read_text(encoding="utf-8") == first_projection


def test_spill_record_middle_truncates_with_original_size_and_terminator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(spill_mod, "PER_RECORD_CAP_BYTES", 1_024)
    group = _tool_group("large", [("c1", "execute", {"command": "large"}, "x" * 5_000, None)])

    result = write_spill_batch(
        tmp_path,
        SpillQuota(),
        [group],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
    )

    payload = (tmp_path / result.entries[0].relative_path).read_bytes()
    assert len(payload) <= 1_024
    assert b"middle-truncated from" in payload
    assert payload.endswith(b"<!-- end of record -->\n")


def test_spill_absent_stamp_uses_requested_label_structured_envelope_or_unknown(tmp_path: Path) -> None:
    group = _tool_group(
        "absent-stamps",
        [
            (
                "c1",
                "read_file",
                {"path": "missing.txt"},
                "Error: missing",
                {TOOL_ERROR_KIND_METADATA_KEY: "path_not_found"},
            ),
            ("c2", "custom_tool", {"opaque": "value"}, "looks like an error, but is prose", None),
        ],
    )

    result = write_spill_batch(
        tmp_path,
        SpillQuota(),
        [group],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
    )

    assert [entry.outcome for entry in result.entries] == ["error(path_not_found)", "unknown"]
    assert result.entries[1].display_argument == ""
    record = (tmp_path / result.entries[0].relative_path).read_text(encoding="utf-8")
    assert record.count("Arguments (as requested)") == 2


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("read_file", {"path": "sk-secretvalue123456789"}),
        ("fetch", {"url": "https://example.test/?api_key=hunter2"}),
        ("fetch", {"url": "https://example.test/?key=AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz1234567"}),
        ("fetch", {"url": "https://user:hunter2@example.test/path"}),
        ("open_url", {"url": "https://example.test/#access_token=hunter2"}),
        ("execute", {"command": "curl https://example.test --password hunter2"}),
    ],
)
def test_manifest_display_argument_rejects_sensitive_scalar_content(
    tmp_path: Path,
    tool: str,
    arguments: dict[str, object],
) -> None:
    group = _tool_group(
        "sensitive",
        [("c1", tool, arguments, "value", None)],
    )

    result = write_spill_batch(
        tmp_path,
        SpillQuota(),
        [group],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
    )

    assert result.entries[0].display_argument == ""


def test_manifest_display_argument_per_key_caps_and_truncation_styles(tmp_path: Path) -> None:
    """Paths get the wide cap with middle truncation (filename tail survives);
    commands keep the default cap with end truncation; shell calls append the
    ``reason`` segment under its own cap."""
    from chrys.service.context.compaction.spill import (
        DISPLAY_ARGUMENT_DEFAULT_MAX_CHARS,
        DISPLAY_ARGUMENT_PATH_MAX_CHARS,
        DISPLAY_ARGUMENT_REASON_MAX_CHARS,
    )

    fits = "/Users/example/Repos/Workspace/project/src/package/service/profiles/models/loader_module_name.py"
    long_path = "/repo/" + "component/" * 30 + "deep_leaf_module.py"
    long_command = "x" * 300
    long_reason = "r" * 300
    group = _tool_group(
        "long-args",
        [
            ("c1", "read_file", {"path": fits}, "value", None),
            ("c2", "read_file", {"path": long_path}, "value", None),
            ("c3", "execute", {"command": long_command}, "value", None),
            ("c4", "zsh", {"command": "ls src", "reason": "List sources"}, "value", None),
            ("c5", "zsh", {"command": "ls src", "reason": long_reason}, "value", None),
            ("c6", "zsh", {"command": "ls src", "reason": "token=abc123"}, "value", None),
            ("c7", "write_file", {"path": long_path, "content": "body"}, "value", None),
            ("c8", "edit_file", {"file_path": long_path, "old_string": "a"}, "value", None),
        ],
    )

    result = write_spill_batch(
        tmp_path,
        SpillQuota(),
        [group],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
    )

    assert result.entries[0].display_argument == f'path="{fits}"'
    middle_capped = result.entries[1].display_argument
    assert len(middle_capped) == DISPLAY_ARGUMENT_PATH_MAX_CHARS
    assert "…" in middle_capped
    assert middle_capped.startswith('path="/repo/')
    assert middle_capped.endswith('deep_leaf_module.py"')
    end_capped = result.entries[2].display_argument
    assert len(end_capped) == DISPLAY_ARGUMENT_DEFAULT_MAX_CHARS
    assert end_capped.startswith('command="xxx')
    assert end_capped.endswith("…")
    assert result.entries[3].display_argument == 'command="ls src", reason="List sources"'
    with_long_reason = result.entries[4].display_argument
    command_segment, _, reason_segment = with_long_reason.partition(", ")
    assert command_segment == 'command="ls src"'
    assert len(reason_segment) == DISPLAY_ARGUMENT_REASON_MAX_CHARS
    assert reason_segment.startswith('reason="rrr')
    assert reason_segment.endswith("…")
    assert result.entries[5].display_argument == 'command="ls src"'
    write_capped = result.entries[6].display_argument
    assert len(write_capped) == DISPLAY_ARGUMENT_PATH_MAX_CHARS
    assert write_capped.startswith('path="/repo/')
    assert write_capped.endswith('deep_leaf_module.py"')
    edit_capped = result.entries[7].display_argument
    assert len(edit_capped) == DISPLAY_ARGUMENT_PATH_MAX_CHARS
    assert edit_capped.startswith('file_path="/repo/')
    assert edit_capped.endswith('deep_leaf_module.py"')


def test_catalog_reader_skips_invalid_utf8_line_and_keeps_valid_records(tmp_path: Path) -> None:
    result = write_spill_batch(
        tmp_path,
        SpillQuota(),
        [_tool_group("valid", [("c1", "read_file", {"path": "safe.txt"}, "value", None)])],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
    )
    catalog = tmp_path / CATALOG_RELATIVE_PATH
    with catalog.open("ab") as output:
        output.write(b"\xff\n")

    live = catalog_live_records(tmp_path)
    quota = SpillQuota()
    reconciliation = reconcile_spill_storage(tmp_path, quota)

    assert [record.record_id for record in live] == [result.entries[0].record_id]
    assert reconciliation.live_record_count == 1
    assert quota.spent_bytes == (tmp_path / result.entries[0].relative_path).stat().st_size


def test_spill_quota_has_catalog_record_id_tracks_commit_and_reclaim() -> None:
    quota = SpillQuota()
    relative_path = (dropped_turn_relative_path(1) / build_record_filename(1, "read_file", "a" * 8)).as_posix()
    record = CatalogRecord(
        record_id="a" * 8,
        relative_path=relative_path,
        turn=1,
        round=1,
        tool="read_file",
        bytes=64,
        created_at="now",
    )

    assert not quota.has_catalog_record_id(record.record_id)
    assert quota.try_reserve(64)
    quota.commit(64, 64, catalog_record=record)
    assert quota.has_catalog_record_id(record.record_id)
    quota.reclaim_record(relative_path)
    assert not quota.has_catalog_record_id(record.record_id)


def test_catalog_reader_keeps_first_record_on_duplicate_id_and_allows_tombstone_readd(tmp_path: Path) -> None:
    group = _tool_group("record", [("c1", "read_file", {"path": "safe.txt"}, "value", None)])
    write_spill_batch(
        tmp_path,
        SpillQuota(),
        [group],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
    )
    first_record = catalog_live_records(tmp_path)[0]
    duplicate = first_record.to_catalog_line()
    duplicate["relative_path"] = (
        dropped_turn_relative_path(2) / build_record_filename(1, "read_file", first_record.record_id)
    ).as_posix()
    catalog = tmp_path / CATALOG_RELATIVE_PATH
    with catalog.open("a", encoding="utf-8") as output:
        output.write(json.dumps(duplicate) + "\n")

    live = catalog_live_records(tmp_path)
    assert [record.relative_path for record in live] == [first_record.relative_path]

    # A tombstone frees the ID, so a later line may legitimately re-add it
    # (the eviction unlink-failure restore path relies on this replay).
    with catalog.open("a", encoding="utf-8") as output:
        output.write(json.dumps({"record_id": first_record.record_id, "tombstone": True}) + "\n")
        output.write(json.dumps(duplicate) + "\n")

    live = catalog_live_records(tmp_path)
    assert [record.relative_path for record in live] == [duplicate["relative_path"]]


def test_catalog_reader_reused_tombstoned_id_takes_newest_position(tmp_path: Path) -> None:
    """A record reusing a tombstoned ID must not inherit the dead record's
    catalog position — eviction order would treat the newest spill as oldest."""
    quota = SpillQuota()
    group = _tool_group("record", [("c1", "read_file", {"path": "safe.txt"}, "value", None)])
    write_spill_batch(
        tmp_path,
        quota,
        [group],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
    )
    write_spill_batch(
        tmp_path,
        quota,
        [group],
        record_dir=dropped_turn_relative_path(2),
        absolute_turn=2,
        round_number=1,
        session_id="s",
    )
    older, newer = catalog_live_records(tmp_path)
    reused = older.to_catalog_line()
    reused["relative_path"] = (
        dropped_turn_relative_path(3) / build_record_filename(1, "read_file", older.record_id)
    ).as_posix()
    catalog = tmp_path / CATALOG_RELATIVE_PATH
    with catalog.open("a", encoding="utf-8") as output:
        output.write(json.dumps({"record_id": older.record_id, "tombstone": True}) + "\n")
        output.write(json.dumps(reused) + "\n")

    live = catalog_live_records(tmp_path)
    assert [record.relative_path for record in live] == [newer.relative_path, reused["relative_path"]]


def test_catalog_reader_rejects_duplicate_live_path_without_phantom_quota(tmp_path: Path) -> None:
    group = _tool_group("record", [("c1", "read_file", {"path": "safe.txt"}, "value", None)])
    first = write_spill_batch(
        tmp_path,
        SpillQuota(),
        [group],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
    )
    first_path = tmp_path / first.entries[0].relative_path
    first_size = first_path.stat().st_size
    first_record = catalog_live_records(tmp_path)[0]
    duplicate = first_record.to_catalog_line()
    duplicate["record_id"] = "f" * 8
    catalog = tmp_path / CATALOG_RELATIVE_PATH
    with catalog.open("a", encoding="utf-8") as output:
        output.write(json.dumps(duplicate) + "\n")

    quota = SpillQuota(limit_bytes=first_size + 200)
    reconciliation = reconcile_spill_storage(tmp_path, quota)

    assert [record.record_id for record in catalog_live_records(tmp_path)] == [first_record.record_id]
    assert reconciliation.live_total_bytes == first_size
    assert reconciliation.live_record_count == 1
    assert quota.spent_bytes == first_size
    assert quota.live_record_count() == 1

    second = write_spill_batch(
        tmp_path,
        quota,
        [group],
        record_dir=dropped_turn_relative_path(2),
        absolute_turn=2,
        round_number=1,
        session_id="s",
    )

    second_path = tmp_path / second.entries[0].relative_path
    assert second_path.is_file()
    assert not first_path.exists()
    assert quota.spent_bytes == second_path.stat().st_size
    assert [record.record_id for record in catalog_live_records(tmp_path)] == [second.entries[0].record_id]


def test_catalog_append_isolates_new_record_after_unterminated_torn_tail(tmp_path: Path) -> None:
    catalog = tmp_path / CATALOG_RELATIVE_PATH
    catalog.parent.mkdir(parents=True)
    catalog.write_bytes(b'{"record_id":"torn')

    result = write_spill_batch(
        tmp_path,
        SpillQuota(),
        [_tool_group("new", [("c1", "read_file", {"path": "safe.txt"}, "value", None)])],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
    )
    record_path = tmp_path / result.entries[0].relative_path

    assert not result.unexpected_io_failure
    assert [record.record_id for record in catalog_live_records(tmp_path)] == [result.entries[0].record_id]
    restored_quota = SpillQuota()
    reconciliation = reconcile_spill_storage(tmp_path, restored_quota)
    assert reconciliation.deleted_orphans == 0
    assert record_path.is_file()
    assert restored_quota.spent_bytes == record_path.stat().st_size


def test_spill_round_cap_refuses_triggering_and_remaining_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _assistant_group("first", "small")
    second = _assistant_group("second", "y" * 2_000)
    third = _assistant_group("third", "small")
    first_size = len(
        spill_mod._render_record(
            first,
            record_id="a" * 8,
            absolute_turn=1,
            round_number=1,
            created_at="now",
            session_id="s",
        )
    )
    monkeypatch.setattr(spill_mod, "PER_ROUND_CAP_BYTES", first_size + 100)

    result = write_spill_batch(
        tmp_path,
        SpillQuota(),
        [first, second, third],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
    )

    assert result.entries[0].relative_path
    assert [entry.no_record_reason for entry in result.entries[1:]] == ["round cap", "round cap"]
    assert len(catalog_live_records(tmp_path)) == 1


def test_spill_note_record_archives_superseded_note_with_note_kind(tmp_path: Path) -> None:
    group = _tool_group("g", [("c1", "read_file", {"path": "a.txt"}, "value", None)])
    quota = SpillQuota()
    note = "## Task\nFinish the migration.\n\n## Progress\nHalfway there.\n"

    result = write_spill_batch(
        tmp_path,
        quota,
        [group],
        record_dir=dropped_turn_relative_path(3),
        absolute_turn=3,
        round_number=2,
        session_id="session-1",
        superseded_note=note,
    )

    assert not result.unexpected_io_failure
    assert len(result.entries) == 2
    note_entry = result.entries[-1]
    assert note_entry.tool == "last_words"
    assert note_entry.sequence == 2
    assert note_entry.group_id == "superseded_note"
    assert note_entry.outcome == "merged"
    assert note_entry.display_argument == "superseded by this round's note"
    assert note_entry.size_chars == len(note)
    assert not note_entry.assistant_text
    assert note_entry.available
    record = (tmp_path / note_entry.relative_path).read_text(encoding="utf-8")
    assert record.startswith("# Superseded LAST_WORDS note\n")
    assert note in record
    assert record.endswith("<!-- end of record -->\n")
    catalog = [json.loads(line) for line in (tmp_path / CATALOG_RELATIVE_PATH).read_text(encoding="utf-8").splitlines()]
    kinds = {row["record_id"]: row["kind"] for row in catalog}
    assert kinds == {result.entries[0].record_id: "group", note_entry.record_id: "note"}
    assert quota.spent_bytes == sum(row["bytes"] for row in catalog)
    manifest = (tmp_path / note_entry.relative_path).with_name("manifest.md").read_text(encoding="utf-8")
    assert Path(note_entry.relative_path).name in manifest


def test_spill_note_record_skipped_when_storage_unavailable(tmp_path: Path) -> None:
    quota = SpillQuota()
    quota.disable_storage()

    result = write_spill_batch(
        tmp_path,
        quota,
        [_tool_group("g", [("c", "read_file", {"path": "x"}, "value", None)])],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
        superseded_note="## Task\nnote\n",
    )

    assert [entry.tool for entry in result.entries] == ["read_file"]
    assert result.entries[0].no_record_reason == "storage unavailable"
    assert not (tmp_path / CATALOG_RELATIVE_PATH).exists()


def test_spill_note_record_quota_pressure_skips_note_without_eviction(tmp_path: Path) -> None:
    group = _tool_group("g", [("c1", "read_file", {"path": "a.txt"}, "value", None)])
    quota = SpillQuota(limit_bytes=64 * 1024)
    note = "n" * 100_000

    result = write_spill_batch(
        tmp_path,
        quota,
        [group],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
        superseded_note=note,
    )

    assert not result.unexpected_io_failure
    assert result.evicted_relative_paths == frozenset()
    assert [entry.tool for entry in result.entries] == ["read_file"]
    assert result.entries[0].available
    assert (tmp_path / result.entries[0].relative_path).exists()
    assert len(catalog_live_records(tmp_path)) == 1


def test_spill_note_record_still_written_when_round_cap_stops_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _assistant_group("first", "small")
    second = _assistant_group("second", "y" * 2_000)
    first_size = len(
        spill_mod._render_record(
            first,
            record_id="a" * 8,
            absolute_turn=1,
            round_number=1,
            created_at="now",
            session_id="s",
        )
    )
    monkeypatch.setattr(spill_mod, "PER_ROUND_CAP_BYTES", first_size + 100)

    result = write_spill_batch(
        tmp_path,
        SpillQuota(),
        [first, second],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
        superseded_note="## Task\nnote\n",
    )

    assert result.entries[1].no_record_reason == "round cap"
    note_entry = result.entries[-1]
    assert note_entry.tool == "last_words"
    assert note_entry.available
    assert (tmp_path / note_entry.relative_path).exists()


def test_spill_note_record_write_failure_is_swallowed_without_durability_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = _tool_group("g", [("c1", "read_file", {"path": "a.txt"}, "value", None)])
    quota = SpillQuota()
    real_append = spill_mod._append_catalog_line

    def failing_append(root: Path, value: dict[str, object]) -> None:
        if value.get("kind") == "note":
            raise OSError("catalog append refused")
        real_append(root, value)

    monkeypatch.setattr(spill_mod, "_append_catalog_line", failing_append)

    result = write_spill_batch(
        tmp_path,
        quota,
        [group],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
        superseded_note="## Task\nnote\n",
    )

    assert not result.unexpected_io_failure
    assert [entry.tool for entry in result.entries] == ["read_file"]
    record_path = tmp_path / result.entries[0].relative_path
    assert not list(record_path.parent.glob("*last_words*"))
    catalog = (tmp_path / CATALOG_RELATIVE_PATH).read_text(encoding="utf-8").splitlines()
    assert len(catalog) == 1
    assert quota.spent_bytes == json.loads(catalog[0])["bytes"]


def test_catalog_reader_defaults_missing_or_invalid_kind_to_group(tmp_path: Path) -> None:
    catalog_path = tmp_path / CATALOG_RELATIVE_PATH
    catalog_path.parent.mkdir(parents=True)
    turn_dir = dropped_turn_relative_path(1)
    legacy = {
        "record_id": "aaaaaaaa",
        "relative_path": (turn_dir / build_record_filename(1, "read_file", "aaaaaaaa")).as_posix(),
        "turn": 1,
        "round": 1,
        "tool": "read_file",
        "bytes": 10,
        "created_at": "now",
    }
    invalid_kind = dict(
        legacy,
        record_id="bbbbbbbb",
        relative_path=(turn_dir / build_record_filename(2, "read_file", "bbbbbbbb")).as_posix(),
        kind=7,
    )
    note = dict(
        legacy,
        record_id="cccccccc",
        relative_path=(turn_dir / build_record_filename(3, "last_words", "cccccccc")).as_posix(),
        tool="last_words",
        kind="note",
    )
    catalog_path.write_text(
        "\n".join(json.dumps(row) for row in (legacy, invalid_kind, note)) + "\n",
        encoding="utf-8",
    )

    records = {record.record_id: record for record in catalog_live_records(tmp_path)}

    assert records["aaaaaaaa"].kind == "group"
    assert records["bbbbbbbb"].kind == "group"
    assert records["cccccccc"].kind == "note"


def test_spill_exclusive_create_collision_regenerates_once_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = iter((UUID(hex="1" * 32), UUID(hex="2" * 32)))
    monkeypatch.setattr(spill_mod, "uuid4", lambda: next(ids))
    record_dir = dropped_turn_relative_path(1)
    collision = tmp_path / record_dir / build_record_filename(1, "read_file", "1" * 8)
    collision.parent.mkdir(parents=True)
    collision.write_text("existing", encoding="utf-8")
    group = _tool_group("g", [("c", "read_file", {"path": "a"}, "data", None)])

    result = write_spill_batch(
        tmp_path,
        SpillQuota(),
        [group],
        record_dir=record_dir,
        absolute_turn=1,
        round_number=1,
        session_id="s",
    )

    assert collision.read_text(encoding="utf-8") == "existing"
    assert result.entries[0].record_id == "2" * 8
    assert Path(result.entries[0].relative_path).name.endswith(f"_{'2' * 8}.md")


def test_spill_record_id_regenerated_when_live_in_another_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Truncated IDs stay session-unique: a draw colliding with a live record in
    a different turn directory is redrawn instead of poisoning ``commit()``."""
    quota = SpillQuota()
    first = write_spill_batch(
        tmp_path,
        quota,
        [_tool_group("g1", [("c1", "read_file", {"path": "a"}, "data", None)])],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
    )
    first_id = first.entries[0].record_id

    ids = iter((UUID(hex=first_id * 4), UUID(hex="7" * 32)))
    monkeypatch.setattr(spill_mod, "uuid4", lambda: next(ids))
    second = write_spill_batch(
        tmp_path,
        quota,
        [_tool_group("g2", [("c2", "read_file", {"path": "b"}, "data", None)])],
        record_dir=dropped_turn_relative_path(2),
        absolute_turn=2,
        round_number=1,
        session_id="s",
    )

    assert second.entries[0].record_id == "7" * 8
    assert (tmp_path / first.entries[0].relative_path).is_file()
    assert (tmp_path / second.entries[0].relative_path).is_file()
    assert {record.record_id for record in catalog_live_records(tmp_path)} == {first_id, "7" * 8}


def test_quota_pressure_tombstones_oldest_live_record_and_reclaims(tmp_path: Path) -> None:
    quota = SpillQuota(limit_bytes=1_200)
    first = write_spill_batch(
        tmp_path,
        quota,
        [_tool_group("old", [("c1", "execute", {"command": "one"}, "a" * 400, None)])],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
    )
    old_path = tmp_path / first.entries[0].relative_path
    assert old_path.exists()

    second = write_spill_batch(
        tmp_path,
        quota,
        [_tool_group("new", [("c2", "execute", {"command": "two"}, "b" * 400, None)])],
        record_dir=dropped_turn_relative_path(2),
        absolute_turn=2,
        round_number=1,
        session_id="s",
    )

    assert second.entries[0].relative_path
    assert second.evicted_relative_paths == frozenset({first.entries[0].relative_path})
    assert not old_path.exists()
    lines = [json.loads(line) for line in (tmp_path / CATALOG_RELATIVE_PATH).read_text().splitlines()]
    assert any(line.get("record_id") == first.entries[0].record_id and line.get("tombstone") for line in lines)
    live = catalog_live_records(tmp_path)
    assert [record.record_id for record in live] == [second.entries[0].record_id]
    assert quota.spent_bytes == (tmp_path / second.entries[0].relative_path).stat().st_size
    assert not quota.is_record_available(first.entries[0].relative_path)
    assert quota.is_record_available(second.entries[0].relative_path)


def test_quota_pressure_reclaims_accounted_bytes_when_old_record_is_missing(tmp_path: Path) -> None:
    first = write_spill_batch(
        tmp_path,
        SpillQuota(),
        [_tool_group("old", [("c1", "execute", {"command": "one"}, "a" * 400, None)])],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
    )
    old_path = tmp_path / first.entries[0].relative_path
    first_size = old_path.stat().st_size
    quota = SpillQuota(limit_bytes=first_size)
    reconcile_spill_storage(tmp_path, quota)
    old_path.unlink()

    second = write_spill_batch(
        tmp_path,
        quota,
        [_tool_group("new", [("c2", "execute", {"command": "two"}, "b" * 400, None)])],
        record_dir=dropped_turn_relative_path(2),
        absolute_turn=2,
        round_number=1,
        session_id="s",
    )

    assert second.entries[0].relative_path
    assert second.evicted_relative_paths == frozenset({first.entries[0].relative_path})
    assert quota.spent_bytes == (tmp_path / second.entries[0].relative_path).stat().st_size
    assert [record.record_id for record in catalog_live_records(tmp_path)] == [second.entries[0].record_id]


def test_quota_does_not_reclaim_catalog_bytes_missing_at_reconciliation(tmp_path: Path) -> None:
    missing = write_spill_batch(
        tmp_path,
        SpillQuota(),
        [_tool_group("missing", [("c1", "execute", {"command": "one"}, "a" * 400, None)])],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
    )
    live = write_spill_batch(
        tmp_path,
        SpillQuota(),
        [_tool_group("live", [("c2", "execute", {"command": "two"}, "b" * 400, None)])],
        record_dir=dropped_turn_relative_path(2),
        absolute_turn=2,
        round_number=1,
        session_id="s",
    )
    missing_path = tmp_path / missing.entries[0].relative_path
    live_path = tmp_path / live.entries[0].relative_path
    missing_path.unlink()
    limit_bytes = live_path.stat().st_size
    quota = SpillQuota(limit_bytes=limit_bytes)
    reconcile_spill_storage(tmp_path, quota)

    newest = write_spill_batch(
        tmp_path,
        quota,
        [_tool_group("new", [("c3", "execute", {"command": "new"}, "c" * 400, None)])],
        record_dir=dropped_turn_relative_path(3),
        absolute_turn=3,
        round_number=1,
        session_id="s",
    )

    newest_path = tmp_path / newest.entries[0].relative_path
    assert newest.evicted_relative_paths == frozenset({missing.entries[0].relative_path, live.entries[0].relative_path})
    assert not live_path.exists()
    assert quota.spent_bytes == newest_path.stat().st_size
    assert sum(path.stat().st_size for path in tmp_path.rglob("*.md") if path.name != "manifest.md") <= limit_bytes


def test_catalog_record_cannot_target_session_file_for_eviction(tmp_path: Path) -> None:
    session_file = tmp_path / "session.json"
    session_file.write_text("session state", encoding="utf-8")
    catalog = tmp_path / CATALOG_RELATIVE_PATH
    catalog.parent.mkdir(parents=True)
    catalog.write_text(
        json.dumps(
            {
                "record_id": "unsafe",
                "relative_path": "session.json",
                "turn": 1,
                "round": 1,
                "tool": "execute",
                "bytes": session_file.stat().st_size,
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    quota = SpillQuota(limit_bytes=1_200)

    reconciliation = reconcile_spill_storage(tmp_path, quota)
    result = write_spill_batch(
        tmp_path,
        quota,
        [_tool_group("new", [("c2", "execute", {"command": "two"}, "b" * 400, None)])],
        record_dir=dropped_turn_relative_path(2),
        absolute_turn=2,
        round_number=1,
        session_id="s",
    )

    assert reconciliation.live_record_count == 0
    assert [record.record_id for record in catalog_live_records(tmp_path)] == [result.entries[0].record_id]
    assert session_file.read_text(encoding="utf-8") == "session state"


def test_catalog_record_cannot_escape_session_through_symlink(tmp_path: Path) -> None:
    session_root = tmp_path / "session"
    outside = tmp_path / "outside"
    outside.mkdir()
    escaped_record = outside / "001_execute_00000000000000000000000000000001.md"
    escaped_record.write_text("outside record", encoding="utf-8")
    turn_directory = session_root / dropped_turn_relative_path(1)
    turn_directory.parent.mkdir(parents=True)
    try:
        turn_directory.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    catalog = session_root / CATALOG_RELATIVE_PATH
    catalog.write_text(
        json.dumps(
            {
                "record_id": "unsafe-symlink",
                "relative_path": ("compactions/dropped/turn001/001_execute_00000000000000000000000000000001.md"),
                "turn": 1,
                "round": 1,
                "tool": "execute",
                "bytes": escaped_record.stat().st_size,
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    quota = SpillQuota()

    reconciliation = reconcile_spill_storage(session_root, quota)

    assert reconciliation.live_record_count == 0
    assert catalog_live_records(session_root) == []
    assert escaped_record.read_text(encoding="utf-8") == "outside record"


def test_spill_rejects_turn_directory_symlinked_within_session_root(tmp_path: Path) -> None:
    session_root = tmp_path / "session"
    session_root.mkdir()
    unrelated_manifest = session_root / "manifest.md"
    unrelated_manifest.write_text("unrelated\n", encoding="utf-8")
    turn_directory = session_root / dropped_turn_relative_path(1)
    turn_directory.parent.mkdir(parents=True)
    try:
        turn_directory.symlink_to(session_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    quota = SpillQuota(limit_bytes=4_096)

    result = write_spill_batch(
        session_root,
        quota,
        [_tool_group("g", [("c", "read_file", {"path": "x"}, "value", None)])],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
    )

    assert result.unexpected_io_failure
    assert result.entries[0].no_record_reason == "I/O failure"
    assert unrelated_manifest.read_text(encoding="utf-8") == "unrelated\n"
    assert not list(session_root.glob("[0-9][0-9][0-9]_*.md"))
    assert quota.try_reserve(4_096)


def test_reconciliation_rejects_record_symlinked_within_session_root(tmp_path: Path) -> None:
    session_root = tmp_path / "session"
    session_file = session_root / "session.json"
    session_file.parent.mkdir()
    session_file.write_text("private session state", encoding="utf-8")
    record_path = session_root / dropped_turn_relative_path(1) / "001_read_file_00000000000000000000000000000001.md"
    record_path.parent.mkdir(parents=True)
    try:
        record_path.symlink_to(session_file)
    except OSError as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")
    catalog = session_root / CATALOG_RELATIVE_PATH
    catalog.write_text(
        json.dumps(
            {
                "record_id": "unsafe-internal-symlink",
                "relative_path": record_path.relative_to(session_root).as_posix(),
                "turn": 1,
                "round": 1,
                "tool": "read_file",
                "bytes": session_file.stat().st_size,
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    quota = SpillQuota()

    reconciliation = reconcile_spill_storage(session_root, quota)

    assert reconciliation.live_record_count == 0
    assert reconciliation.available_relative_paths == frozenset()
    assert catalog_live_records(session_root) == []
    assert record_path.is_symlink()
    assert session_file.read_text(encoding="utf-8") == "private session state"


def test_catalog_symlink_cannot_redirect_spill_append(tmp_path: Path) -> None:
    session_root = tmp_path / "session"
    catalog = session_root / CATALOG_RELATIVE_PATH
    catalog.parent.mkdir(parents=True)
    victim = tmp_path / "victim.txt"
    victim.write_text("keep\n", encoding="utf-8")
    try:
        catalog.symlink_to(victim)
    except OSError as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")

    result = write_spill_batch(
        session_root,
        SpillQuota(),
        [_tool_group("g", [("c", "read_file", {"path": "x"}, "value", None)])],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
    )

    assert result.unexpected_io_failure
    assert result.entries[0].no_record_reason == "I/O failure"
    assert victim.read_text(encoding="utf-8") == "keep\n"
    assert catalog.is_symlink()
    assert not list((session_root / dropped_turn_relative_path(1)).glob("*.md"))


def test_reconciliation_preserves_records_when_catalog_is_symlinked(tmp_path: Path) -> None:
    session_root = tmp_path / "session"
    record_dir = session_root / dropped_turn_relative_path(1)
    record_dir.mkdir(parents=True)
    orphan = record_dir / "001_read_file_00000000000000000000000000000001.md"
    orphan.write_text("record", encoding="utf-8")
    victim = tmp_path / "victim.txt"
    victim.write_text("external catalog", encoding="utf-8")
    catalog = session_root / CATALOG_RELATIVE_PATH
    try:
        catalog.symlink_to(victim)
    except OSError as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")
    quota = SpillQuota()

    reconciliation = reconcile_spill_storage(session_root, quota)

    assert reconciliation == spill_mod.SpillReconciliationResult(0, 0, frozenset())
    assert not quota.storage_available
    assert quota.spent_bytes == 0
    assert orphan.read_text(encoding="utf-8") == "record"
    assert victim.read_text(encoding="utf-8") == "external catalog"


def test_reconciliation_does_not_follow_symlinked_dropped_directory(tmp_path: Path) -> None:
    session_root = tmp_path / "session"
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "victim.md"
    victim.write_text("keep", encoding="utf-8")
    compactions = session_root / "compactions"
    compactions.mkdir(parents=True)
    try:
        (compactions / "dropped").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    reconciliation = reconcile_spill_storage(session_root, SpillQuota())

    assert reconciliation.deleted_orphans == 0
    assert victim.read_text(encoding="utf-8") == "keep"


def test_reconciliation_does_not_enter_nested_symlinked_sub_agent_dropped_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_root = tmp_path / "session"
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "victim.md"
    victim.write_text("keep", encoding="utf-8")
    nested_dropped = session_root / "compactions" / "sub_agents" / "explore" / "invocation" / "dropped"
    nested_dropped.parent.mkdir(parents=True)
    try:
        nested_dropped.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    original_rglob = Path.rglob

    def guarded_rglob(path: Path, pattern: str) -> Iterator[Path]:
        if path == nested_dropped:
            raise AssertionError("reconciliation entered a symlinked dropped directory")
        return original_rglob(path, pattern)

    monkeypatch.setattr(Path, "rglob", guarded_rglob)

    reconciliation = reconcile_spill_storage(session_root, SpillQuota())

    assert reconciliation.deleted_orphans == 0
    assert victim.read_text(encoding="utf-8") == "keep"


def test_manifest_rebuild_replaces_symlink_without_overwriting_target(tmp_path: Path) -> None:
    result = write_spill_batch(
        tmp_path,
        SpillQuota(),
        [_tool_group("g", [("c", "read_file", {"path": "x"}, "value", None)])],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
    )
    record_path = tmp_path / result.entries[0].relative_path
    manifest_path = record_path.with_name("manifest.md")
    manifest_path.unlink()
    victim = tmp_path / "victim.txt"
    victim.write_text("keep", encoding="utf-8")
    try:
        manifest_path.symlink_to(victim)
    except OSError as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")

    reconcile_spill_storage(tmp_path, SpillQuota())

    assert victim.read_text(encoding="utf-8") == "keep"
    assert not manifest_path.is_symlink()
    assert record_path.name in manifest_path.read_text(encoding="utf-8")


def test_reconciliation_preserves_manifest_outside_owned_spill_layouts(tmp_path: Path) -> None:
    unrelated_manifest = tmp_path / "compactions" / "unrelated" / "manifest.md"
    unrelated_manifest.parent.mkdir(parents=True)
    unrelated_manifest.write_text("unrelated compaction state\n", encoding="utf-8")

    reconcile_spill_storage(tmp_path, SpillQuota())

    assert unrelated_manifest.read_text(encoding="utf-8") == "unrelated compaction state\n"


def test_tombstone_append_failure_keeps_old_record_and_catalog_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quota = SpillQuota(limit_bytes=1_200)
    first = write_spill_batch(
        tmp_path,
        quota,
        [_tool_group("old", [("c1", "execute", {"command": "one"}, "a" * 400, None)])],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
    )
    old_path = tmp_path / first.entries[0].relative_path
    append_catalog_line = spill_mod._append_catalog_line

    def fail_tombstone(root: Path, value: dict[str, object]) -> None:
        if value.get("tombstone") is True:
            raise OSError("catalog unavailable")
        append_catalog_line(root, value)

    monkeypatch.setattr(spill_mod, "_append_catalog_line", fail_tombstone)

    second = write_spill_batch(
        tmp_path,
        quota,
        [_tool_group("new", [("c2", "execute", {"command": "two"}, "b" * 400, None)])],
        record_dir=dropped_turn_relative_path(2),
        absolute_turn=2,
        round_number=1,
        session_id="s",
    )

    assert second.unexpected_io_failure
    assert old_path.is_file()
    assert [record.record_id for record in catalog_live_records(tmp_path)] == [first.entries[0].record_id]


def test_reconciliation_deletes_orphans_uses_actual_sizes_and_rebuilds_manifest(tmp_path: Path) -> None:
    result = write_spill_batch(
        tmp_path,
        SpillQuota(),
        [_tool_group("g", [("c", "read_file", {"path": "x"}, "value", None)])],
        record_dir=dropped_turn_relative_path(3),
        absolute_turn=3,
        round_number=1,
        session_id="s",
    )
    record_path = tmp_path / result.entries[0].relative_path
    manifest_path = record_path.with_name("manifest.md")
    manifest_path.write_text("stale in-memory projection", encoding="utf-8")
    orphan = record_path.with_name("orphan.md")
    orphan.write_text("partial", encoding="utf-8")
    quota = SpillQuota()

    reconciliation = reconcile_spill_storage(tmp_path, quota)

    assert reconciliation.deleted_orphans == 1
    assert not orphan.exists()
    assert reconciliation.available_relative_paths == frozenset({result.entries[0].relative_path})
    assert reconciliation.live_total_bytes == record_path.stat().st_size
    assert quota.spent_bytes == record_path.stat().st_size
    assert quota.live_record_count() == 1
    rebuilt = manifest_path.read_text(encoding="utf-8")
    assert "stale in-memory projection" not in rebuilt
    assert record_path.name in rebuilt


def test_reconciliation_ignores_non_authoritative_manifest_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = write_spill_batch(
        tmp_path,
        SpillQuota(),
        [_assistant_group("g", "retained")],
        record_dir=dropped_turn_relative_path(1),
        absolute_turn=1,
        round_number=1,
        session_id="s",
    )
    record_path = tmp_path / result.entries[0].relative_path

    def fail_manifest_rebuild(_root: Path, **_kwargs: object) -> None:
        raise PermissionError("manifest is read-only")

    monkeypatch.setattr(
        spill_mod,
        "_rebuild_manifest_projections",
        fail_manifest_rebuild,
    )
    quota = SpillQuota()

    reconciliation = reconcile_spill_storage(tmp_path, quota)

    assert reconciliation.available_relative_paths == frozenset({result.entries[0].relative_path})
    assert reconciliation.live_total_bytes == record_path.stat().st_size
    assert quota.spent_bytes == record_path.stat().st_size


def test_shared_quota_serializes_main_and_sub_agent_catalog_writers(tmp_path: Path) -> None:
    quota = SpillQuota()
    barrier = Barrier(3)

    def write(group_id: str, record_dir: Path) -> None:
        barrier.wait()
        result = write_spill_batch(
            tmp_path,
            quota,
            [_tool_group(group_id, [(group_id, "read_file", {"path": group_id}, group_id, None)])],
            record_dir=record_dir,
            absolute_turn=1,
            round_number=1,
            session_id="s",
        )
        assert result.entries[0].relative_path

    with ThreadPoolExecutor(max_workers=2) as executor:
        main = executor.submit(write, "main", dropped_turn_relative_path(1))
        sub = executor.submit(write, "sub", sub_agent_dropped_turn_relative_path("Explore", "inv-1", 1))
        barrier.wait()
        main.result()
        sub.result()

    live = catalog_live_records(tmp_path)
    assert len(live) == 2
    assert len({record.record_id for record in live}) == 2
    assert quota.spent_bytes == sum((tmp_path / record.relative_path).stat().st_size for record in live)

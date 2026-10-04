# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for JsonFileStateStore.fork_session — copying, identity rewriting, collision retries, and cleanup."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from uuid import UUID

import pytest

import chrys.service.state._fork as fork_module
from chrys.foundation.models.history_markers import ANTHROPIC_THINKING_STRIPPED_KEY
from chrys.foundation.platform.files import atomic_write_owner_only_text
from chrys.foundation.text.mentions import format_file_mention
from chrys.foundation.util.session_ids import session_short_id
from chrys.kernel import Content, Message
from chrys.service.context.providers.history import CompressedBlock
from chrys.service.state.serializers import (
    serialize_message,
)
from chrys.service.state.store import (
    RAW_HTTP_LOG_FILE_NAME,
    SESSION_RECOVERY_FILE_NAME,
    JsonFileStateStore,
    SessionForkError,
)
from chrys.service.trajectory.state import TRAJECTORY_STATE_KEY


async def test_fork_copies_document_artifacts_and_resolves_handles_in_fork(tmp_path: Path) -> None:
    from PIL import Image

    from chrys.foundation.models.session_env import SessionEnvironment
    from chrys.foundation.platform.files import secure_open_owner_only_binary
    from chrys.service.tools.builtins.doc_converter import _write_unique_markdown
    from chrys.service.tools.builtins.doc_converter.artifacts import DocumentImageSink
    from chrys.service.tools.builtins.filesystem import FilesystemTools
    from chrys.service.tools.session_artifacts import (
        resolve_document_image_artifact_handle,
        resolve_document_markdown_artifact_handle,
    )

    store = JsonFileStateStore(tmp_path)
    parent_id = "parent-document-image"
    await store.save_session(parent_id, {"messages": [], "compressed_msgs": []})
    parent_dir = store.session_dir(parent_id)
    image_bytes = BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(image_bytes, format="PNG")
    sink = DocumentImageSink(parent_dir / "doc_converter", source_stem="report")
    assert sink.try_reserve_occurrence()
    occurrence = sink.save_image(image_bytes.getvalue(), location="Page 1", ordinal=1, source_name="page.png")
    assert occurrence is not None
    sink.commit_occurrences((occurrence,))
    handle = occurrence.reference
    resolved_parent_image = resolve_document_image_artifact_handle(handle, parent_dir)
    assert resolved_parent_image is not None
    parent_image = Path(resolved_parent_image)
    parent_absolute_path = str(parent_image)
    markdown_artifact = _write_unique_markdown(str(parent_dir / "doc_converter"), "report", "# Report")
    markdown_handle = markdown_artifact.handle
    resolved_parent_markdown = resolve_document_markdown_artifact_handle(markdown_handle, parent_dir)
    assert resolved_parent_markdown is not None
    parent_markdown = Path(resolved_parent_markdown)

    fork_id = store.fork_session(parent_id)
    fork_dir = store.session_dir(fork_id)
    resolved_fork_image = resolve_document_image_artifact_handle(handle, fork_dir)
    assert resolved_fork_image is not None
    fork_image = Path(resolved_fork_image)
    resolved_fork_markdown = resolve_document_markdown_artifact_handle(markdown_handle, fork_dir)
    assert resolved_fork_markdown is not None
    fork_markdown = Path(resolved_fork_markdown)

    assert fork_image.read_bytes() == parent_image.read_bytes()
    with secure_open_owner_only_binary(fork_image) as copied_image:
        assert copied_image.read() == image_bytes.getvalue()
    reused_sink = DocumentImageSink(fork_dir / "doc_converter", source_stem="reused")
    assert reused_sink.try_reserve_occurrence()
    reused = reused_sink.save_image(
        image_bytes.getvalue(),
        location="Page 1",
        ordinal=1,
        source_name="reused.png",
    )
    assert reused is not None
    assert reused.reference == handle
    reused_sink.commit_occurrences((reused,))
    assert fork_markdown.read_text(encoding="utf-8") == "# Report"
    runtime = SessionEnvironment.capture()
    filesystem = FilesystemTools(runtime, session_dir=fork_dir)
    viewed = filesystem.view_image(handle)
    assert viewed[0].media_type == "image/png"
    assert viewed[0].additional_properties["source_path"] == str(fork_image)
    assert "1|# Report" in filesystem.read_file(markdown_handle)

    parent_image.unlink()
    parent_markdown.unlink()
    assert not Path(parent_absolute_path).exists()
    assert filesystem.view_image(handle)[0].media_type == "image/png"
    assert "1|# Report" in filesystem.read_file(markdown_handle)


async def test_fork_session_copies_and_rewrites_session_identity(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    parent_id = "parent-session-id"
    parent_dir = store.session_dir(parent_id)
    workspace_file = tmp_path / "workspace" / "file.py"
    workspace_file.parent.mkdir()
    workspace_file.write_text("print('shared workspace')", encoding="utf-8")
    clipboard_dir = parent_dir / "attachments" / "clipboard"
    clipboard_dir.mkdir(parents=True)
    # A double-quote exercises mention quote-escaping but is an illegal path
    # character on Windows; only include it where the filesystem allows it.
    # (The escape round-trip itself is covered platform-independently by the
    # string-only mention tests in tests/orchestration/engine/run/test_attachments.py.)
    clipboard_image = clipboard_dir / ("screen one.png" if sys.platform == "win32" else 'screen "one".png')
    clipboard_image.write_bytes(b"png")
    parent_mention = format_file_mention(clipboard_image)
    second_clipboard_image = clipboard_dir / "screen two.png"
    second_clipboard_image.write_bytes(b"png2")
    second_parent_mention = format_file_mention(second_clipboard_image)
    outside_image = tmp_path / "outside.png"
    outside_image.write_bytes(b"outside")
    outside_mention = format_file_mention(outside_image)
    data_bytes = b"\x00\x01"
    state = {
        "messages": [
            Message(
                "user",
                [
                    f"look at {parent_mention} then {second_parent_mention}; leave {outside_mention}",
                    Content.from_data(data=data_bytes, media_type="image/png"),
                ],
            ),
            Message("assistant", [f"the original path was {parent_mention}"]),
        ],
        "compressed_msgs": [
            CompressedBlock(
                compressed_context_id="ctx_clipboard",
                messages=[
                    Message("user", [f"compressed {parent_mention} and {second_parent_mention}"]),
                    Message("assistant", [f"compressed assistant keeps {parent_mention}"]),
                ],
                summary_text="compressed clipboard mentions",
                marker_id="turn_1",
                turn_range=(1, 1),
                created_at="2026-03-17T00:00:00+00:00",
            )
        ],
        "chrys_mutations": {
            "turns": [
                {
                    "turn_id": 1,
                    "mutations": [
                        {
                            "path": str(workspace_file),
                            "operation": "modify",
                            "source": "edit_file",
                            "tool_call_id": "call_1",
                            "timestamp": 1.0,
                            "before_hash": "old",
                            "after_hash": "new",
                        }
                    ],
                }
            ],
        },
    }
    await store.save_session(
        parent_id,
        state,
        agent_profile="Code",
        primary_cwd=str(workspace_file.parent),
        working_dirs=[str(workspace_file.parent)],
        service_session_id="provider-session",
    )
    (parent_dir / "mutations").mkdir()
    (parent_dir / "mutations" / "marker.txt").write_text("mutation data", encoding="utf-8")
    (parent_dir / SESSION_RECOVERY_FILE_NAME).write_text("{}", encoding="utf-8")
    (parent_dir / RAW_HTTP_LOG_FILE_NAME).write_text(
        json.dumps({"session_id": parent_id}) + "\n",
        encoding="utf-8",
    )
    snapshots = parent_dir / "snapshots"
    snapshots.mkdir()
    (snapshots / "turn_1.json").write_bytes((parent_dir / "session.json").read_bytes())
    (snapshots / "not_a_turn.json").write_text(
        json.dumps({"meta": {"session_id": parent_id}}),
        encoding="utf-8",
    )
    sub_agents = parent_dir / "sub_agents"
    (sub_agents / "pending").mkdir(parents=True)
    (sub_agents / "legacy.json").write_text(json.dumps({"invocation_id": "legacy"}), encoding="utf-8")
    (sub_agents / "pending" / "active.json").write_text(json.dumps({"invocation_id": "active"}), encoding="utf-8")
    (sub_agents / "pending" / "active.json.tmp").write_text("partial", encoding="utf-8")
    (sub_agents / "sessions").mkdir(parents=True)
    (sub_agents / "sessions" / ".Explore_a1b2c3d4e5f6.json.tmp").write_text("partial", encoding="utf-8")
    atomic_write_owner_only_text(
        sub_agents / "sessions" / "Explore_a1b2c3d4e5f6.json",
        json.dumps(
            {
                "meta": {
                    "record_type": "sub_agent_session",
                    "parent_session_id": parent_id,
                    "invocation_id": "a1b2c3d4e5f6",
                    "status": "completed",
                },
                "state": {
                    "messages": [
                        serialize_message(Message("user", [f"sub-agent saw {parent_mention}"])),
                    ]
                },
            }
        ),
    )
    parent_before = {
        path.relative_to(parent_dir): path.read_bytes() for path in parent_dir.rglob("*") if path.is_file()
    }

    fork_id = store.fork_session(parent_id)

    fork_dir = store.session_dir(fork_id)
    assert fork_id != parent_id
    assert fork_dir == tmp_path / session_short_id(fork_id)
    assert fork_dir.is_dir()
    parent_after = {path.relative_to(parent_dir): path.read_bytes() for path in parent_dir.rglob("*") if path.is_file()}
    assert parent_after == parent_before

    parent_envelope = json.loads((parent_dir / "session.json").read_text(encoding="utf-8"))
    fork_envelope = json.loads((fork_dir / "session.json").read_text(encoding="utf-8"))
    parent_meta = parent_envelope["meta"]
    fork_meta = fork_envelope["meta"]
    assert fork_meta["session_id"] == fork_id
    assert fork_meta["parent_session_id"] == parent_id
    assert fork_meta["created_at"] == parent_meta["created_at"]
    assert datetime.fromisoformat(fork_meta["updated_at"]) >= datetime.fromisoformat(parent_meta["updated_at"])
    assert fork_meta["service_session_id"] == ""
    assert fork_meta["primary_cwd"] == str(workspace_file.parent)
    assert fork_meta["working_dirs"] == [str(workspace_file.parent)]

    fork_backup = json.loads((fork_dir / "session.json.bak").read_text(encoding="utf-8"))
    fork_snapshot = json.loads((fork_dir / "snapshots" / "turn_1.json").read_text(encoding="utf-8"))
    assert fork_backup["meta"]["session_id"] == fork_id
    assert fork_snapshot["meta"]["session_id"] == fork_id
    assert fork_snapshot["meta"]["parent_session_id"] == parent_id
    assert (
        json.loads((fork_dir / "snapshots" / "not_a_turn.json").read_text(encoding="utf-8"))["meta"]["session_id"]
        == parent_id
    )
    assert not (fork_dir / SESSION_RECOVERY_FILE_NAME).exists()
    assert not (fork_dir / RAW_HTTP_LOG_FILE_NAME).exists()
    assert not (fork_dir / "sub_agents" / "legacy.json").exists()
    assert not (fork_dir / "sub_agents" / "pending").exists()
    assert not list((fork_dir / "sub_agents").rglob("*.tmp"))
    fork_sub_agent_log = json.loads(
        (fork_dir / "sub_agents" / "sessions" / "Explore_a1b2c3d4e5f6.json").read_text(encoding="utf-8")
    )
    assert fork_sub_agent_log["meta"]["parent_session_id"] == fork_id
    fork_sub_agent_text = fork_sub_agent_log["state"]["messages"][0]["contents"][0]["text"]
    assert format_file_mention(fork_dir / "attachments" / "clipboard" / clipboard_image.name) in fork_sub_agent_text
    assert parent_mention not in fork_sub_agent_text
    assert (fork_dir / "mutations" / "marker.txt").read_text(encoding="utf-8") == "mutation data"
    assert (fork_dir / "attachments" / "clipboard" / clipboard_image.name).read_bytes() == b"png"
    assert (fork_dir / "attachments" / "clipboard" / second_clipboard_image.name).read_bytes() == b"png2"

    fork_clipboard_image = fork_dir / "attachments" / "clipboard" / clipboard_image.name
    fork_second_clipboard_image = fork_dir / "attachments" / "clipboard" / second_clipboard_image.name
    user_text = fork_envelope["state"]["messages"][0]["contents"][0]["text"]
    assistant_text = fork_envelope["state"]["messages"][1]["contents"][0]["text"]
    assert format_file_mention(fork_clipboard_image) in user_text
    assert format_file_mention(fork_second_clipboard_image) in user_text
    assert outside_mention in user_text
    assert parent_mention not in user_text
    assert second_parent_mention not in user_text
    assert str(parent_dir) not in user_text
    assert parent_mention in assistant_text
    compressed_user_text = fork_envelope["state"]["compressed_msgs"][0]["messages"][0]["contents"][0]["text"]
    compressed_assistant_text = fork_envelope["state"]["compressed_msgs"][0]["messages"][1]["contents"][0]["text"]
    assert format_file_mention(fork_clipboard_image) in compressed_user_text
    assert format_file_mention(fork_second_clipboard_image) in compressed_user_text
    assert parent_mention not in compressed_user_text
    assert second_parent_mention not in compressed_user_text
    assert parent_mention in compressed_assistant_text
    assert (
        fork_envelope["state"]["messages"][0]["contents"][1]["uri"]
        == parent_envelope["state"]["messages"][0]["contents"][1]["uri"]
    )
    assert fork_envelope["state"]["chrys_mutations"]["turns"][0]["mutations"][0]["path"] == str(workspace_file)


async def test_fork_drops_the_turn_registry_that_names_the_parents_log(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    registry = {"turns": {"1": {"turn_id": "a" * 32, "started_sequence": 42}}}
    await store.save_session(
        "parent",
        {"messages": [Message("user", ["hi"])], "compressed_msgs": [], TRAJECTORY_STATE_KEY: dict(registry)},
    )
    parent_dir = store.session_dir("parent")
    snapshots = parent_dir / "snapshots"
    snapshots.mkdir()
    (snapshots / "turn_1.json").write_bytes((parent_dir / "session.json").read_bytes())

    fork_id = store.fork_session("parent")

    fork_dir = store.session_dir(fork_id)
    for name in ("session.json", "snapshots/turn_1.json"):
        envelope = json.loads((fork_dir / name).read_text(encoding="utf-8"))
        # The fork numbers its own log from one, so a sequence copied from the
        # parent would make a rollback here name a range that never existed.
        assert TRAJECTORY_STATE_KEY not in envelope["state"], name
    parent_envelope = json.loads((parent_dir / "session.json").read_text(encoding="utf-8"))
    assert parent_envelope["state"][TRAJECTORY_STATE_KEY] == registry


async def test_refused_thinking_stays_marked_through_save_load_and_fork(tmp_path: Path) -> None:
    """Thinking the service refused is never replayed, also after a restart or in a fork."""

    def refused(text: str) -> Content:
        content = Content.from_text_reasoning(text=text, protected_data=f"sig-{text}")
        content.additional_properties[ANTHROPIC_THINKING_STRIPPED_KEY] = True
        return content

    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "parent",
        {
            "messages": [Message("user", ["hi"]), Message("assistant", [refused("live"), Content.from_text("ok")])],
            "compressed_msgs": [
                CompressedBlock(
                    compressed_context_id="ctx_refused",
                    messages=[Message("assistant", [refused("archived"), Content.from_text("earlier")])],
                    summary_text="summary",
                    marker_id="turn_1",
                    turn_range=(1, 1),
                    created_at="2026-03-17T00:00:00+00:00",
                )
            ],
        },
    )
    fork_id = store.fork_session("parent")

    for session_id in ("parent", fork_id):
        state = await store.load_session(session_id)
        assert state is not None
        [live] = state["messages"][1].contents[:1]
        [archived] = state["compressed_msgs"][0].messages[0].contents[:1]
        assert (live.text, archived.text) == ("live", "archived")
        assert live.additional_properties[ANTHROPIC_THINKING_STRIPPED_KEY] is True, session_id
        assert archived.additional_properties[ANTHROPIC_THINKING_STRIPPED_KEY] is True, session_id


async def test_fork_session_retries_short_id_collisions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session("parent", {"messages": [Message("user", ["hi"])], "compressed_msgs": []})
    first = UUID("11111111-1111-4111-8111-111111111111")
    second = UUID("22222222-2222-4222-8222-222222222222")
    temp = UUID("33333333-3333-4333-8333-333333333333")
    store.session_dir(str(first)).mkdir()
    values = iter([first, second, temp])
    monkeypatch.setattr(fork_module, "uuid4", lambda: next(values))

    fork_id = store.fork_session("parent")

    assert fork_id == str(second)
    assert store.session_dir(fork_id).is_dir()


async def test_fork_session_skips_invalid_and_malformed_auxiliary_envelopes(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    parent_id = "parent"
    await store.save_session(parent_id, {"messages": [Message("user", ["hi"])], "compressed_msgs": []})
    parent_dir = store.session_dir(parent_id)
    (parent_dir / "session.json.bak").write_text(
        json.dumps({"state": {"messages": [], "compressed_msgs": []}}),
        encoding="utf-8",
    )
    snapshots = parent_dir / "snapshots"
    snapshots.mkdir()
    (snapshots / "turn_1.json").write_text("{ broken snapshot", encoding="utf-8")
    (snapshots / "turn_2.json").write_bytes((parent_dir / "session.json").read_bytes())
    (snapshots / "turn_3.json").write_text(
        json.dumps({"state": {"messages": [], "compressed_msgs": []}}),
        encoding="utf-8",
    )

    fork_id = store.fork_session(parent_id)

    fork_dir = store.session_dir(fork_id)
    fork_primary = json.loads((fork_dir / "session.json").read_text(encoding="utf-8"))
    fork_backup = json.loads((fork_dir / "session.json.bak").read_text(encoding="utf-8"))
    fork_snapshot = json.loads((fork_dir / "snapshots" / "turn_2.json").read_text(encoding="utf-8"))
    assert fork_primary["meta"]["session_id"] == fork_id
    assert fork_backup["meta"]["session_id"] == fork_id
    assert fork_backup["state"]["messages"][0]["contents"][0]["text"] == "hi"
    assert not (fork_dir / "snapshots" / "turn_1.json").exists()
    assert not (fork_dir / "snapshots" / "turn_3.json").exists()
    assert fork_snapshot["meta"]["session_id"] == fork_id
    assert fork_snapshot["meta"]["parent_session_id"] == parent_id


async def test_fork_session_retries_destination_recheck_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session("parent", {"messages": [Message("user", ["hi"])], "compressed_msgs": []})
    first = UUID("11111111-1111-4111-8111-111111111111")
    second = UUID("22222222-2222-4222-8222-222222222222")
    temp = UUID("33333333-3333-4333-8333-333333333333")
    values = iter([first, second, temp])
    monkeypatch.setattr(fork_module, "uuid4", lambda: next(values))
    real_assert_destination_available = store._assert_fork_destination_available
    collided = False

    def flaky_assert_destination_available(session_id: str, *, include_write_lock: bool = True) -> None:
        nonlocal collided
        if session_id == str(first) and not include_write_lock and not collided:
            collided = True
            store.session_dir(session_id).mkdir()
            raise SessionForkError("Fork destination collision at simulated target")
        real_assert_destination_available(session_id, include_write_lock=include_write_lock)

    monkeypatch.setattr(store, "_assert_fork_destination_available", flaky_assert_destination_available)

    fork_id = store.fork_session("parent")

    assert collided is True
    assert fork_id == str(second)
    assert store.session_dir(fork_id).is_dir()


async def test_fork_session_stops_after_collision_retry_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session("parent", {"messages": [Message("user", ["hi"])], "compressed_msgs": []})
    colliding = UUID("11111111-1111-4111-8111-111111111111")
    store.session_dir(str(colliding)).mkdir()
    monkeypatch.setattr(fork_module, "uuid4", lambda: colliding)

    with pytest.raises(SessionForkError):
        store.fork_session("parent")


async def test_fork_session_failure_cleans_temp_dir_and_preserves_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session("parent", {"messages": [Message("user", ["hi"])], "compressed_msgs": []})
    parent_dir = store.session_dir("parent")
    parent_before = {
        path.relative_to(parent_dir): path.read_bytes() for path in parent_dir.rglob("*") if path.is_file()
    }

    def fail_rewrite(*args: object, **kwargs: object) -> None:
        raise RuntimeError("rewrite failed")

    monkeypatch.setattr(JsonFileStateStore, "_rewrite_fork_envelope_file", fail_rewrite)

    with pytest.raises(SessionForkError, match="Failed to fork session"):
        store.fork_session("parent")

    parent_after = {path.relative_to(parent_dir): path.read_bytes() for path in parent_dir.rglob("*") if path.is_file()}
    assert parent_after == parent_before
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".") and path.name != ".locks"]

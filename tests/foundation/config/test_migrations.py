# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the three-phase ``dotenv_v0`` migration."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any
from unittest.mock import create_autospec

import pytest
import yaml

import chrys.foundation.config.migrations as migrations
import chrys.foundation.platform as platform_mod
from chrys.foundation.config.coercion import CoerceReason
from chrys.foundation.config.env_layers import _reset_process_env_snapshot_for_tests, freeze_process_env
from chrys.foundation.config.migrations import (
    DOTENV_COMPONENT,
    NOTIFICATIONS_COMPONENT,
    migrate_dotenv_v0,
    migrate_notifications_v0,
    migrate_renamed_theme,
)
from chrys.foundation.config.settings import persist_theme
from chrys.foundation.config.settings_store import load_settings
from chrys.foundation.config.spec import Source
from chrys.foundation.config.user_settings import user_settings_path


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    fake = dataclasses.replace(
        platform_mod.get_platform(),
        config_dir=tmp_path / "config",
        data_dir=tmp_path / "data",
    )
    monkeypatch.setattr(platform_mod, "get_platform", lambda: fake)
    return fake.config_dir


def _write_env(config_dir: Path, text: str) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / ".env"
    path.write_text(text, encoding="utf-8")
    return path


def _read_doc(config_dir: Path) -> dict:
    return yaml.safe_load((config_dir / "settings.yaml").read_text(encoding="utf-8"))


def test_migration_moves_chrys_keys_and_leaves_other_lines(config_dir: Path) -> None:
    env_path = _write_env(
        config_dir,
        "MY_TOKEN=secret\nCHRYS_THEME=solar\nCHRYS_MODEL_PROFILE=gpt\nCHRYS_SESSION_TITLE_AUTO=0\n",
    )

    result = migrate_dotenv_v0()

    assert result.performed
    assert result.migrated == {
        "ui.theme": "solar",
        "model.profile.active": "gpt",
        "session.title.auto": False,
    }
    assert result.warnings == ()

    doc = _read_doc(config_dir)
    assert doc["ui"]["theme"] == "solar"
    assert doc["model"]["profile"]["active"] == "gpt"
    assert doc["session"]["title"]["auto"] is False
    assert isinstance(doc["migrations"][DOTENV_COMPONENT], str)

    remaining = env_path.read_text(encoding="utf-8")
    assert "MY_TOKEN=secret" in remaining
    assert "CHRYS_" not in remaining


def test_migrated_values_load_as_the_user_layer(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHRYS_THEME", raising=False)
    freeze_process_env()
    _write_env(config_dir, "CHRYS_THEME=solar\n")

    migrate_dotenv_v0()
    loaded = load_settings()

    assert loaded.settings.theme == "solar"
    assert loaded.source_for("ui.theme").layer is Source.USER


def test_migration_reads_the_alias_grammar_and_writes_the_key_grammar(config_dir: Path) -> None:
    """``CHRYS_HISTORY_DISABLE=1`` means *disable*; a one-step read flips it."""
    _write_env(config_dir, "CHRYS_HISTORY_DISABLE=1\n")

    result = migrate_dotenv_v0()

    assert result.migrated == {"history.prompt.enabled": False}
    assert _read_doc(config_dir)["history"]["prompt"]["enabled"] is False


def test_an_invalid_value_is_dropped_with_a_warning_and_still_cleaned(config_dir: Path) -> None:
    env_path = _write_env(config_dir, "CHRYS_SESSION_TITLE_AUTO=nonsense\n")

    result = migrate_dotenv_v0()

    assert result.migrated == {}
    (warning,) = result.warnings
    assert warning.key == "session.title.auto"
    assert warning.origin.layer is Source.USER_ENV
    assert warning.origin.path == env_path
    assert warning.outcome.reason is CoerceReason.EXPECTED_BOOL
    assert "CHRYS_" not in env_path.read_text(encoding="utf-8")
    assert "session" not in _read_doc(config_dir)


def test_a_blank_value_is_left_in_place_without_migrating_or_warning(config_dir: Path) -> None:
    """Cleanup deletes what was imported from; a blank line gave nothing.

    Harmless to leave: *this key's* coercer folds a blank into "says
    nothing", so the line cannot shadow the document the migration just
    wrote. Whether a blank speaks is per-key — see the disable-alias test
    below for the other kind.
    """
    env_path = _write_env(config_dir, "CHRYS_MODEL_PROFILE=\n")

    result = migrate_dotenv_v0()

    assert result.migrated == {}
    assert result.warnings == ()
    assert env_path.read_text(encoding="utf-8") == "CHRYS_MODEL_PROFILE=\n"


def test_a_blank_the_coercer_hears_migrates_and_is_retired(config_dir: Path) -> None:
    """``CHRYS_HISTORY_DISABLE=`` is not silence: it reads as "not 1" — enabled.

    The line therefore migrates like any other value, and phase B must retire
    it: left behind, it would sit in the higher USER_ENV layer and forever
    override the very document it was just folded into.
    """
    env_path = _write_env(config_dir, "CHRYS_HISTORY_DISABLE=\n")

    result = migrate_dotenv_v0()

    assert result.performed
    assert result.migrated == {"history.prompt.enabled": True}
    assert result.warnings == ()
    assert _read_doc(config_dir)["history"]["prompt"]["enabled"] is True
    assert "CHRYS_" not in env_path.read_text(encoding="utf-8")


def test_a_leftover_speaking_blank_reimports_after_completion(config_dir: Path) -> None:
    """Self-heal: a completed ledger plus a speaking blank line is reappearance.

    A file that speaks again after completion is the newer state, and that
    holds when what it says is an empty string the key's coercer hears.
    """
    _write_env(config_dir, "CHRYS_THEME=solar\n")
    first = migrate_dotenv_v0()
    assert first.performed
    env_path = _write_env(config_dir, "CHRYS_HISTORY_DISABLE=\n")

    second = migrate_dotenv_v0()

    assert second.performed
    assert second.migrated == {"history.prompt.enabled": True}
    assert _read_doc(config_dir)["history"]["prompt"]["enabled"] is True
    assert "CHRYS_" not in env_path.read_text(encoding="utf-8")


def test_a_second_run_is_a_no_op(config_dir: Path) -> None:
    _write_env(config_dir, "CHRYS_THEME=solar\n")
    first = migrate_dotenv_v0()
    assert first.performed
    stamp = _read_doc(config_dir)["migrations"][DOTENV_COMPONENT]

    second = migrate_dotenv_v0()

    assert not second.performed
    assert second.migrated == {}
    assert _read_doc(config_dir)["migrations"][DOTENV_COMPONENT] == stamp


def test_a_pending_rerun_finishes_cleanup_without_rolling_back_newer_edits(config_dir: Path) -> None:
    """Crash between phases: the document's value is newer and must survive."""
    env_path = _write_env(config_dir, "CHRYS_THEME=old\n")
    config_dir.joinpath("settings.yaml").write_text(
        f"schema_version: 1\nui:\n  theme: newer\nmigrations:\n  {DOTENV_COMPONENT}:\n    pending_cleanup: true\n",
        encoding="utf-8",
    )

    result = migrate_dotenv_v0()

    assert result.performed
    assert result.migrated == {}
    doc = _read_doc(config_dir)
    assert doc["ui"]["theme"] == "newer"
    assert isinstance(doc["migrations"][DOTENV_COMPONENT], str)
    assert "CHRYS_" not in env_path.read_text(encoding="utf-8")


def test_a_legacy_value_the_document_outranks_is_named_in_the_log(
    config_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Phase B deletes the line, so this is the last moment the two disagree.

    The outcome is correct — the document is newer — but silence here means a
    user whose old value quietly stopped applying has nothing to look at.
    """
    # A value no incidental word could contain, so "the log does not name it"
    # cannot pass by accident and no quoting or punctuation around a leak can
    # hide it either.
    _write_env(config_dir, "CHRYS_THEME=zqx-legacy-value\n")
    config_dir.joinpath("settings.yaml").write_text(
        f"schema_version: 1\nui:\n  theme: newer\nmigrations:\n  {DOTENV_COMPONENT}:\n    pending_cleanup: true\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.INFO, logger="chrys.foundation.config.migrations"):
        migrate_dotenv_v0()

    assert any("ui.theme" in message and "CHRYS_THEME" in message for message in caplog.messages)
    # The value the user would have to search for stays out of the log.
    assert not any("zqx-legacy-value" in message for message in caplog.messages)


def test_a_fresh_install_records_completion_without_growing_an_env_file(config_dir: Path) -> None:
    result = migrate_dotenv_v0()

    assert result.performed
    assert result.migrated == {}
    assert isinstance(_read_doc(config_dir)["migrations"][DOTENV_COMPONENT], str)
    assert not (config_dir / ".env").exists()


def test_migration_folds_spellings_the_way_the_os_does(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for os_name, migrated, line_removed in (("windows", {"ui.theme": "folded"}, True), ("linux", {}, False)):
        fake = dataclasses.replace(
            platform_mod.get_platform(),
            os_name=os_name,
            config_dir=tmp_path / os_name / "config",
            data_dir=tmp_path / os_name / "data",
        )
        monkeypatch.setattr(platform_mod, "get_platform", lambda fake=fake: fake)
        env_path = _write_env(fake.config_dir, "chrys_theme=folded\n")

        result = migrate_dotenv_v0()

        assert result.migrated == migrated, os_name
        assert ("chrys_theme" not in env_path.read_text(encoding="utf-8")) is line_removed, os_name


def test_dotenv_cleanup_defers_when_the_file_moved_since_it_was_read(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase B only retires the exact bytes phase A read."""
    # The removal list is chosen from that read, so a key written in between is
    # one the document will never carry — and an unqualified removal deletes it
    # anyway. Deferring costs one idempotent retry; not deferring costs the
    # value.
    env_path = _write_env(config_dir, "CHRYS_THEME=solar\n")
    real_update = migrations.update_env_file
    raced: list[bool] = []

    def write_then_clean(*args: Any, **kwargs: Any) -> bool:
        if not raced:
            raced.append(True)
            env_path.write_text("CHRYS_THEME=solar\nCHRYS_LOCALE=zh-Hans\n", encoding="utf-8")
        return real_update(*args, **kwargs)

    monkeypatch.setattr(migrations, "update_env_file", write_then_clean)

    first = migrate_dotenv_v0()

    assert first.migrated == {"ui.theme": "solar"}
    assert env_path.read_text(encoding="utf-8") == "CHRYS_THEME=solar\nCHRYS_LOCALE=zh-Hans\n"
    assert _read_doc(config_dir)["migrations"][DOTENV_COMPONENT]["pending_cleanup"] is True

    second = migrate_dotenv_v0()

    # The next start folds in what the racing write left, then retires the lot.
    # Both keys, not just the new one: the file no longer digests to what phase
    # A imported from, so the document's copy has lost its claim to being newer.
    assert second.migrated == {"ui.theme": "solar", "ui.locale": "zh-Hans"}
    assert "CHRYS_" not in env_path.read_text(encoding="utf-8")
    assert isinstance(_read_doc(config_dir)["migrations"][DOTENV_COMPONENT], str)


def test_a_deferred_cleanup_still_imports_a_value_the_source_changed_since(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pending is not a licence to skip the key forever — only to prefer the document."""
    # "The document is newer" holds because we imported the file and nothing has
    # written it since. Once something has, the file is the newer of the two —
    # and it is also the layer that still outranks the document at load time, so
    # skipping it would leave the user looking at ``midnight`` with the document
    # saying ``solar``, right up until phase B deletes the line and the theme
    # changes under them.
    env_path = _write_env(config_dir, "CHRYS_THEME=solar\n")
    real_update = migrations.update_env_file
    raced: list[bool] = []

    def rewrite_then_clean(*args: Any, **kwargs: Any) -> bool:
        if not raced:
            raced.append(True)
            env_path.write_text("CHRYS_THEME=midnight\n", encoding="utf-8")
        return real_update(*args, **kwargs)

    monkeypatch.setattr(migrations, "update_env_file", rewrite_then_clean)

    assert migrate_dotenv_v0().migrated == {"ui.theme": "solar"}
    ledger = _read_doc(config_dir)["migrations"][DOTENV_COMPONENT]
    assert ledger["pending_cleanup"] is True

    second = migrate_dotenv_v0()

    assert second.migrated == {"ui.theme": "midnight"}
    assert _read_doc(config_dir)["ui"]["theme"] == "midnight"
    assert "CHRYS_" not in env_path.read_text(encoding="utf-8")


def test_a_dotenv_edit_that_is_undone_cannot_slip_past_the_digest(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The digest has to cover the bytes that were *parsed*, not a second read of them."""
    # The lock serialises Chrys writers; nothing serialises a text editor. Two
    # reads are two files as far as it is concerned: digest ``solar``, parse
    # ``midnight``, and by phase B the editor has undone its change — the
    # recheck passes on bytes nothing was imported from, and the migration
    # writes ``midnight`` into the document while deleting the line that says
    # ``solar``, with nothing left anywhere that remembers it.
    env_path = _write_env(config_dir, "CHRYS_THEME=solar\n")
    real_read_bytes = Path.read_bytes
    reads: list[bool] = []

    def edit_after_each_read(self: Path) -> bytes:
        data = real_read_bytes(self)
        if self == env_path:
            reads.append(True)
            if len(reads) == 1:
                env_path.write_text("CHRYS_THEME=midnight\n", encoding="utf-8")
            elif len(reads) == 2:
                env_path.write_text("CHRYS_THEME=solar\n", encoding="utf-8")
        return data

    monkeypatch.setattr(Path, "read_bytes", edit_after_each_read)

    result = migrate_dotenv_v0()

    # One read, so there is no "in between": what was imported is what the
    # digest describes, and phase B refuses because what it finds is not that.
    assert result.migrated == {"ui.theme": "solar"}
    assert env_path.read_text(encoding="utf-8") == "CHRYS_THEME=solar\n"
    assert _read_doc(config_dir)["migrations"][DOTENV_COMPONENT]["pending_cleanup"] is True

    # And it still converges once the editor stops.
    migrate_dotenv_v0()
    assert _read_doc(config_dir)["ui"]["theme"] == "solar"
    assert "CHRYS_" not in env_path.read_text(encoding="utf-8")


def test_an_unreadable_dotenv_is_left_untouched_instead_of_emptied(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed read must not reach cleanup wearing the shape of an empty file."""
    # Cleanup removes the whole key space rather than the keys that were found,
    # so a parse that fails and reports nothing looks exactly like a file with
    # nothing of ours in it. The digest cannot tell them apart either — the
    # bytes never changed, so it still matches — and every setting in the file
    # is deleted having been imported nowhere.
    env_path = _write_env(config_dir, "MY_TOKEN=secret\nCHRYS_THEME=solar\n")
    real_read_bytes = Path.read_bytes
    failed: list[bool] = []

    def unreadable_once(self: Path) -> bytes:
        # Transient on purpose: a failure that persists would also fail the
        # cleanup's own read, so only a read that recovers can reach the
        # destructive step with nothing imported.
        if self == env_path and not failed:
            failed.append(True)
            raise OSError(5, "Input/output error")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", unreadable_once)

    result = migrate_dotenv_v0()

    assert not result.performed
    assert result.migrated == {}
    assert env_path.read_text(encoding="utf-8") == "MY_TOKEN=secret\nCHRYS_THEME=solar\n"
    # Untouched ledger, so a start that can read the file still migrates it.
    assert not (config_dir / "settings.yaml").exists()

    assert migrate_dotenv_v0().migrated == {"ui.theme": "solar"}
    assert env_path.read_text(encoding="utf-8") == "MY_TOKEN=secret\n"


def test_a_dotenv_key_written_after_completion_wins_over_the_document(config_dir: Path) -> None:
    """Past completion the old source is the newer intent, and reopens the work."""
    _write_env(config_dir, "CHRYS_THEME=solar\n")
    assert migrate_dotenv_v0().migrated == {"ui.theme": "solar"}

    # Only a downgraded install writes this format now, and it cannot see the
    # document. Importing it under the document-wins rule would delete
    # ``midnight`` and keep ``solar``; ignoring it would leave the line
    # outranking every panel write for good.
    env_path = _write_env(config_dir, "CHRYS_THEME=midnight\n")
    result = migrate_dotenv_v0()

    assert result.performed
    assert result.migrated == {"ui.theme": "midnight"}
    assert _read_doc(config_dir)["ui"]["theme"] == "midnight"
    assert "CHRYS_" not in env_path.read_text(encoding="utf-8")
    assert isinstance(_read_doc(config_dir)["migrations"][DOTENV_COMPONENT], str)


def test_a_completed_dotenv_migration_ignores_keys_it_never_owned(config_dir: Path) -> None:
    """``.env`` outlives the migration for SDK variables; presence is not return."""
    env_path = _write_env(config_dir, "MY_TOKEN=secret\nCHRYS_THEME=solar\n")
    migrate_dotenv_v0()
    stamp = _read_doc(config_dir)["migrations"][DOTENV_COMPONENT]

    result = migrate_dotenv_v0()

    assert not result.performed
    assert env_path.read_text(encoding="utf-8") == "MY_TOKEN=secret\n"
    assert _read_doc(config_dir)["migrations"][DOTENV_COMPONENT] == stamp


def test_user_settings_path_sits_next_to_the_config_dotenv(config_dir: Path) -> None:
    assert user_settings_path() == config_dir / "settings.yaml"


# ── notifications_v0 ───────────────────────────────────────────────────


def _write_notifications(config_dir: Path, doc: dict) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "notifications.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


def test_notifications_migration_moves_values_and_retires_the_file(config_dir: Path) -> None:
    legacy = {
        "schema_version": 1,
        "enabled": False,
        "sound": False,
        "stray": "ignored",
        "events": {"ask_user": False, "bogus_event": False},
    }
    path = _write_notifications(config_dir, legacy)

    result = migrate_notifications_v0()

    assert result.performed
    assert result.migrated == {
        "notifications.enabled": False,
        "notifications.delivery.sound": False,
        "notifications.events.ask_user": False,
    }

    doc = _read_doc(config_dir)
    assert doc["notifications"]["enabled"] is False
    assert doc["notifications"]["delivery"]["sound"] is False
    assert doc["notifications"]["events"]["ask_user"] is False
    assert "stray" not in doc["notifications"]
    assert "bogus_event" not in doc["notifications"]["events"]
    assert isinstance(doc["migrations"][NOTIFICATIONS_COMPONENT], str)

    # Renamed, never deleted: the old document survives for rollback.
    assert not path.exists()
    retired = config_dir / "notifications.yaml.migrated"
    assert yaml.safe_load(retired.read_text(encoding="utf-8")) == legacy


def test_notifications_migration_second_run_is_a_noop(config_dir: Path) -> None:
    _write_notifications(config_dir, {"enabled": False})
    first = migrate_notifications_v0()
    stamp = _read_doc(config_dir)["migrations"][NOTIFICATIONS_COMPONENT]

    second = migrate_notifications_v0()

    assert first.performed
    assert not second.performed
    assert second.migrated == {}
    assert _read_doc(config_dir)["migrations"][NOTIFICATIONS_COMPONENT] == stamp


def test_notifications_migration_prefers_newer_document_edits(config_dir: Path) -> None:
    """A crash-rerun with ``pending_cleanup`` must not roll back a panel edit."""
    path = _write_notifications(config_dir, {"enabled": False})
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "settings.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "notifications": {"enabled": True},
                "migrations": {NOTIFICATIONS_COMPONENT: {"pending_cleanup": True}},
            }
        ),
        encoding="utf-8",
    )

    result = migrate_notifications_v0()

    assert result.performed
    assert result.migrated == {}
    doc = _read_doc(config_dir)
    assert doc["notifications"]["enabled"] is True
    assert isinstance(doc["migrations"][NOTIFICATIONS_COMPONENT], str)
    assert not path.exists()


def test_a_legacy_notification_the_document_outranks_is_named_in_the_log(
    config_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The dotenv branch's twin: retiring the file is the last moment to say so."""
    path = _write_notifications(config_dir, {"enabled": False})
    config_dir.joinpath("settings.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "notifications": {"enabled": True},
                "migrations": {NOTIFICATIONS_COMPONENT: {"pending_cleanup": True}},
            }
        ),
        encoding="utf-8",
    )

    with caplog.at_level(logging.INFO, logger="chrys.foundation.config.migrations"):
        migrate_notifications_v0()

    assert any("notifications.enabled" in message and str(path) in message for message in caplog.messages)


def test_notifications_cleanup_defers_when_the_file_moved_since_it_was_read(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retiring a file that changed since phase A carries away what nobody imported."""
    path = _write_notifications(config_dir, {"enabled": False})
    real_update = migrations.update_yaml_doc
    raced: list[bool] = []

    def rewrite_legacy_after_phase_a(*args: Any, **kwargs: Any) -> dict:
        doc = real_update(*args, **kwargs)
        if not raced:
            raced.append(True)
            _write_notifications(config_dir, {"enabled": False, "desktop": False})
        return doc

    monkeypatch.setattr(migrations, "update_yaml_doc", rewrite_legacy_after_phase_a)

    first = migrate_notifications_v0()

    assert first.migrated == {"notifications.enabled": False}
    assert path.is_file()
    assert not (config_dir / "notifications.yaml.migrated").exists()
    assert _read_doc(config_dir)["migrations"][NOTIFICATIONS_COMPONENT]["pending_cleanup"] is True

    second = migrate_notifications_v0()

    # Both fields: once the file has moved since phase A read it, the pending
    # document is no longer the newer of the two.
    assert second.migrated == {"notifications.enabled": False, "notifications.delivery.desktop": False}
    assert not path.exists()
    assert isinstance(_read_doc(config_dir)["migrations"][NOTIFICATIONS_COMPONENT], str)


def test_a_notifications_file_written_after_completion_wins_over_the_document(config_dir: Path) -> None:
    """Completion renames this path away, so anything back at it was written since."""
    # The old format cannot say "unset" — its reader defaulted every absent
    # field to on — so a partial file that comes back still describes a whole
    # state. Importing only its explicit fields would leave ``sound`` off here
    # while the build that wrote the file is showing the user a sound on.
    _write_notifications(config_dir, {"enabled": False, "sound": False})
    migrate_notifications_v0()
    assert _read_doc(config_dir)["notifications"]["delivery"]["sound"] is False

    path = _write_notifications(config_dir, {"enabled": True, "desktop": False})
    result = migrate_notifications_v0()

    assert result.performed
    assert result.migrated == {
        "notifications.enabled": True,
        "notifications.delivery.desktop": False,
        "notifications.delivery.sound": True,
        "notifications.suppress_when_focused": True,
        "notifications.events.approval_required": True,
        "notifications.events.ask_user": True,
        "notifications.events.turn_complete": True,
        "notifications.events.turn_error": True,
    }
    doc = _read_doc(config_dir)
    assert doc["notifications"]["enabled"] is True
    assert doc["notifications"]["delivery"]["desktop"] is False
    assert doc["notifications"]["delivery"]["sound"] is True
    assert not path.exists()
    assert isinstance(doc["migrations"][NOTIFICATIONS_COMPONENT], str)


def test_an_unfinished_notifications_migration_does_not_invent_the_absent_fields(config_dir: Path) -> None:
    """Before completion the sparse read is the right one — defaults would overwrite."""
    # A crash-rerun folding in "on" for every field the legacy file omits would
    # undo panel edits made between the two phases, which is the opposite of
    # what the document-wins rule is there for.
    _write_notifications(config_dir, {"enabled": False})
    (config_dir / "settings.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "notifications": {"delivery": {"sound": False}},
                "migrations": {NOTIFICATIONS_COMPONENT: {"pending_cleanup": True}},
            }
        ),
        encoding="utf-8",
    )

    result = migrate_notifications_v0()

    assert result.migrated == {"notifications.enabled": False}
    assert _read_doc(config_dir)["notifications"]["delivery"]["sound"] is False


def test_notifications_migration_fresh_install_records_completion(config_dir: Path) -> None:
    result = migrate_notifications_v0()

    assert result.performed
    assert result.migrated == {}
    assert isinstance(_read_doc(config_dir)["migrations"][NOTIFICATIONS_COMPONENT], str)
    assert not (config_dir / "notifications.yaml.migrated").exists()


def test_a_notifications_edit_between_parse_and_digest_cannot_be_certified(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fingerprint has to describe the bytes that were parsed, not a re-read."""
    # One write from an editor that ignores the sidecar lock is enough if the
    # two are separate reads: phase A parses ``enabled: false``, the file
    # becomes ``enabled: true``, and the fingerprint taken afterwards belongs
    # to the *true* file. Phase B then retires a file nobody read and records
    # the state it replaced as migrated.
    path = _write_notifications(config_dir, {"enabled": False})
    real_read_bytes, real_read_text = Path.read_bytes, Path.read_text
    reads: list[bool] = []

    def edit_once_after_the_first_read(self: Path) -> None:
        # Whichever call touches the file first — the editor does not know or
        # care which of our reads it lands behind.
        if self == path and not reads:
            reads.append(True)
            _write_notifications(config_dir, {"enabled": True})

    def read_bytes(self: Path) -> bytes:
        data = real_read_bytes(self)
        edit_once_after_the_first_read(self)
        return data

    def read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        data = real_read_text(self, *args, **kwargs)
        edit_once_after_the_first_read(self)
        return data

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(Path, "read_text", read_text)

    result = migrate_notifications_v0()

    assert result.migrated == {"notifications.enabled": False}
    # The imported state and the retired file must be the same file, so the one
    # that arrived since survives for the next start to fold in.
    assert path.is_file()
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {"enabled": True}
    assert _read_doc(config_dir)["migrations"][NOTIFICATIONS_COMPONENT]["pending_cleanup"] is True


def test_a_notifications_file_that_cannot_be_read_at_all_is_not_retired(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two failed digests are not proof that the file is the one phase A read."""
    # A file nobody could read has no identity: phase A's digest is ``None`` and
    # so is phase B's, and comparing them equal would rename away preferences
    # that were never imported — a perfectly good file, as here, that this
    # machine simply cannot read right now.
    path = _write_notifications(config_dir, {"enabled": False})
    real_read_bytes = Path.read_bytes

    def unreadable(self: Path) -> bytes:
        if self == path:
            raise OSError(5, "Input/output error")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", unreadable)

    result = migrate_notifications_v0()

    assert result.migrated == {}
    assert path.is_file()
    assert not (config_dir / "notifications.yaml.migrated").exists()
    # Still pending, so a start that can read the file finishes the work.
    assert _read_doc(config_dir)["migrations"][NOTIFICATIONS_COMPONENT] == {"pending_cleanup": True}


def test_a_notifications_file_that_reads_but_does_not_parse_still_retires(config_dir: Path) -> None:
    """Otherwise the component never converges: every boot retries a file that cannot improve."""
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "notifications.yaml").write_text("enabled: [not valid yaml", encoding="utf-8")

    result = migrate_notifications_v0()

    assert result.performed
    assert result.migrated == {}
    assert not (config_dir / "notifications.yaml").exists()
    assert (config_dir / "notifications.yaml.migrated").exists()
    assert isinstance(_read_doc(config_dir)["migrations"][NOTIFICATIONS_COMPONENT], str)


def test_notifications_migration_reads_backup_when_primary_is_corrupt(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "notifications.yaml").write_text("enabled: [not valid yaml", encoding="utf-8")
    (config_dir / "notifications.yaml.bak").write_text(yaml.safe_dump({"desktop": False}), encoding="utf-8")

    result = migrate_notifications_v0()

    assert result.migrated == {"notifications.delivery.desktop": False}
    assert _read_doc(config_dir)["notifications"]["delivery"]["desktop"] is False
    # Both copies retire together; a surviving backup would resurrect the
    # primary on the next boot's read.
    assert not (config_dir / "notifications.yaml").exists()
    assert not (config_dir / "notifications.yaml.bak").exists()
    assert (config_dir / "notifications.yaml.migrated").exists()
    assert (config_dir / "notifications.yaml.bak.migrated").exists()


def test_migrated_notification_values_load_as_the_user_layer(config_dir: Path) -> None:
    freeze_process_env()
    _write_notifications(config_dir, {"enabled": False, "events": {"turn_error": False}})

    migrate_notifications_v0()
    loaded = load_settings()

    assert loaded.settings.notifications_enabled is False
    assert loaded.settings.notifications_event_turn_error is False
    assert loaded.settings.notifications_desktop is True
    assert loaded.source_for("notifications.enabled").layer is Source.USER


def test_migration_keeps_a_no_timeout_value_instead_of_dropping_it(config_dir: Path) -> None:
    # ``0`` is the legacy spelling of "no timeout", and the field's canonical
    # value for it is ``None``. Re-reading that through the key's own coercer
    # turns it back into "say nothing", which would drop the setting while
    # phase B deletes the only line that still remembered it.
    _write_env(config_dir, "CHRYS_ASK_USER_TIMEOUT_SECONDS=0\n")
    freeze_process_env()

    result = migrate_dotenv_v0()

    assert result.performed
    assert result.migrated == {"tools.ask_user.timeout_seconds": None}
    assert result.warnings == ()
    assert _read_doc(config_dir)["tools"]["ask_user"]["timeout_seconds"] is None
    assert load_settings().settings.ask_user_timeout_seconds is None


def test_migration_leaves_the_file_alone_while_dotenv_is_disabled(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The opt-out has to reach a migration that *deletes* what it reads, or a
    # disabled boot empties the file and moves its values somewhere the opt-out
    # no longer covers. Nothing is marked, so an ordinary launch still migrates.
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    env_path = _write_env(config_dir, "CHRYS_THEME=solar\n")
    freeze_process_env()

    result = migrate_dotenv_v0()

    assert not result.performed
    assert result.migrated == {}
    assert env_path.read_text(encoding="utf-8") == "CHRYS_THEME=solar\n"
    assert not (config_dir / "settings.yaml").exists()

    # A launch from a shell without the flag migrates as usual. Modelled as a
    # new snapshot rather than a bare ``delenv`` because the opt-out is decided
    # once per process: clearing it mid-run deliberately changes nothing.
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED")
    assert migrate_dotenv_v0().migrated == {}
    _reset_process_env_snapshot_for_tests()
    freeze_process_env()
    assert migrate_dotenv_v0().migrated == {"ui.theme": "solar"}


# ── A built-in theme that changed its name ──────────────────────────


def _write_settings(config_dir: Path, text: str) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "settings.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "text", ["ui:\n  theme: chrys-dark\n  locale: en\n", "ui.theme: chrys-dark\nui:\n  locale: en\n"]
)
def test_a_saved_theme_is_renamed_to_what_it_goes_by_now(config_dir: Path, text: str) -> None:
    path = _write_settings(config_dir, text + "mystery: kept\n")

    result = migrate_renamed_theme()

    assert result.performed
    assert result.migrated == {"ui.theme": "chrys"}
    doc = _read_doc(config_dir)
    assert doc["ui"] == {"theme": "chrys", "locale": "en"}
    assert doc["mystery"] == "kept"
    # The rewrite is its own record: nothing is left to do, so nothing is kept to say so.
    assert "migrations" not in doc
    assert "chrys-dark" not in path.read_text(encoding="utf-8")
    assert not migrate_renamed_theme().performed


@pytest.mark.parametrize(
    "text",
    [
        "ui:\n  theme: chrys\n",
        "ui:\n  theme: chrys-legacy\n",
        # A user theme saved from the old one kept its own name, and keeps it.
        "ui:\n  theme: chrys-dark-1\n",
        "ui:\n  theme: 7\n",
        "ui: chrys-dark\n",
        "ui:\n  locale: en\n",
        ": not yaml\n",
    ],
)
def test_a_theme_that_was_not_renamed_costs_no_write(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    path = _write_settings(config_dir, text)
    write = create_autospec(migrations.update_yaml_doc, side_effect=AssertionError("nothing to write"))
    monkeypatch.setattr(migrations, "update_yaml_doc", write)

    assert not migrate_renamed_theme().performed

    write.assert_not_called()
    assert path.read_text(encoding="utf-8") == text


def test_renaming_a_theme_does_not_create_the_config_directory(config_dir: Path) -> None:
    assert not migrate_renamed_theme().performed
    assert not config_dir.exists()


def test_a_theme_renamed_between_the_look_and_the_lock_is_left_alone(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_settings(config_dir, "ui:\n  theme: nord\n")
    look = create_autospec(migrations.read_yaml_doc, return_value={"ui": {"theme": "chrys-dark"}})
    monkeypatch.setattr(migrations, "read_yaml_doc", look)

    assert not migrate_renamed_theme().performed

    look.assert_called_once()
    assert path.read_text(encoding="utf-8") == "ui:\n  theme: nord\n"


def test_a_retired_theme_name_loads_as_the_new_one_from_a_layer_nothing_rewrites(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHRYS_THEME", "chrys-dark")
    freeze_process_env()

    loaded = load_settings()

    assert loaded.settings.theme == "chrys"
    assert loaded.source_for("ui.theme").layer is Source.ENV
    assert loaded.warnings == ()


def test_a_retired_theme_name_loads_as_the_new_one_before_the_document_is_rewritten(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CHRYS_THEME", raising=False)
    freeze_process_env()
    _write_settings(config_dir, "ui:\n  theme: chrys-dark\n")

    loaded = load_settings()

    assert loaded.settings.theme == "chrys"
    assert loaded.source_for("ui.theme").layer is Source.USER


def test_a_retired_theme_name_cannot_be_saved_again(config_dir: Path) -> None:
    persist_theme("chrys-dark")

    assert _read_doc(config_dir)["ui"]["theme"] == "chrys"


def test_a_retired_theme_name_in_the_legacy_dotenv_is_imported_under_the_new_one(config_dir: Path) -> None:
    _write_env(config_dir, "CHRYS_THEME=chrys-dark\n")

    assert migrate_dotenv_v0().migrated == {"ui.theme": "chrys"}
    assert not migrate_renamed_theme().performed

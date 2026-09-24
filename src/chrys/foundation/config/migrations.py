# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Component migrations into the user settings document.

Importing an external legacy format is tracked in the ``migrations:`` ledger,
never inferred from the document's existence: a crash between "wrote the YAML"
and "cleaned the old source" must retry the cleanup on the next start, and a
user who pre-created ``settings.yaml`` by hand has not thereby migrated
anything. Each component entry is either a completion timestamp or
``{pending_cleanup: true}``.

Every migration is strictly three-phase, holding at most one configuration
lock at any moment:

- **A** (settings lock): read the old source, coerce, merge and commit with
  ``pending_cleanup: true``.
- **B** (the old source's own lock, never the settings lock): retire the old
  source. Nesting ``.env.lock`` inside the settings lock is a deadlock waiting
  for its second process, so cross-file consistency comes from the resumable
  ledger, not from lock nesting.
- **C** (settings lock again): replace ``pending_cleanup`` with the
  completion timestamp. A crash before C reruns B, which is idempotent.

Which side wins the merge depends on the ledger, and the two answers are not
in tension:

- While the component is **unfinished** — no entry, or ``pending_cleanup`` —
  the document wins. A rerun after a crash between phases must not roll a
  later panel edit back to the old file.
- Once it is **complete**, an old source that is active again became active
  *after* we finished, and the only thing that writes these formats now is a
  downgraded install that cannot see the document. That makes it the newer
  intent: it wins, and the component reopens to retire it again. Importing it
  under the document-wins rule instead would delete the newer value and keep
  the stale one, and skipping it would leave a legacy layer outranking every
  panel write for good — the exact "I changed it and nothing happened" these
  migrations exist to end.

Phase B never retires a source it cannot prove is the one phase A read, and
never on a read that did not succeed. The two are separate guarantees:

- *Same bytes.* The retirement is gated on a fingerprint taken before the
  parse and rechecked under the source's own lock. A source that moved in
  between digests as its older self, so the recheck fails and the component
  stays pending for the next start to fold in and retry.
- *A real read.* Cleanup removes the whole key space, not the keys that were
  found, so "this source holds nothing of ours" and "this source could not be
  read" must not arrive here as the same empty mapping. An unreadable source
  aborts the migration before anything is written or deleted.

:func:`migrate_renamed_theme` is not an import and needs none of this: it
rewrites one value inside the document, in one phase under the one lock.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chrys.foundation.config.coercion import Coerced, CoerceStatus
from chrys.foundation.config.env_file import config_env_path, update_env_file
from chrys.foundation.config.env_layers import (
    canonical_env_name,
    dotenv_disabled,
    process_env_snapshot,
    read_dotenv_snapshot,
)
from chrys.foundation.config.notification_events import NotificationEvent
from chrys.foundation.config.settings import RENAMED_THEMES, Settings
from chrys.foundation.config.settings_store import SettingsWarning
from chrys.foundation.config.spec import SettingOrigin, SettingSpec, Source, kind_accepts, specs_by_field
from chrys.foundation.config.user_settings import apply_settings_patch, flatten_user_doc, user_settings_path
from chrys.foundation.config.yaml_store import (
    LOCK_TIMEOUT_SECONDS,
    backup_path_for,
    lock_path_for,
    read_yaml_doc,
    read_yaml_document,
    update_yaml_doc,
)
from chrys.foundation.platform.files import file_fingerprint
from chrys.foundation.util.lock import FileLock

logger = logging.getLogger(__name__)

DOTENV_COMPONENT = "dotenv_v0"
NOTIFICATIONS_COMPONENT = "notifications_v0"

_DEFERRED_CLEANUP = "Legacy %s changed while it was being migrated; %s stays pending for the next start: %s"


class _AlreadyComplete(Exception):
    """Raised inside the mutator so the document is not rewritten per boot."""


@dataclass(frozen=True, slots=True)
class MigrationResult:
    """What one migration run moved, and what it had to leave behind."""

    migrated: dict[str, Any]
    warnings: tuple[SettingsWarning, ...]
    performed: bool
    """``False`` when the ledger said the work was already done."""


def migrate_dotenv_v0() -> MigrationResult:
    """Move the chrys keys out of ``~/.chrys/.env`` into ``settings.yaml``.

    Left in place, a legacy ``CHRYS_THEME`` line keeps loading as the
    ``USER_ENV`` layer and silently outranks every panel write — "I changed it
    and nothing happened". So the values move to the document and the lines
    go, in the three phases described in the module docstring.

    Reading is two-step because the two ends speak different grammars (the
    alias's ``env_coerce`` first, then the key's own ``coerce`` on the
    canonical result): ``CHRYS_HISTORY_DISABLE=1`` means *disable*, and a
    one-step read through the positive-polarity key coercer would migrate it
    as ``history.prompt.enabled: true`` — flipping the user's setting in the
    act of preserving it. The second step keeps migration from being the one
    channel that can write a value into the YAML no panel could.

    Values interpolate against the bootstrap snapshot, exactly as the
    ``USER_ENV`` layer would have resolved them: migration writes what the
    loader would have loaded.
    """
    if dotenv_disabled():
        # The opt-out has to reach this too, and not only because the values
        # would be unreadable: phase B *deletes* the lines it migrated, so
        # running with the file disabled would empty a file the user told us
        # not to read and move its values somewhere the opt-out cannot reach.
        # Nothing is marked complete, so a later ordinary launch migrates.
        return MigrationResult(migrated={}, warnings=(), performed=False)

    env_path = config_env_path()
    snapshot = process_env_snapshot()
    base = snapshot.values if snapshot is not None else {}
    specs = specs_by_field(Settings)
    aliases = {canonical_env_name(entry.env): entry for entry in specs.values() if entry.env is not None}
    known_keys = frozenset(entry.key for entry in specs.values())

    # Read before the settings lock is taken, and unconditionally — ahead of
    # the ledger — because a completed component can only notice that its
    # source came back by looking. The parsed values and the digest come from
    # one snapshot: cleanup deletes every alias regardless of what was found,
    # so "nothing to import" and "could not read it" must not look alike.
    snapshot = read_dotenv_snapshot(env_path, base=base)
    if snapshot is None:
        # The file is there and unreadable. Nothing is imported, nothing is
        # deleted, and the ledger is untouched, so the next start tries again.
        return MigrationResult(migrated={}, warnings=(), performed=False)
    fingerprint = snapshot.fingerprint
    candidates: dict[str, tuple[SettingSpec, str]] = {}
    for env_name, value in snapshot.values.items():
        spec_entry = aliases.get(canonical_env_name(env_name))
        if spec_entry is not None:
            # Later spellings win a fold collision, same as the loader.
            candidates[spec_entry.key] = (spec_entry, value)

    # Whether a line said anything is the coercer's call, not the string's:
    # for most keys an empty assignment is silence, but ``CHRYS_HISTORY_DISABLE=``
    # reads as "not 1" and imports as *enabled*. Both the completion guard and
    # phase B's retirement must hear a line exactly the way fold-in does, or a
    # speaking-empty line migrates yet survives — and, sitting in the higher
    # USER_ENV layer, shadows the very document it was folded into.
    speaking = frozenset(
        key
        for key, (spec_entry, value) in candidates.items()
        if (spec_entry.env_coerce or spec_entry.coerce)(value).status is not CoerceStatus.MISSING
    )

    migrated: dict[str, Any] = {}
    warnings: list[SettingsWarning] = []
    origin = SettingOrigin(layer=Source.USER_ENV, path=env_path)

    def fold_in(doc: dict[str, Any]) -> dict[str, Any]:
        entry = _ledger_entry(doc, DOTENV_COMPONENT)
        completed = entry is not None and not _pending_cleanup(entry)
        if completed and not speaking:
            # Presence is not reappearance here: ``.env`` outlives the
            # migration on purpose, still carrying the SDK variables — and the
            # lines phase B deliberately left behind, which say nothing and so
            # are not somebody writing here again either.
            raise _AlreadyComplete
        document_wins = not completed and _document_is_newer(entry, fingerprint)
        migrated.clear()
        warnings.clear()

        existing, _ = flatten_user_doc(doc, known_keys)
        for key, (spec_entry, value) in candidates.items():
            if key in existing and document_wins:
                # A rerun after a crash between phases must not roll a later
                # panel edit back to the old file. Once the source has moved
                # since we imported from it — or the component has completed —
                # the reverse holds; see the module docstring.
                #
                # Logged because phase B then deletes the line: without this the
                # legacy value and every trace of it having existed disappear in
                # the same run, and the only remaining evidence of the disagreement
                # is a document that quietly does not say what the file said. The
                # value itself stays out of the log — the key names the setting
                # well enough to explain the outcome.
                logger.info(
                    "Settings document already sets %s; dropping the legacy %s in %s",
                    key,
                    spec_entry.env or key,
                    env_path,
                )
                continue
            first = (spec_entry.env_coerce or spec_entry.coerce)(value)
            if first.status is CoerceStatus.MISSING:
                continue
            if first.status is CoerceStatus.INVALID:
                warnings.append(SettingsWarning(key=key, origin=origin, outcome=first))
                continue
            if first.status is CoerceStatus.CLAMPED:
                warnings.append(SettingsWarning(key=key, origin=origin, outcome=first))
            second = spec_entry.coerce(first.value)
            if second.status is CoerceStatus.MISSING and kind_accepts(spec_entry.kind, first.value):
                # The first step already produced the field's own canonical
                # value, and a coercer reads a canonical "unset" — ``None`` for
                # an ``OPTIONAL_INT`` — as "say nothing" rather than as the
                # value it is. Re-coercing it would drop the very setting being
                # migrated: ``CHRYS_ASK_USER_TIMEOUT_SECONDS=0`` means *no
                # timeout*, and the second read turns that into silence while
                # phase B deletes the line that said it.
                second = Coerced(status=CoerceStatus.VALID, value=first.value)
            if second.status is not CoerceStatus.VALID:
                warnings.append(SettingsWarning(key=key, origin=origin, outcome=second))
                continue
            migrated[key] = second.value

        apply_settings_patch(doc, migrated)
        _set_ledger(doc, DOTENV_COMPONENT, _pending(fingerprint))
        return doc

    try:
        update_yaml_doc(user_settings_path(), fold_in)
    except _AlreadyComplete:
        return MigrationResult(migrated={}, warnings=(), performed=False)

    result = MigrationResult(migrated=dict(migrated), warnings=tuple(warnings), performed=True)

    # Phase B. The existence check keeps a fresh install from growing an empty
    # ``.env``. Everything else — a file that raced in, changed, or stopped
    # being readable since phase A — fails the fingerprint inside the file's
    # own lock, and the component stays pending so the next start folds in
    # what changed before retiring anything.
    #
    # Only the lines this read actually heard a value on are removed. A line
    # that resolved to nothing was imported from nowhere, so deleting it would
    # be the one thing this migration must never do — destroy the only
    # remaining record of a setting. That is not hypothetical: cross-file
    # ``${VAR}`` references no longer resolve, and such a line reads as empty
    # here while still being the user's writing.
    retired = sorted(
        canonical_env_name(spec_entry.env)
        for key, (spec_entry, _) in candidates.items()
        if key in speaking and spec_entry.env is not None
    )
    if env_path.is_file() and retired:
        removed = fingerprint is not None and update_env_file(
            {},
            remove_keys=retired,
            expect_fingerprint=fingerprint,
        )
        if not removed:
            logger.warning(_DEFERRED_CLEANUP, "dotenv", DOTENV_COMPONENT, env_path)
            return result

    update_yaml_doc(user_settings_path(), _mark_complete(DOTENV_COMPONENT))
    return result


_LEGACY_NOTIFICATION_FIELDS: dict[str, str] = {
    "enabled": "notifications.enabled",
    "desktop": "notifications.delivery.desktop",
    "sound": "notifications.delivery.sound",
    "suppress_when_focused": "notifications.suppress_when_focused",
}


def _all_notification_keys() -> tuple[str, ...]:
    """Every settings key the retired ``notifications.yaml`` could speak for."""
    return (
        *_LEGACY_NOTIFICATION_FIELDS.values(),
        *(f"notifications.events.{event.value}" for event in NotificationEvent),
    )


def _legacy_notifications_path() -> Path:
    from chrys.foundation.platform import get_platform

    return get_platform().config_dir / "notifications.yaml"


def migrate_notifications_v0() -> MigrationResult:
    """Fold ``notifications.yaml`` into the ``notifications.*`` settings keys.

    The legacy loader's reading was ``bool(value)`` per present field with
    missing fields defaulting on, and that is reproduced here so migration
    writes what the old reader would have read. Every written value is a
    canonical bool, so nothing enters the document that a panel could not
    write. Unknown event names and stray fields are dropped, as they always
    were; ``schema_version`` belongs to the retired format.
    """
    legacy_path = _legacy_notifications_path()
    # Read under the legacy file's own lock *before* taking the settings lock:
    # migrations hold at most one configuration lock at any moment. Once the
    # component completes, the rename below makes this a cheap miss per boot.
    # The fingerprint comes out of the same call rather than from a digest
    # taken beside it: this read repairs a corrupt primary from its backup, so
    # a digest taken before it describes a file that no longer exists and one
    # taken after it races the read it is supposed to describe.
    read = read_yaml_document(legacy_path)
    legacy_doc, fingerprint = read.doc, read.fingerprint

    candidates: dict[str, bool] = {}
    if legacy_doc is not None:
        for field_name, key in _LEGACY_NOTIFICATION_FIELDS.items():
            if field_name in legacy_doc:
                candidates[key] = bool(legacy_doc[field_name])
        events = legacy_doc.get("events")
        if isinstance(events, dict):
            for raw_name, value in events.items():
                try:
                    event = NotificationEvent(str(raw_name))
                except ValueError:
                    continue
                candidates[f"notifications.events.{event.value}"] = bool(value)
    # What the old reader would have reported for the whole feature, which is
    # not the same list: it defaulted every absent field to on, so this format
    # has no way to say "unset" and a partial file still describes a complete
    # state. That only matters once the component has completed and the file
    # came back — then the file *is* the newer state and has to replace the
    # document wholesale, or a field the returning writer never mentioned keeps
    # a value the user's old build is not showing them. Before completion the
    # sparse list is the right one: filling in defaults there would overwrite
    # panel edits with "on" for every field the legacy file happens to omit.
    # (``.env`` needs no such distinction — absent keys there really are unset,
    # and fall through to the layers below.)
    complete_state = dict.fromkeys(_all_notification_keys(), True) | candidates

    known_keys = frozenset(entry.key for entry in specs_by_field(Settings).values())
    migrated: dict[str, Any] = {}

    def fold_in(doc: dict[str, Any]) -> dict[str, Any]:
        entry = _ledger_entry(doc, NOTIFICATIONS_COMPONENT)
        completed = entry is not None and not _pending_cleanup(entry)
        if completed and legacy_doc is None:
            # Presence *is* reappearance here, unlike ``.env``: this file has
            # no role left once the component completes, because completing it
            # renamed the file away. Anything back at this path was written by
            # something that does not know the document exists.
            raise _AlreadyComplete
        document_wins = not completed and _document_is_newer(entry, fingerprint)
        migrated.clear()
        existing, _ = flatten_user_doc(doc, known_keys)
        for key, value in (complete_state if completed else candidates).items():
            if key in existing and document_wins:
                # The document's value is newer: a rerun after a crash between
                # phases must not roll a later panel edit back to the old file.
                # Past completion the reverse holds — see the module docstring.
                # Logged for the same reason as its dotenv twin: phase B retires
                # the file, so this is the last moment the disagreement exists.
                logger.info(
                    "Settings document already sets %s; dropping the legacy value in %s",
                    key,
                    legacy_path,
                )
                continue
            migrated[key] = value
        apply_settings_patch(doc, migrated)
        _set_ledger(doc, NOTIFICATIONS_COMPONENT, _pending(fingerprint))
        return doc

    try:
        update_yaml_doc(user_settings_path(), fold_in)
    except _AlreadyComplete:
        return MigrationResult(migrated={}, warnings=(), performed=False)

    result = MigrationResult(migrated=dict(migrated), warnings=(), performed=True)

    # Phase B: the old document is renamed, never deleted — kept for rollback
    # and debugging. Its backup is renamed first; left behind, a crashed rerun
    # would restore the primary from it and resurrect the file just retired —
    # which is why the backup is retired even when the primary is already gone.
    # Under the file's own lock and the same fingerprint gate the dotenv
    # cleanup uses: renaming a file that changed since phase A read it carries
    # away preferences nobody imported. They would still be recoverable from
    # the ``.migrated`` name, which is the only reason this is not the P1 the
    # dotenv deletion would be.
    backup = backup_path_for(legacy_path)
    if legacy_path.is_file() or backup.is_file():
        with FileLock(lock_path_for(legacy_path), timeout=LOCK_TIMEOUT_SECONDS):
            current = file_fingerprint(legacy_path)
            # A file that is present but cannot be digested has no identity to
            # compare against, and ``None == None`` would pass for proof that
            # it is unchanged — retiring bytes nobody managed to read. A file
            # that reads but does not *parse* still digests, so that case
            # retires as it should rather than deferring forever.
            unreadable = current is None and legacy_path.is_file()
            if current != fingerprint or unreadable:
                logger.warning(_DEFERRED_CLEANUP, "notifications", NOTIFICATIONS_COMPONENT, legacy_path)
                return result
            if backup.is_file():
                backup.replace(backup.with_name(backup.name + ".migrated"))
            if legacy_path.is_file():
                legacy_path.replace(legacy_path.with_name(legacy_path.name + ".migrated"))

    update_yaml_doc(user_settings_path(), _mark_complete(NOTIFICATIONS_COMPONENT))
    return result


def migrate_renamed_theme() -> MigrationResult:
    """Rewrite a saved ``ui.theme`` that names a built-in theme by a name it no longer has.

    Nothing draws wrong without this — the loader follows :data:`RENAMED_THEMES`
    at every layer. It is here so the document says what the panel shows, and
    still names a theme for a build that has dropped the redirect.

    There is no ledger entry, because the rewrite is its own record: once the
    old name is gone there is nothing left to do, and a downgraded install that
    writes it back is simply renamed again. Looking before locking keeps the
    ordinary start — nothing to rename — from needing a writable config
    directory, which a read does not.
    """
    path = user_settings_path()
    key = specs_by_field(Settings)["theme"].key
    done = MigrationResult(migrated={}, warnings=(), performed=False)
    if _renamed_theme(read_yaml_doc(path), key) is None:
        return done

    migrated: dict[str, Any] = {}

    def rename(doc: dict[str, Any]) -> dict[str, Any]:
        successor = _renamed_theme(doc, key)
        if successor is None:
            # Another process got here between the look and the lock.
            raise _AlreadyComplete
        migrated[key] = successor
        return apply_settings_patch(doc, migrated)

    try:
        update_yaml_doc(path, rename)
    except _AlreadyComplete:
        return done
    return MigrationResult(migrated=dict(migrated), warnings=(), performed=True)


def _renamed_theme(doc: dict[str, Any] | None, key: str) -> str | None:
    """The name the document's theme goes by now, when it is saved under a retired one."""
    if doc is None:
        return None
    saved = flatten_user_doc(doc, frozenset({key}))[0].get(key)
    return RENAMED_THEMES.get(saved.strip()) if isinstance(saved, str) else None


def _mark_complete(component: str) -> Any:
    def mutate(doc: dict[str, Any]) -> dict[str, Any]:
        _set_ledger(doc, component, datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
        return doc

    return mutate


def _pending(fingerprint: str | None) -> dict[str, Any]:
    """The ledger entry for an import whose cleanup has not run yet.

    It carries the digest the values were imported from, because "the document
    is newer than the source" is only true while the source has not moved
    since; without recording it, the two reasons a component can be pending
    are indistinguishable. See :func:`_document_is_newer`.
    """
    entry: dict[str, Any] = {"pending_cleanup": True}
    if fingerprint is not None:
        entry["source_fingerprint"] = fingerprint
    return entry


def _document_is_newer(entry: Any, fingerprint: str | None) -> bool:
    """Whether the document outranks the legacy source where both hold a key.

    True for a component that has never run — the document may be something
    the user wrote by hand — and while the source still digests to what the
    pending import read, which is the crash-between-phases case: a rerun there
    must not roll a later panel edit back to the old file.

    False once the source has changed since that import, which is exactly what
    a *deferred* cleanup means: phase B refused because somebody wrote there
    afterwards. Treating the document as newer then is how the value that
    caused the deferral gets skipped on the rerun and deleted by the cleanup
    that follows it — imported nowhere, gone for good.

    A pending entry with no digest recorded reads as "unchanged". Only a build
    that predates this could write one, and the crash-rerun reading is the one
    that cannot lose data.
    """
    if not isinstance(entry, dict):
        return True
    recorded = entry.get("source_fingerprint")
    return not isinstance(recorded, str) or recorded == fingerprint


def _set_ledger(doc: dict[str, Any], component: str, value: Any) -> None:
    ledger = doc.get("migrations")
    if not isinstance(ledger, dict):
        ledger = {}
        doc["migrations"] = ledger
    ledger[component] = value


def _ledger_entry(doc: dict[str, Any], component: str) -> Any:
    ledger = doc.get("migrations")
    return ledger.get(component) if isinstance(ledger, dict) else None


def _pending_cleanup(entry: Any) -> bool:
    return isinstance(entry, dict) and bool(entry.get("pending_cleanup"))

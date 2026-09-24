# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the memory loader."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from chrys.service.context.memory_loader import (
    DEFAULT_TOKEN_CAP,
    FOLDER_MAX_FILES,
    load_memory_content,
    validate_memory_config,
)
from chrys.service.profiles.agents.schema import MemoryConfig

# ---------------------------------------------------------------------------
# Empty / no-op cases
# ---------------------------------------------------------------------------


def test_empty_config_returns_empty_content() -> None:
    result = load_memory_content(MemoryConfig(), workspace_cwd="/tmp")
    assert result.text == ""
    assert result.loaded_files == []
    assert result.token_count == 0
    assert result.truncated is False


def test_no_workspace_returns_empty_content(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("hi", encoding="utf-8")
    config = MemoryConfig(files=["a.md"])
    result = load_memory_content(config, workspace_cwd=None)
    assert result.text == ""
    assert result.loaded_files == []


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def test_loads_single_md_file(tmp_path: Path) -> None:
    (tmp_path / "spec.md").write_text("# Spec\n\nThe spec content.", encoding="utf-8")
    config = MemoryConfig(files=["spec.md"])

    result = load_memory_content(config, workspace_cwd=str(tmp_path))

    assert result.loaded_files == ["spec.md"]
    assert "## File: spec.md" in result.text
    assert "The spec content." in result.text
    assert result.truncated is False


def test_loads_single_txt_file(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("plain text notes", encoding="utf-8")
    config = MemoryConfig(files=["notes.txt"])

    result = load_memory_content(config, workspace_cwd=str(tmp_path))

    assert result.loaded_files == ["notes.txt"]
    assert "plain text notes" in result.text


def test_skips_non_md_txt_extensions(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text("print('hi')", encoding="utf-8")
    config = MemoryConfig(files=["code.py"])

    result = load_memory_content(config, workspace_cwd=str(tmp_path))

    # Wrong-extension files are reported as missing (not silently dropped) so
    # the caller can warn the user.
    assert "code.py" in result.missing_paths
    assert result.loaded_files == []


def test_missing_file_reported(tmp_path: Path) -> None:
    config = MemoryConfig(files=["does-not-exist.md"])

    result = load_memory_content(config, workspace_cwd=str(tmp_path))

    assert "does-not-exist.md" in result.missing_paths
    assert result.loaded_files == []
    assert result.text == ""


def test_file_path_appears_before_content(tmp_path: Path) -> None:
    """Agent must know the source — path must precede content."""
    (tmp_path / "ref.md").write_text("CONTENT_MARKER", encoding="utf-8")
    config = MemoryConfig(files=["ref.md"])

    result = load_memory_content(config, workspace_cwd=str(tmp_path))

    path_idx = result.text.index("## File: ref.md")
    content_idx = result.text.index("CONTENT_MARKER")
    assert path_idx < content_idx


# ---------------------------------------------------------------------------
# Folders — bounded breadth-first scan + filtering
# ---------------------------------------------------------------------------


def test_folder_recursive_walk(tmp_path: Path) -> None:
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "knowledge" / "api.md").write_text("api docs", encoding="utf-8")
    (tmp_path / "knowledge" / "nested").mkdir()
    (tmp_path / "knowledge" / "nested" / "deep.md").write_text("deep docs", encoding="utf-8")
    (tmp_path / "knowledge" / "ignore.py").write_text("nope", encoding="utf-8")
    config = MemoryConfig(folders=["knowledge"])

    result = load_memory_content(config, workspace_cwd=str(tmp_path))

    # Both .md files loaded, .py ignored
    paths = result.loaded_files
    assert any(p.endswith("api.md") for p in paths)
    assert any(p.endswith("deep.md") for p in paths)
    assert not any(p.endswith(".py") for p in paths)
    # Folder header appears before its files
    assert "## Folder: knowledge/" in result.text


def test_folder_walk_stops_below_depth_cap(tmp_path: Path) -> None:
    # The bounded scan covers the folder itself plus FOLDER_MAX_DEPTH
    # subdirectory levels; anything deeper is deliberately not visited.
    root = tmp_path / "notes"
    level1 = root / "a"
    level2 = level1 / "b"
    level3 = level2 / "c"
    level3.mkdir(parents=True)
    (root / "top.md").write_text("top", encoding="utf-8")
    (level1 / "one.md").write_text("one", encoding="utf-8")
    (level2 / "two.md").write_text("two", encoding="utf-8")
    (level3 / "three.md").write_text("three", encoding="utf-8")
    config = MemoryConfig(folders=["notes"])

    result = load_memory_content(config, workspace_cwd=str(tmp_path))

    paths = result.loaded_files
    assert any(p.endswith("top.md") for p in paths)
    assert any(p.endswith("one.md") for p in paths)
    assert any(p.endswith("two.md") for p in paths)
    assert not any(p.endswith("three.md") for p in paths)


def test_folder_walk_caps_file_count_shallow_first(tmp_path: Path) -> None:
    # The per-folder cap keeps the folder's own (typically curated) files
    # and drops deep ones: breadth-first, the top level wins.
    root = tmp_path / "notes"
    sub = root / "sub"
    sub.mkdir(parents=True)
    for i in range(FOLDER_MAX_FILES - 1):
        (root / f"top_{i:03d}.md").write_text("t", encoding="utf-8")
    (sub / "kept.md").write_text("kept", encoding="utf-8")
    (sub / "zz_dropped.md").write_text("dropped", encoding="utf-8")
    config = MemoryConfig(folders=["notes"])

    result = load_memory_content(config, workspace_cwd=str(tmp_path))

    paths = result.loaded_files
    assert len(paths) == FOLDER_MAX_FILES
    assert sum(1 for p in paths if "top_" in p) == FOLDER_MAX_FILES - 1
    assert any(p.endswith("kept.md") for p in paths)
    assert not any(p.endswith("zz_dropped.md") for p in paths)


def test_shallow_files_win_token_cap_over_deep_lexical_order(tmp_path: Path) -> None:
    # The token cap consumes candidates in list order, so the walk must
    # emit the folder's own files before subdirectory files even when a
    # subdirectory sorts lexically earlier: a/deep.md must not crowd out
    # z_top.md under a tight cap.
    notes = tmp_path / "notes"
    sub = notes / "a"
    sub.mkdir(parents=True)
    (notes / "z_top.md").write_text("top", encoding="utf-8")
    (sub / "deep.md").write_text("deep " * 20_000, encoding="utf-8")
    config = MemoryConfig(folders=["notes"])

    result = load_memory_content(config, workspace_cwd=str(tmp_path), token_cap=1_000)

    assert result.loaded_files == ["notes/z_top.md"]
    assert result.skipped_files == ["notes/a/deep.md"]
    assert result.truncated is True


def test_walk_work_bounded_on_wide_fileless_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The file/depth caps bound what is collected, not what is scanned —
    # the entry budget must bound the traversal itself so a wide,
    # file-less tree cannot trigger one scandir per subdirectory.
    import chrys.service.context.memory_loader as memory_loader

    monkeypatch.setattr(memory_loader, "FOLDER_SCAN_MAX_ENTRIES", 50)
    root = tmp_path / "wide"
    root.mkdir()
    for i in range(250):
        (root / f"child_{i:03d}").mkdir()
    calls = 0
    real_scandir = os.scandir

    def counting_scandir(path: object) -> object:
        nonlocal calls
        calls += 1
        return real_scandir(path)

    monkeypatch.setattr(memory_loader.os, "scandir", counting_scandir)
    config = MemoryConfig(folders=["wide"])

    result = load_memory_content(config, workspace_cwd=str(tmp_path))

    assert result.loaded_files == []
    # Every queued subdirectory was an entry of its parent, so scandir
    # calls are bounded by the entry budget (plus the root itself).
    assert calls <= 50 + 1


def test_walk_entry_budget_truncates_oversized_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A single directory wider than the budget is truncated in filesystem
    # order instead of being fully materialized and sorted.
    import chrys.service.context.memory_loader as memory_loader

    monkeypatch.setattr(memory_loader, "FOLDER_SCAN_MAX_ENTRIES", 10)
    root = tmp_path / "big"
    root.mkdir()
    for i in range(15):
        (root / f"f_{i:02d}.md").write_text("x", encoding="utf-8")
    config = MemoryConfig(folders=["big"])

    result = load_memory_content(config, workspace_cwd=str(tmp_path))

    assert len(result.loaded_files) == 10


def test_explicit_file_wins_over_folder_discovery(tmp_path: Path) -> None:
    # A file configured explicitly AND discovered by a folder scan loads
    # exactly once — in the Files section; the folder section omits it.
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "spec.md").write_text("spec", encoding="utf-8")
    (docs / "other.md").write_text("other", encoding="utf-8")
    config = MemoryConfig(files=["docs/spec.md"], folders=["docs"])

    result = load_memory_content(config, workspace_cwd=str(tmp_path))

    assert result.text.count("File: docs/spec.md") == 1
    assert "## File: docs/spec.md" in result.text  # files-section level
    assert "### File: docs/other.md" in result.text  # folder-section level
    assert sum(1 for p in result.loaded_files if p.endswith("spec.md")) == 1


def test_overlapping_folders_dedupe_discovered_files(tmp_path: Path) -> None:
    # ``docs`` and ``docs/sub`` both discover sub/x.md — the earlier
    # configured folder wins and the file loads once.
    sub = tmp_path / "docs" / "sub"
    sub.mkdir(parents=True)
    (sub / "x.md").write_text("x", encoding="utf-8")
    config = MemoryConfig(folders=["docs", "docs/sub"])

    result = load_memory_content(config, workspace_cwd=str(tmp_path))

    assert result.text.count("File: docs/sub/x.md") == 1
    assert sum(1 for p in result.loaded_files if p.endswith("x.md")) == 1


def test_case_alias_spellings_dedupe_to_one_load(tmp_path: Path) -> None:
    # On a case-insensitive filesystem (macOS default, Windows) both
    # spellings resolve to the same physical file; identity is stat-based,
    # so the case alias collapses.  On case-sensitive filesystems the
    # second spelling simply doesn't exist — either way exactly one load.
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "spec.md").write_text("spec", encoding="utf-8")
    config = MemoryConfig(files=["docs/spec.md", "DOCS/SPEC.MD"], folders=["docs"])

    result = load_memory_content(config, workspace_cwd=str(tmp_path))

    assert sum(1 for p in result.loaded_files if p.lower().endswith("spec.md")) == 1
    assert result.text.lower().count("file: docs/spec.md") == 1


def test_symlinked_folder_alias_does_not_erase_earlier_folder(tmp_path: Path) -> None:
    # folders=["real", "alias"] where alias symlinks to real: the alias
    # scan would dedupe every file and must NOT overwrite the earlier
    # folder's group with an empty list.
    real = tmp_path / "real"
    real.mkdir()
    (real / "a.md").write_text("a", encoding="utf-8")
    try:
        (tmp_path / "alias").symlink_to(real, target_is_directory=True)
    except OSError, NotImplementedError:
        pytest.skip("symlink creation not supported")
    config = MemoryConfig(folders=["real", "alias"])

    result = load_memory_content(config, workspace_cwd=str(tmp_path))

    assert sum(1 for p in result.loaded_files if p.endswith("a.md")) == 1
    assert "### File: real/a.md" in result.text
    assert "alias" not in result.missing_paths


def test_alias_duplicates_do_not_consume_folder_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The per-folder cap counts unique physical files: a symlink alias of
    # an already-counted file must not eat a cap slot, so a unique file
    # sorting after the alias still loads.
    import chrys.service.context.memory_loader as memory_loader

    monkeypatch.setattr(memory_loader, "FOLDER_MAX_FILES", 2)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a_real.md").write_text("real", encoding="utf-8")
    try:
        (docs / "b_alias.md").symlink_to(docs / "a_real.md")
    except OSError, NotImplementedError:
        pytest.skip("symlink creation not supported")
    (docs / "z_unique.md").write_text("unique", encoding="utf-8")
    config = MemoryConfig(folders=["docs"])

    result = load_memory_content(config, workspace_cwd=str(tmp_path))

    paths = result.loaded_files
    assert sum(1 for p in paths if p.endswith(("a_real.md", "b_alias.md"))) == 1
    assert any(p.endswith("z_unique.md") for p in paths)
    assert len(paths) == 2


def test_loaded_files_grouped_by_configured_source(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("agent notes", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "design.md").write_text("design notes", encoding="utf-8")
    config = MemoryConfig(files=["AGENTS.md"], folders=["docs"])

    result = load_memory_content(config, workspace_cwd=str(tmp_path))

    assert result.loaded_files_by_source == {
        "AGENTS.md": ["AGENTS.md"],
        "docs/": ["docs/design.md"],
    }


def test_folder_includes_hidden_md_files(tmp_path: Path) -> None:
    """User spec: include hidden files as long as they end with .md/.txt."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / ".hidden.md").write_text("hidden but loaded", encoding="utf-8")
    config = MemoryConfig(folders=["docs"])

    result = load_memory_content(config, workspace_cwd=str(tmp_path))

    assert any(p.endswith(".hidden.md") for p in result.loaded_files)
    assert "hidden but loaded" in result.text


def test_missing_folder_reported(tmp_path: Path) -> None:
    config = MemoryConfig(folders=["does-not-exist"])

    result = load_memory_content(config, workspace_cwd=str(tmp_path))

    assert "does-not-exist" in result.missing_paths
    assert result.loaded_files == []


def test_folder_files_sorted_for_stable_order(tmp_path: Path) -> None:
    (tmp_path / "k").mkdir()
    (tmp_path / "k" / "z.md").write_text("z", encoding="utf-8")
    (tmp_path / "k" / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "k" / "m.md").write_text("m", encoding="utf-8")
    config = MemoryConfig(folders=["k"])

    result = load_memory_content(config, workspace_cwd=str(tmp_path))

    folder_files = [p for p in result.loaded_files if p.startswith("k")]
    # Sort key is the workspace-relative path, so alphabetical order.
    # Loader emits POSIX separators on every platform.
    assert folder_files == [f"k/{n}" for n in ("a.md", "m.md", "z.md")]


# ---------------------------------------------------------------------------
# Symlink cycle detection
# ---------------------------------------------------------------------------


def test_symlink_cycle_does_not_loop(tmp_path: Path) -> None:
    """Folder containing a symlink back to itself must not infinite-loop."""
    (tmp_path / "loop").mkdir()
    (tmp_path / "loop" / "real.md").write_text("real content", encoding="utf-8")
    cycle = tmp_path / "loop" / "back"
    try:
        cycle.symlink_to(tmp_path / "loop")
    except OSError, NotImplementedError:
        # Windows without symlink privileges — skip
        import pytest

        pytest.skip("symlink creation not supported")
    config = MemoryConfig(folders=["loop"])

    # Should terminate; if cycle detection is broken this hangs.
    result = load_memory_content(config, workspace_cwd=str(tmp_path))

    assert any(p.endswith("real.md") for p in result.loaded_files)


# ---------------------------------------------------------------------------
# Workspace containment and absolute file paths
# ---------------------------------------------------------------------------


def test_loads_absolute_file_path(tmp_path: Path) -> None:
    """Individual memory files may be configured as absolute paths."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("ABSOLUTE MEMORY", encoding="utf-8")

    config = MemoryConfig(files=[str(outside)])
    result = load_memory_content(config, workspace_cwd=str(workspace))

    assert result.loaded_files == [outside.as_posix()]
    assert f"## File: {outside.as_posix()}" in result.text
    assert "ABSOLUTE MEMORY" in result.text


def test_rejects_dotdot_traversal(tmp_path: Path) -> None:
    """``..`` traversal that escapes the workspace must be rejected."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("LEAK", encoding="utf-8")

    config = MemoryConfig(files=["../outside.md"])
    result = load_memory_content(config, workspace_cwd=str(workspace))

    assert result.loaded_files == []
    assert "../outside.md" in result.missing_paths
    assert "LEAK" not in result.text


def test_relative_file_symlink_escape_skipped(tmp_path: Path) -> None:
    """Relative file entries must stay workspace-contained after symlink resolution."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("SECRET", encoding="utf-8")

    try:
        (workspace / "alias.md").symlink_to(outside)
    except OSError, NotImplementedError:
        import pytest

        pytest.skip("symlink creation not supported")

    config = MemoryConfig(files=["alias.md"])
    result = load_memory_content(config, workspace_cwd=str(workspace))

    assert result.loaded_files == []
    assert "alias.md" in result.missing_paths
    assert "SECRET" not in result.text


def test_loads_absolute_folder_path(tmp_path: Path) -> None:
    """Folders may be configured as absolute paths outside the workspace,
    mirroring absolute files; contents display as absolute paths."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "outside_dir"
    outside.mkdir()
    (outside / "notes.md").write_text("ABSOLUTE FOLDER MEMORY", encoding="utf-8")

    config = MemoryConfig(folders=[str(outside)])
    result = load_memory_content(config, workspace_cwd=str(workspace))

    expected = (outside / "notes.md").resolve().as_posix()
    assert result.loaded_files == [expected]
    assert f"## Folder: {outside.resolve().as_posix()}/" in result.text
    assert "ABSOLUTE FOLDER MEMORY" in result.text
    assert result.missing_paths == []


def test_foreign_absolute_folder_reported_missing(tmp_path: Path) -> None:
    """An absolute folder for the other platform cannot load on this host."""
    foreign = "/other/host/docs" if os.name == "nt" else r"C:\other\host\docs"
    config = MemoryConfig(folders=[foreign])
    result = load_memory_content(config, workspace_cwd=str(tmp_path))

    assert result.loaded_files == []
    assert foreign in result.missing_paths


def test_absolute_folder_symlink_escape_skipped(tmp_path: Path) -> None:
    """An absolute folder's walk is contained to the folder itself: a
    symlink inside it pointing elsewhere must be skipped."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "outside_dir"
    outside.mkdir()
    (outside / "real.md").write_text("REAL", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "secret.md").write_text("SECRET", encoding="utf-8")

    try:
        (outside / "linked").symlink_to(elsewhere)
    except OSError, NotImplementedError:
        import pytest

        pytest.skip("symlink creation not supported")

    config = MemoryConfig(folders=[str(outside)])
    result = load_memory_content(config, workspace_cwd=str(workspace))

    assert any(p.endswith("real.md") for p in result.loaded_files)
    assert "SECRET" not in result.text
    assert not any("secret.md" in p for p in result.loaded_files)


def test_folder_symlink_escape_skipped(tmp_path: Path) -> None:
    """A symlink inside a configured folder that targets outside the
    workspace must be skipped, not silently followed."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    docs = workspace / "docs"
    docs.mkdir()
    (docs / "real.md").write_text("REAL", encoding="utf-8")

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("SECRET", encoding="utf-8")

    try:
        (docs / "linked").symlink_to(outside)
    except OSError, NotImplementedError:
        import pytest

        pytest.skip("symlink creation not supported")

    config = MemoryConfig(folders=["docs"])
    result = load_memory_content(config, workspace_cwd=str(workspace))

    assert any(p.endswith("real.md") for p in result.loaded_files)
    assert "SECRET" not in result.text
    assert not any("secret.md" in p for p in result.loaded_files)


def test_file_symlink_escape_skipped(tmp_path: Path) -> None:
    """A single-file symlink inside the workspace whose target is outside
    must be skipped during folder walk."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    docs = workspace / "docs"
    docs.mkdir()
    (docs / "real.md").write_text("REAL", encoding="utf-8")

    outside = tmp_path / "outside_secret.md"
    outside.write_text("SECRET", encoding="utf-8")

    try:
        (docs / "alias.md").symlink_to(outside)
    except OSError, NotImplementedError:
        import pytest

        pytest.skip("symlink creation not supported")

    config = MemoryConfig(folders=["docs"])
    result = load_memory_content(config, workspace_cwd=str(workspace))

    assert any(p.endswith("real.md") for p in result.loaded_files)
    assert "SECRET" not in result.text


def test_paths_emitted_in_posix_form(tmp_path: Path) -> None:
    """Paths surfaced to the agent must use ``/`` regardless of OS."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "api.md").write_text("body", encoding="utf-8")

    config = MemoryConfig(folders=["docs"])
    result = load_memory_content(config, workspace_cwd=str(tmp_path))

    for p in result.loaded_files:
        assert "\\" not in p
    assert "## Folder: docs/" in result.text
    assert "## File: docs/api.md" in result.text


# ---------------------------------------------------------------------------
# Token cap
# ---------------------------------------------------------------------------


def test_truncates_at_token_cap(tmp_path: Path) -> None:
    big = "word " * 5000  # ~5k tokens with heuristic
    (tmp_path / "a.md").write_text(big, encoding="utf-8")
    (tmp_path / "b.md").write_text(big, encoding="utf-8")
    (tmp_path / "c.md").write_text(big, encoding="utf-8")
    config = MemoryConfig(files=["a.md", "b.md", "c.md"])

    # Cap small enough that not all three fit.
    result = load_memory_content(config, workspace_cwd=str(tmp_path), token_cap=6000)

    assert result.truncated is True
    assert len(result.loaded_files) < 3
    assert len(result.skipped_files) >= 1
    # Loaded ∩ skipped should be empty
    assert set(result.loaded_files).isdisjoint(set(result.skipped_files))


def test_default_cap_is_35k() -> None:
    assert DEFAULT_TOKEN_CAP == 35_000


def test_token_count_matches_rendered_text(tmp_path: Path) -> None:
    """``token_count`` is the budget gate — must equal the tokenizer's count
    of the rendered text, otherwise the warn/truncate path drifts."""
    from chrys.foundation.text.tokenizer import MixedLanguageTokenizer

    (tmp_path / "spec.md").write_text("# Header\n\nBody paragraph.", encoding="utf-8")
    config = MemoryConfig(files=["spec.md"])

    result = load_memory_content(config, workspace_cwd=str(tmp_path))

    assert result.token_count > 0
    assert result.token_count == MixedLanguageTokenizer().count_tokens(result.text)


def test_truncation_inside_folder_partially_loads(tmp_path: Path) -> None:
    """Cap hit *between* files inside a folder — earlier ones load, later
    ones go to ``skipped_files`` and the folder header is still emitted.

    Uses a smaller per-file budget than ``test_truncates_at_token_cap``
    so at least the first folder file definitely fits before the cap.
    """
    medium = "word " * 2000  # ~2.5k tokens each
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text(medium, encoding="utf-8")
    (tmp_path / "docs" / "b.md").write_text(medium, encoding="utf-8")
    (tmp_path / "docs" / "c.md").write_text(medium, encoding="utf-8")
    config = MemoryConfig(folders=["docs"])

    # Cap fits ~2 of the 3 medium files.
    result = load_memory_content(config, workspace_cwd=str(tmp_path), token_cap=6000)

    assert result.truncated is True
    assert 1 <= len(result.loaded_files) < 3
    assert len(result.skipped_files) >= 1
    # Folder header should appear because at least the first file fit under it.
    assert "## Folder: docs/" in result.text
    # Loaded ∩ skipped is empty.
    assert set(result.loaded_files).isdisjoint(set(result.skipped_files))


# ---------------------------------------------------------------------------
# Files-first then folders ordering
# ---------------------------------------------------------------------------


def test_empty_folder_yields_no_folder_block(tmp_path: Path) -> None:
    """A folder that exists but contains no ``.md``/``.txt`` files is a
    silent no-op — no header emitted, no error reported, no duplicate
    entry in ``missing_paths`` (the folder itself wasn't missing)."""
    (tmp_path / "f").mkdir()
    (tmp_path / "f" / "ignore.py").write_text("nope", encoding="utf-8")
    config = MemoryConfig(folders=["f"])

    result = load_memory_content(config, workspace_cwd=str(tmp_path))

    assert result.loaded_files == []
    assert "## Folder: f/" not in result.text
    assert "f" not in result.missing_paths


def test_multiple_folders_each_get_a_header(tmp_path: Path) -> None:
    """Two configured folders ⇒ two ``## Folder:`` headers, in config order."""
    (tmp_path / "alpha").mkdir()
    (tmp_path / "alpha" / "a.md").write_text("ALPHA_BODY", encoding="utf-8")
    (tmp_path / "beta").mkdir()
    (tmp_path / "beta" / "b.md").write_text("BETA_BODY", encoding="utf-8")
    config = MemoryConfig(folders=["alpha", "beta"])

    result = load_memory_content(config, workspace_cwd=str(tmp_path))

    alpha_idx = result.text.index("## Folder: alpha/")
    beta_idx = result.text.index("## Folder: beta/")
    assert alpha_idx < beta_idx
    assert "ALPHA_BODY" in result.text
    assert "BETA_BODY" in result.text


def test_mixed_loaded_and_missing_in_one_call(tmp_path: Path) -> None:
    """Real-world common case — some configured paths exist, others don't.
    Loader must populate both ``loaded_files`` and ``missing_paths`` and
    still render the surviving content."""
    (tmp_path / "real.md").write_text("REAL", encoding="utf-8")
    config = MemoryConfig(files=["real.md", "ghost.md", "wrong.pdf"])

    result = load_memory_content(config, workspace_cwd=str(tmp_path))

    assert result.loaded_files == ["real.md"]
    assert "ghost.md" in result.missing_paths
    assert "wrong.pdf" in result.missing_paths
    assert "REAL" in result.text
    assert result.truncated is False


def test_files_render_before_folders(tmp_path: Path) -> None:
    (tmp_path / "top.md").write_text("TOP", encoding="utf-8")
    (tmp_path / "f").mkdir()
    (tmp_path / "f" / "inner.md").write_text("INNER", encoding="utf-8")
    config = MemoryConfig(files=["top.md"], folders=["f"])

    result = load_memory_content(config, workspace_cwd=str(tmp_path))

    top_idx = result.text.index("TOP")
    inner_idx = result.text.index("INNER")
    assert top_idx < inner_idx


# ---------------------------------------------------------------------------
# Encoding detection
# ---------------------------------------------------------------------------


def test_decodes_non_utf8_file(tmp_path: Path) -> None:
    """Non-UTF-8 files must decode via decode_bytes.  We use a UTF-16-BOM
    sample because BOM detection is deterministic — short Latin-1 strings
    can be mis-detected by statistical heuristics, which would make the
    assertion flaky without exercising the real concern (that the loader
    delegates to ``decode_bytes`` rather than blindly assuming UTF-8)."""
    p = tmp_path / "utf16.txt"
    p.write_bytes("café résumé — full sentence with enough body.".encode("utf-16"))
    config = MemoryConfig(files=["utf16.txt"])

    result = load_memory_content(config, workspace_cwd=str(tmp_path))

    assert "utf16.txt" in result.loaded_files
    assert "café résumé" in result.text


# ---------------------------------------------------------------------------
# validate_memory_config — duplicates, casing, file-under-folder
# ---------------------------------------------------------------------------


def test_validate_empty_config_has_no_errors() -> None:
    assert validate_memory_config(MemoryConfig()) == []


def test_validate_blank_entries_are_skipped() -> None:
    # Blanks are filtered by the panel on save and shouldn't surface as errors.
    cfg = MemoryConfig(files=["", "   ", "spec.md"], folders=["", "docs"])
    assert validate_memory_config(cfg) == []


def test_validate_accepts_absolute_file_path() -> None:
    cfg = MemoryConfig(files=["/etc/secret.md"])
    assert validate_memory_config(cfg) == []


def test_validate_accepts_absolute_folder_path() -> None:
    cfg = MemoryConfig(folders=["/var/notes"])
    assert validate_memory_config(cfg) == []


def test_validate_accepts_windows_absolute_file_path_cross_platform() -> None:
    cfg = MemoryConfig(files=[r"C:\secret.md"])
    assert validate_memory_config(cfg) == []


def test_validate_accepts_windows_absolute_folder_path_cross_platform() -> None:
    cfg = MemoryConfig(folders=[r"C:\notes"])
    assert validate_memory_config(cfg) == []


def test_validate_workspace_cwd_dedupes_absolute_and_relative_folders() -> None:
    """An absolute folder spelling of a workspace-relative folder is a
    duplicate once normalized through the workspace cwd."""
    cfg = MemoryConfig(folders=["docs", "/ws/docs"])
    errors = validate_memory_config(cfg, workspace_cwd="/ws")
    assert len(errors) == 1
    assert "Memory Folder 2" in errors[0]
    assert "duplicates Folder 1" in errors[0]


def test_validate_rejects_root_folders() -> None:
    # Roots keep their trailing slash through normalization ("/", "//"),
    # collapse to a bare drive letter ("C:\\" → "c:"), or normalize to a
    # UNC host/share root ("\\\\server\\share" → "//server/share");
    # walking one would scan the entire volume, so the validator rejects
    # the entry outright.
    for raw in (
        "/",
        "C:\\",
        "C:",
        "//",
        r"\\server\share",
        "//server/share",
        r"\\?\C:\\",
        r"\\?\C:",
        r"\\?\UNC\server\share",
        "//?/unc/server/share",
        r"\\.\C:\\",
        r"\\?\GLOBALROOT\Device\HarddiskVolume1",
        r"\\.\GLOBALROOT\Device\HarddiskVolumeShadowCopy1",
        r"\\?\Volume{d5588183-0d3f-4a24-a8ba-0c72e7b195e2}\\",
        r"\\.\UNC\server\share",
        "//./unc/server/share",
        r"\\?\GLOBALROOT\Device\LanmanRedirector\server\share",
        r"\\?\GLOBALROOT\Device\Mup\server\share",
        r"\\?\GLOBALROOT\Device\P9Rdr\wsl$\Ubuntu",
        r"\\?\GLOBALROOT\Device\WebDavRedirector\server\share",
        r"\\?\GLOBALROOT\??\C:",
        r"\\?\GLOBALROOT\??\UNC\server\share",
        r"\\;LanmanRedirector\server\share",
        "//;lanmanredirector/server/share",
        r"\\;WebDavRedirector;CSC\server\share",
        r"\\;LanmanRedirector\server",
        r"\\;LanmanRedirector",
        r"\\;LanmanRedirector\;a\server\share",
        "//;lanmanredirector/;a/server/share",
        r"\\;LanmanRedirector\;a\;b\server\share",
        r"\\?\UNC\;LanmanRedirector\;a\server\share",
        r"\\?\GLOBALROOT\Device\Mup\;LanmanRedirector\server\share",
        r"\\?\GLOBALROOT\Device\LanmanRedirector\;a\server\share",
        r"C:\..",
        "C:/..",
        r"C:\folder\..\..",
        r"C:\a\b\..\..\..",
        "C://",
        r"C:\\..",
        "/..",
        "/a/../..",
        r"\\server\share\sub\..",
        r"\\server\share\..",
        r"\\server\share\docs\..\..",
        r"\\;LanmanRedirector\server\share\docs\..",
        r"\\;LanmanRedirector\;a\server\share\docs\..",
    ):
        cfg = MemoryConfig(folders=[raw])
        errors = validate_memory_config(cfg)
        assert len(errors) == 1, raw
        assert "filesystem root" in errors[0], raw


def test_validate_rejects_bare_drive_even_with_workspace_rebase() -> None:
    # "C:" is drive-relative (ntpath.isabs is false), so it rebases onto
    # workspace_cwd before the normalized-form root check — the raw-form
    # check must still catch it.
    cfg = MemoryConfig(folders=["C:"])
    errors = validate_memory_config(cfg, workspace_cwd="C:\\repo")
    assert len(errors) == 1
    assert "filesystem root" in errors[0]


def test_validate_accepts_unc_share_subfolder() -> None:
    # Only the host/share ROOT is rejected — a real folder under a share
    # remains a valid (foreign-or-native absolute) configuration, in both
    # the plain and extended-length ("\\\\?\\...") spellings.
    for raw in (
        r"\\server\share\docs",
        r"\\?\UNC\server\share\docs",
        "//?/unc/server/share/docs",
        r"\\?\C:\docs",
        r"\\.\C:\docs",
        r"\\.\UNC\server\share\docs",
        r"\\?\GLOBALROOT\Device\HarddiskVolume1\docs",
        r"\\?\GLOBALROOT\Device\HarddiskVolumeShadowCopy1\docs",
        r"\\?\GLOBALROOT\Device\LanmanRedirector\server\share\docs",
        r"\\?\GLOBALROOT\Device\P9Rdr\wsl$\Ubuntu\home",
        r"\\?\GLOBALROOT\??\C:\docs",
        r"\\?\Volume{d5588183-0d3f-4a24-a8ba-0c72e7b195e2}\docs",
        r"\\;LanmanRedirector\server\share\docs",
        "//;lanmanredirector/server/share/docs",
        r"\\;WebDavRedirector;CSC\server\share\docs",
        r"\\;LanmanRedirector\;a\server\share\docs",
        r"\\?\UNC\;LanmanRedirector\;a\server\share\docs",
        r"\\?\GLOBALROOT\Device\Mup\;LanmanRedirector\server\share\docs",
        r"\\?\GLOBALROOT\Device\LanmanRedirector\;a\server\share\docs",
        r"C:\folder\..\other",
        r"C:\..\docs",
        "C://docs",
        r"\\server\share\..\docs",
        "//server/share/../docs",
        r"\\server\share\a\..\b",
        r"\\;LanmanRedirector\server\share\..\docs",
        r"\\;LanmanRedirector\;a\server\share\..\docs",
    ):
        cfg = MemoryConfig(folders=[raw])
        assert validate_memory_config(cfg) == [], raw


def test_validate_flags_drive_separator_variant_duplicates() -> None:
    # "C://docs" and "C:\\docs" address the same Windows folder; the
    # drive-anchored tail must not inherit POSIX's two-leading-slash
    # special case, or separator variants dodge duplicate detection.
    cfg = MemoryConfig(folders=[r"C:\docs", "C://docs"])
    errors = validate_memory_config(cfg)
    assert len(errors) == 1
    assert "Memory Folder 2" in errors[0]
    assert "duplicates Folder 1" in errors[0]


def test_validate_flags_redirector_parent_variant_duplicates() -> None:
    # Leading ";" components are resolver directives, so the UNC anchor is
    # the server+share pair after them.  Parent traversal clamps there for
    # both the simple and chained-specifier spellings.
    for direct, parent_variant in (
        (
            r"\\;LanmanRedirector\server\share\docs",
            r"\\;LanmanRedirector\server\share\..\docs",
        ),
        (
            r"\\;LanmanRedirector\;a\server\share\docs",
            r"\\;LanmanRedirector\;a\server\share\..\docs",
        ),
    ):
        errors = validate_memory_config(MemoryConfig(folders=[direct, parent_variant]))
        assert len(errors) == 1
        assert "Memory Folder 2" in errors[0]
        assert "duplicates Folder 1" in errors[0]


def test_validate_root_folder_is_excluded_from_coverage_checks() -> None:
    # The rejected root must not additionally flag files as covered by it.
    cfg = MemoryConfig(files=["/note.md"], folders=["/"])
    errors = validate_memory_config(cfg)
    assert len(errors) == 1
    assert "filesystem root" in errors[0]


def test_root_folder_reported_missing_without_walk(tmp_path: Path) -> None:
    # The loader refuses filesystem roots before any is_dir()/walk — a
    # configured "/" (or bare drive / UNC share root) must not trigger a
    # whole-volume scan at agent construction.
    for raw in ("/", "C:", r"\\server\share", r"\\;LanmanRedirector\server\share"):
        result = load_memory_content(MemoryConfig(folders=[raw]), workspace_cwd=str(tmp_path))
        assert result.loaded_files == [], raw
        assert raw in result.missing_paths, raw


def test_validate_rejects_unsupported_extension() -> None:
    cfg = MemoryConfig(files=["spec.pdf"])
    errors = validate_memory_config(cfg)
    assert len(errors) == 1
    assert ".md or .txt" in errors[0]


def test_validate_accepts_uppercase_extension() -> None:
    """``.MD`` / ``.TXT`` are valid — extension match is case-insensitive."""
    cfg = MemoryConfig(files=["NOTES.MD", "log.TXT"])
    assert validate_memory_config(cfg) == []


def test_validate_flags_exact_duplicate_files() -> None:
    cfg = MemoryConfig(files=["spec.md", "spec.md"])
    errors = validate_memory_config(cfg)
    assert len(errors) == 1
    assert "Memory File 2" in errors[0]
    assert "duplicates File 1" in errors[0]


def test_validate_flags_case_insensitive_duplicate_files() -> None:
    """Catches the macOS gotcha where ``AGENTS.md`` and ``agents.md`` resolve
    to the same file on case-insensitive filesystems — and the user's
    intent is almost certainly a typo regardless of platform."""
    cfg = MemoryConfig(files=["AGENTS.md", "agents.md"])
    errors = validate_memory_config(cfg)
    assert len(errors) == 1
    assert "Memory File 2" in errors[0]
    assert "case-insensitively" in errors[0]


def test_validate_flags_normalized_duplicate_files() -> None:
    """``./docs/spec.md`` and ``docs/spec.md`` normalize to the same path."""
    cfg = MemoryConfig(files=["docs/spec.md", "./docs/spec.md"])
    errors = validate_memory_config(cfg)
    assert len(errors) == 1
    assert "Memory File 2" in errors[0]


def test_validate_flags_duplicate_folders() -> None:
    """Trailing slash and casing both collapse for folder dedup."""
    cfg = MemoryConfig(folders=["docs/", "DOCS"])
    errors = validate_memory_config(cfg)
    assert len(errors) == 1
    assert "Memory Folder 2" in errors[0]


def test_validate_allows_file_under_configured_folder() -> None:
    """Explicit files inside configured folders are deliberately allowed.

    The bounded folder scan (depth / file-count caps) does not guarantee
    the folder reaches a nested file, so the explicit entry is how users
    pin one; the loader deduplicates when both do load it.
    """
    for cfg in (
        MemoryConfig(files=["docs/spec.md"], folders=["docs"]),
        MemoryConfig(files=["docs/a/b/c/deep.md"], folders=["docs"]),
        MemoryConfig(files=["/var/notes/a.md"], folders=["/var/notes"]),
        MemoryConfig(files=["spec.md"], folders=["."]),
        MemoryConfig(files=["notes.md"], folders=["notes.md"]),
    ):
        assert validate_memory_config(cfg) == [], cfg


def test_validate_workspace_cwd_dedupes_absolute_and_relative_files() -> None:
    cfg = MemoryConfig(files=["docs/spec.md", "/repo/docs/spec.md"])
    errors = validate_memory_config(cfg, workspace_cwd="/repo")
    assert len(errors) == 1
    assert "Memory File 2" in errors[0]
    assert "duplicates File 1" in errors[0]


def test_validate_indices_align_with_list_position() -> None:
    """Blank entries still consume a slot so error labels match UI cards."""
    cfg = MemoryConfig(files=["", "spec.md", "spec.md"])
    errors = validate_memory_config(cfg)
    assert len(errors) == 1
    # Third card is the duplicate, regardless of the leading blank.
    assert "Memory File 3" in errors[0]
    assert "duplicates File 2" in errors[0]


def test_validate_allows_nested_folders() -> None:
    """``a`` and ``a/b`` overlap but are not duplicates — the loader
    dedupes any doubly-discovered file at load time instead."""
    cfg = MemoryConfig(files=["a/b/c.md"], folders=["a", "a/b"])
    assert validate_memory_config(cfg) == []


def test_validate_normalizes_backslash_separators() -> None:
    """A YAML profile authored on Windows may contain ``docs\\spec.md``.
    The validator must compare it equal to the POSIX form so duplicates
    still trigger across-platform."""
    cfg = MemoryConfig(files=["docs\\spec.md", "docs/spec.md"])
    errors = validate_memory_config(cfg)
    assert len(errors) == 1
    assert "duplicates File 1" in errors[0]


def test_validate_collects_multiple_independent_errors() -> None:
    """All issues should surface in one save round-trip, not one-at-a-time."""
    cfg = MemoryConfig(
        files=["spec.pdf", "spec.pdf"],
        folders=["docs", "docs"],
    )
    errors = validate_memory_config(cfg)
    # wrong extension (x2), duplicate file, duplicate folder.
    joined = "\n".join(errors)
    assert "Memory File 2" in joined  # duplicate file
    assert "duplicates File 1" in joined
    assert "Memory Folder 2" in joined  # duplicate folder
    assert any(".md or .txt" in e for e in errors)

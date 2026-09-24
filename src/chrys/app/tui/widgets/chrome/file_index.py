# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""In-memory search index for project path suggestions.

The index is built during file-suggestion warmup, not during synchronous input
handling.  Construction pre-normalizes each candidate once and query returns a
bounded top-k result with the same ordering semantics as ``fuzzy_filter``:
match position, original path length, then original path text.
"""

from __future__ import annotations

from bisect import bisect_left

from chrys.app.tui.widgets.chrome.file_scanner import (
    ProjectPathScanResult,
    ProjectPathSuggestion,
    _normalize_for_search,
)


class ProjectPathIndex:
    """Pre-normalized substring index for ``@`` file suggestions."""

    def __init__(self, scan: ProjectPathScanResult) -> None:
        self.root = scan.root
        self.file_count = scan.file_count
        self.suggestion_count = scan.suggestion_count
        self.truncated = scan.truncated
        self.file_budget = scan.file_budget
        self.suggestion_budget = scan.suggestion_budget
        self.source_truncations = dict(scan.source_truncations)
        self.ordering = scan.ordering
        self._suggestions = list(scan.paths)
        self._normalized = [_normalize_for_search(suggestion.path) for suggestion in self._suggestions]

    @classmethod
    def build(cls, scan: ProjectPathScanResult) -> ProjectPathIndex:
        """Build an index from a scan result."""
        return cls(scan)

    def head(self, *, limit: int = 30) -> list[ProjectPathSuggestion]:
        """Return the first suggestions in insertion order."""
        if limit <= 0:
            return []
        return self._suggestions[:limit]

    def query(self, query: str, *, limit: int = 30) -> list[ProjectPathSuggestion]:
        """Return exact current fuzzy-filter ordering for the best file matches."""
        if limit <= 0:
            return []
        if not query:
            return self.head(limit=limit)

        normalized_query = _normalize_for_search(query)
        best: list[tuple[tuple[int, int, str], int]] = []
        for index, normalized_path in enumerate(self._normalized):
            match_index = normalized_path.find(normalized_query)
            if match_index < 0:
                continue
            suggestion = self._suggestions[index]
            entry = ((match_index, len(suggestion.path), suggestion.path), index)
            if len(best) < limit:
                best.insert(bisect_left(best, entry), entry)
            elif entry < best[-1]:
                best.insert(bisect_left(best, entry), entry)
                best.pop()
        return [self._suggestions[index] for _score, index in best]

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""DocParser protocol — interface for document-to-Markdown converters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from chrys.foundation.text.images import IMAGE_DECODE_FORMATS

# Documents also embed scanned pages as TIFF and JPEG 2000, the formats pypdf
# hands such PDF images over in. Windows metafiles (EMF/WMF), which Pillow
# rasterizes through GDI on Windows only, are deliberately not decoded.
DOCUMENT_IMAGE_DECODE_FORMATS: tuple[str, ...] = (*IMAGE_DECODE_FORMATS, "TIFF", "JPEG2000")


@dataclass(frozen=True, slots=True)
class VisualOccurrence:
    """One document location that can be inspected through ``view_image``."""

    location: str
    ordinal: int
    reference: str
    source_name: str = ""


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Structured parser output shared by every supported document format."""

    markdown: str
    visuals: tuple[VisualOccurrence, ...] = ()
    warnings: tuple[str, ...] = ()


class DocumentImageSink(Protocol):
    """Per-conversion destination for bounded embedded-image artifacts."""

    def try_reserve_occurrence(self) -> bool:
        """Reserve one explicit image-candidate retrieval."""
        ...

    def record_unprocessed_occurrences(self, count: int) -> None:
        """Record candidates skipped after the retrieval budget is exhausted."""
        ...

    def record_candidate_failure(self) -> None:
        """Record one image candidate that could not be read."""
        ...

    def record_linked_image(self) -> None:
        """Record one linked-only image with no embedded bytes."""
        ...

    def save_image(
        self,
        data: bytes,
        *,
        location: str,
        ordinal: int,
        source_name: str,
    ) -> VisualOccurrence | None:
        """Normalize and persist one embedded image candidate."""
        ...

    @property
    def warnings(self) -> tuple[str, ...]:
        """Return bounded warning summaries accumulated so far."""
        ...


@runtime_checkable
class DocParser(Protocol):
    """Protocol for document-to-Markdown converters.

    Each implementation handles one or more file extensions and converts
    the document content to Markdown text.  Implementations should:

    - Lazy-import their dependencies inside ``parse()`` to avoid startup cost.
    - Produce Markdown with headings so TOC extraction works.
    - Raise ``ImportError`` with a clear install hint if dependencies are missing.
    """

    @property
    def supported_extensions(self) -> frozenset[str]:
        """File extensions this parser handles (e.g. ``{".pdf"}``)."""
        ...

    def parse(self, path: str, *, image_sink: DocumentImageSink | None = None) -> ParsedDocument:
        """Convert the document at *path* to structured Markdown output.

        This is a **blocking** call — the caller is responsible for
        running it in a thread via ``asyncio.to_thread()``.

        Returns:
            Parsed Markdown, visual occurrences, and bounded warnings.

        Raises:
            ImportError: If required libraries are not installed.
            Exception: On conversion failure.
        """
        ...

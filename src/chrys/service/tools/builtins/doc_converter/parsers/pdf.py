# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""PDF parser — converts PDF documents to Markdown using pypdf."""

from __future__ import annotations

from contextlib import suppress

from chrys.service.tools.builtins.doc_converter.parsers.base import (
    DocumentImageSink,
    ParsedDocument,
    VisualOccurrence,
)


class PdfParser:
    """Convert PDF files to Markdown via ``pypdf``."""

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".pdf"})

    def parse(self, path: str, *, image_sink: DocumentImageSink | None = None) -> ParsedDocument:
        from pypdf import PdfReader

        reader = PdfReader(path)
        parts: list[str] = []
        visuals: list[VisualOccurrence] = []
        seen_indirect: dict[object, VisualOccurrence] = {}
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                parts.append(f"# Page {i}\n\n{text}")
            else:
                parts.append(f"# Page {i}\n\n(no text content)")
            if image_sink is None:
                continue

            try:
                images = page.images
                image_keys = images.keys()
            except Exception:
                image_sink.record_candidate_failure()
                continue
            for candidate_index, image_key in enumerate(image_keys):
                if not image_sink.try_reserve_occurrence():
                    image_sink.record_unprocessed_occurrences(len(image_keys) - candidate_index)
                    break
                ordinal = candidate_index + 1
                try:
                    image = images[image_key]
                except Exception:
                    image_sink.record_candidate_failure()
                    continue
                if image.is_displayed is False:
                    continue

                indirect_reference = image.indirect_reference
                if indirect_reference is not None:
                    try:
                        prior = seen_indirect.get(indirect_reference)
                    except TypeError:
                        prior = None
                    if prior is not None:
                        visuals.append(
                            VisualOccurrence(
                                location=f"Page {i}",
                                ordinal=ordinal,
                                reference=prior.reference,
                                source_name=image.name or prior.source_name,
                            )
                        )
                        continue

                occurrence = image_sink.save_image(
                    image.data,
                    location=f"Page {i}",
                    ordinal=ordinal,
                    source_name=image.name,
                )
                if occurrence is None:
                    continue
                visuals.append(occurrence)
                if indirect_reference is not None:
                    with suppress(TypeError):
                        seen_indirect[indirect_reference] = occurrence
        return ParsedDocument(
            markdown="\n\n".join(parts),
            visuals=tuple(visuals),
            warnings=image_sink.warnings if image_sink is not None else (),
        )

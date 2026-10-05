# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""PDF parser — converts PDF documents to Markdown using pypdf."""

from __future__ import annotations

from contextlib import suppress
from types import ModuleType
from typing import IO, Any

from chrys.service.tools.builtins.doc_converter.parsers.base import (
    DOCUMENT_IMAGE_DECODE_FORMATS,
    DocumentImageSink,
    ParsedDocument,
    VisualOccurrence,
)

# What an open that names its formats may keep: a format a later pypdf names is
# dropped, not decoded, until it is reviewed here.
_PYPDF_NAMED_FORMATS = (*DOCUMENT_IMAGE_DECODE_FORMATS, "PPM")


class _DocumentFormatsImage:
    """``PIL.Image`` as pypdf's image extraction sees it: ``open`` decodes document formats (and PPM) only.

    pypdf opens a ``/DCTDecode`` stream with every Pillow decoder, whatever its
    bytes are (EPS among them, which runs Ghostscript), and re-encodes the
    result as JPEG. Its other opens name the formats they expect; those keep
    PPM too, which its ``/JBIG2Decode`` open accepts because jbig2dec built
    without libpng writes PBM even when asked for PNG.
    """

    def __init__(self, image_module: ModuleType) -> None:
        self._image_module = image_module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._image_module, name)

    def open(self, fp: str | IO[bytes], mode: str = "r", formats: list[str] | tuple[str, ...] | None = None) -> Any:
        allowed = (
            DOCUMENT_IMAGE_DECODE_FORMATS
            if formats is None
            else tuple(name for name in formats if name in _PYPDF_NAMED_FORMATS)
        )
        return self._image_module.open(fp, mode, allowed)


def _restrict_pypdf_image_decoders() -> None:
    """Route pypdf's image extraction through ``_DocumentFormatsImage``; idempotent.

    Raises instead of extracting with every decoder when a pypdf upgrade no
    longer decodes through the module-level ``PIL.Image`` this replaces.
    """
    from PIL import Image
    from pypdf.generic import _image_xobject

    current = _image_xobject.Image
    if isinstance(current, _DocumentFormatsImage):
        return
    if current is not Image:
        raise RuntimeError("pypdf no longer decodes images through PIL.Image; restrict its decoders again")
    # Through the namespace: the attribute is typed as the PIL.Image module itself.
    vars(_image_xobject)["Image"] = _DocumentFormatsImage(Image)


class PdfParser:
    """Convert PDF files to Markdown via ``pypdf``."""

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".pdf"})

    def parse(self, path: str, *, image_sink: DocumentImageSink | None = None) -> ParsedDocument:
        from pypdf import PdfReader

        if image_sink is not None:
            _restrict_pypdf_image_decoders()
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

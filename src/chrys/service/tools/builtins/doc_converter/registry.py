# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Parser registry — maps file extensions to DocParser implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chrys.service.tools.builtins.doc_converter.parsers.base import DocParser

# Lazy-loaded singleton registry: extension → parser instance.
_registry: dict[str, DocParser] | None = None


def _build_registry() -> dict[str, DocParser]:
    """Instantiate all parsers and build the extension → parser map.

    Parsers are imported here (module level is cheap — heavy deps like
    ``pypdf`` are only imported inside each parser's ``parse()`` method).
    """
    from chrys.service.tools.builtins.doc_converter.parsers.docx import DocxParser
    from chrys.service.tools.builtins.doc_converter.parsers.pdf import PdfParser
    from chrys.service.tools.builtins.doc_converter.parsers.pptx import PptxParser
    from chrys.service.tools.builtins.doc_converter.parsers.xls import XlsParser
    from chrys.service.tools.builtins.doc_converter.parsers.xlsx import XlsxParser

    parsers: list[DocParser] = [
        PdfParser(),
        DocxParser(),
        PptxParser(),
        XlsxParser(),
        XlsParser(),
    ]
    mapping: dict[str, DocParser] = {}
    for parser in parsers:
        for ext in parser.supported_extensions:
            mapping[ext] = parser
    return mapping


def get_parser(ext: str) -> DocParser | None:
    """Return the parser for the given file extension, or ``None``."""
    global _registry
    if _registry is None:
        _registry = _build_registry()
    return _registry.get(ext.lower())


def supported_extensions() -> frozenset[str]:
    """Return all supported file extensions."""
    global _registry
    if _registry is None:
        _registry = _build_registry()
    return frozenset(_registry.keys())

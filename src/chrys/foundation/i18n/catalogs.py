# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Class-based GNU gettext catalog loading."""

from __future__ import annotations

import gettext
import struct
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

type CatalogRoot = str | Path | Traversable


class CatalogLoadError(Exception):
    """A requested GNU MO catalog could not be loaded safely."""


def load_catalog(
    locale_name: str,
    *,
    catalog_root: CatalogRoot | None = None,
    domain: str = "chrys",
) -> gettext.GNUTranslations:
    """Load one locale's MO as an isolated ``GNUTranslations`` instance."""
    root = _catalog_root(catalog_root)
    catalog_path = root / locale_name / "LC_MESSAGES" / f"{domain}.mo"
    try:
        with catalog_path.open("rb") as stream:
            catalog = gettext.GNUTranslations(stream)
    except (OSError, EOFError, LookupError, SyntaxError, ValueError, struct.error) as error:
        raise CatalogLoadError("The i18n catalog could not be loaded.") from error
    return catalog


def _catalog_root(configured: CatalogRoot | None) -> Traversable:
    if configured is None:
        return resources.files("chrys.foundation.i18n") / "_catalogs"
    if isinstance(configured, str):
        return Path(configured)
    return configured

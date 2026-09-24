# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared helpers for the catalog artifact tests: tmp-tree builders, PO/MO readers, and the PO-to-MO semantic gate."""

from __future__ import annotations

import gettext
import io
import json
from pathlib import Path

from babel.messages.catalog import Catalog
from babel.messages.mofile import read_mo, write_mo
from babel.messages.pofile import read_po, write_po
from scripts import i18n

from chrys.foundation.i18n.formatting import has_visible_content


def _source_root(tmp_path: Path, source: str) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    (root / "messages.py").write_text(source, encoding="utf-8")
    return root


def _catalog_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    return (
        tmp_path / "locales" / "chrys.pot",
        tmp_path / "locales" / "zh-Hans" / "LC_MESSAGES" / "chrys.po",
        tmp_path / "catalogs" / "zh-Hans" / "LC_MESSAGES" / "chrys.mo",
    )


def _read_catalog(path: Path) -> Catalog:
    with path.open("rb") as stream:
        return read_po(stream, domain="chrys", abort_invalid=True)


def _write_catalog(path: Path, catalog: Catalog) -> None:
    with path.open("wb") as stream:
        write_po(stream, catalog, width=0, sort_output=True, include_lineno=True)


def _message(catalog: Catalog, key: str):
    message = catalog.get(key)
    assert message is not None
    return message


def _prepare_translated_catalog(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source_root = _source_root(
        tmp_path,
        "from chrys.foundation.i18n import msg\n"
        "CLOSE = msg('dialog.close', fallback='Close {name}')\n"
        "FILES = msg(\n"
        "    'dialog.files',\n"
        "    fallback='One {name}',\n"
        "    plural_fallback='{count} {name}s',\n"
        ")\n"
        "EMPTY = msg('dialog.empty', fallback='Untranslated')\n"
        "FUZZY = msg('dialog.fuzzy', fallback='Review me')\n",
    )
    pot_path, po_path, mo_path = _catalog_paths(tmp_path)
    i18n.extract_catalog(source_root=source_root, pot_path=pot_path, location_root=source_root)
    i18n.update_catalog(source_root=source_root, po_path=po_path, location_root=source_root)
    catalog = _read_catalog(po_path)
    _message(catalog, "dialog.close").string = "关闭 {name}"
    _message(catalog, "dialog.files").string = ("{count} 个 {name}",)
    fuzzy = _message(catalog, "dialog.fuzzy")
    fuzzy.string = "需要复核"
    fuzzy.flags.add("fuzzy")
    _write_catalog(po_path, catalog)
    i18n.compile_catalog(
        source_root=source_root,
        pot_path=pot_path,
        po_path=po_path,
        mo_path=mo_path,
        location_root=source_root,
    )
    return source_root, pot_path, po_path, mo_path


_SEMANTIC_METADATA_HEADERS = (
    "Project-Id-Version",
    "Report-Msgid-Bugs-To",
    "POT-Creation-Date",
    "PO-Revision-Date",
    "Last-Translator",
    "Language",
    "Language-Team",
    "Plural-Forms",
    "MIME-Version",
    "Content-Type",
    "Content-Transfer-Encoding",
    "Generated-By",
)


def _normalized_catalog_metadata(catalog: Catalog) -> dict[str, str | int]:
    headers = {name.casefold(): " ".join(value.split()) for name, value in catalog.mime_headers}
    report_address = "report-msgid-bugs-to"
    if headers.get(report_address) == "EMAIL@ADDRESS":
        # Babel's public MO reader synthesizes its default for the PO's empty
        # value, so these two public-reader representations are equivalent.
        headers[report_address] = ""
    normalized = {name.casefold(): headers.get(name.casefold(), "") for name in _SEMANTIC_METADATA_HEADERS}
    normalized.update(
        {
            "charset": (catalog.charset or "").casefold(),
            "locale": str(catalog.locale),
            "num-plurals": catalog.num_plurals,
            "plural-expression": " ".join(catalog.plural_expr.split()),
        }
    )
    return normalized


def _entry_forms(message) -> tuple[str, ...]:
    if isinstance(message.string, list):
        return tuple(form or "" for form in message.string)
    return i18n._translation_forms(message)


def _entry_id(message) -> str | tuple[str, str]:
    return message.id if isinstance(message.id, str) else tuple(message.id)


def _effective_catalog_entries(catalog: Catalog) -> dict[str | tuple[str, str], tuple[str, ...]]:
    effective: dict[str | tuple[str, str], tuple[str, ...]] = {}
    for message in catalog:
        if not message.id or message.fuzzy:
            continue
        forms = _entry_forms(message)
        if forms and all(has_visible_content(form) for form in forms):
            effective[_entry_id(message)] = forms
    return effective


def _mo_catalog_entries(catalog: Catalog) -> dict[str | tuple[str, str], tuple[str, ...]]:
    # The compiler only emits effective entries, so an MO row without visible
    # content in every form is a violation in its own right — filtering it out
    # the way the PO side does would let the runtime serve invisible text.
    entries: dict[str | tuple[str, str], tuple[str, ...]] = {}
    for message in catalog:
        if not message.id:
            continue
        forms = _entry_forms(message)
        assert forms and all(has_visible_content(form) for form in forms), (
            "MO entries must all carry visible translation content"
        )
        entries[_entry_id(message)] = forms
    return entries


def _plural_representative_counts(catalog: Catalog) -> dict[int, int]:
    plural_index = gettext.c2py(catalog.plural_expr)
    representatives: dict[int, int] = {}
    for count in range(1001):
        index = plural_index(count)
        if 0 <= index < catalog.num_plurals:
            representatives.setdefault(index, count)
        if len(representatives) == catalog.num_plurals:
            break
    assert set(representatives) == set(range(catalog.num_plurals)), "plural indexes need representative counts"
    return representatives


def _assert_po_mo_semantically_consistent(
    *,
    source_root: Path,
    pot_path: Path,
    po_path: Path,
    mo_path: Path,
) -> None:
    # No explicit ``location_root``: extraction derives it from *source_root*,
    # which is the only spelling that is right for both callers — a tmp tree
    # roots at itself, the live tree roots at the repo, and the tracked
    # catalogs record ``src/chrys/...`` paths that only the latter produces.
    _messages, po = i18n.check_catalogs(
        source_root=source_root,
        pot_path=pot_path,
        po_path=po_path,
    )
    try:
        mo_bytes = mo_path.read_bytes()
        mo = read_mo(io.BytesIO(mo_bytes))
        runtime = gettext.GNUTranslations(io.BytesIO(mo_bytes))
    except (OSError, EOFError, LookupError, RuntimeError, SyntaxError, TypeError, ValueError) as error:
        raise AssertionError("tracked MO must be loadable by Babel and stdlib gettext") from error

    assert all(message.context is None for message in po if message.id), "PO context entries are forbidden"
    assert all(message.context is None for message in mo if message.id), "MO context entries are forbidden"
    assert _normalized_catalog_metadata(mo) == _normalized_catalog_metadata(po), "catalog metadata is stale"

    # Babel's MO reader synthesizes defaults for missing header fields, so the
    # parsed-catalog comparison above cannot prove the header block is
    # physically present; stdlib's own parse of the MO header is the physical
    # truth the runtime reads.
    po_headers = {name.casefold(): " ".join(value.split()) for name, value in po.mime_headers}
    mo_physical_headers = {name.casefold(): " ".join(value.split()) for name, value in runtime.info().items()}
    for name in _SEMANTIC_METADATA_HEADERS:
        expected = po_headers.get(name.casefold(), "")
        if expected:
            assert mo_physical_headers.get(name.casefold()) == expected, (
                f"MO physical header {name} is missing or does not match the PO"
            )

    po_entries = _effective_catalog_entries(po)
    mo_entries = _mo_catalog_entries(mo)
    assert set(mo_entries) == set(po_entries), "MO effective entry set does not match the PO"
    assert mo_entries == po_entries, "MO translation content is stale relative to the PO"

    plural_forms = tuple(field.strip() for field in runtime.info().get("plural-forms", "").split(";") if field.strip())
    assert plural_forms == (f"nplurals={po.num_plurals}", f"plural={po.plural_expr}"), (
        "stdlib gettext plural metadata does not match the PO"
    )
    representatives = _plural_representative_counts(po)
    for message_id, forms in po_entries.items():
        if isinstance(message_id, tuple):
            singular_id, plural_id = message_id
            for index, count in representatives.items():
                assert runtime.ngettext(singular_id, plural_id, count) == forms[index]
        else:
            assert runtime.gettext(message_id) == forms[0]


def _read_mo_catalog(path: Path) -> Catalog:
    with path.open("rb") as stream:
        return read_mo(stream)


def _write_mo_catalog(path: Path, catalog: Catalog) -> None:
    with path.open("wb") as stream:
        write_mo(stream, catalog, use_fuzzy=False)


def _mutate_first_metadata_comment(po_path: Path, respell) -> None:
    text = po_path.read_text(encoding="utf-8")
    line = next(candidate for candidate in text.splitlines() if candidate.startswith("#. chrys-meta={"))
    data = json.loads(line[len("#. chrys-meta=") :])
    po_path.write_text(text.replace(line, "#. chrys-meta=" + respell(data), 1), encoding="utf-8")

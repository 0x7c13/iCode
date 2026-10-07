# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Src rule: Pillow decodes only the formats its caller names, plus its red/green proofs and allowlist pin."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.architecture._hygiene_core import _meta_guard_problem, _pins_allowlist, _qualified_name, _src_sources, _tree
from tests.support.ci import CI_LINUX_ONLY

# Platform-independent source analysis: the Linux CI job covers it.
pytestmark = CI_LINUX_ONLY

_IMAGE_OPEN = "PIL.Image.open"
# Pillow entry points that choose a decoder from the bytes alone and take no
# formats argument to narrow it.
_ANY_FORMAT_DECODERS = frozenset({"PIL.ImageFile.Parser", "PIL.ImageGrab.grabclipboard"})
# "<path>::<qualname>" -> why that function may reach an any-format decoder.
_PILLOW_ANY_FORMAT_ALLOWLIST: dict[str, str] = {
    "src/chrys/app/tui/widgets/chrome/image_paste.py::_grab_windows_clipboard": (
        "Windows only: there grabclipboard builds the PNG or DIB image itself instead of opening the bytes"
    ),
}


_Scope = ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
# Scope (by id) -> name -> the dotted targets that scope's own imports bind the name to.
_Bindings = dict[int, dict[str, set[str]]]


def _imported_names(node: ast.Import | ast.ImportFrom) -> Iterator[tuple[str, str]]:
    """Yield ``(name, dotted target)`` for each name an absolute import binds."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.asname:
                yield alias.asname, alias.name
            else:
                top = alias.name.partition(".")[0]
                yield top, top
    elif node.level == 0 and node.module:
        for alias in node.names:
            yield alias.asname or alias.name, f"{node.module}.{alias.name}"


def _scoped_nodes(node: ast.AST, scopes: tuple[_Scope, ...]) -> Iterator[tuple[ast.AST, tuple[_Scope, ...]]]:
    """Yield every node below *node* with the scopes enclosing it, the module first."""
    for child in ast.iter_child_nodes(node):
        yield child, scopes
        inner = (*scopes, child) if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) else scopes
        yield from _scoped_nodes(child, inner)


def _resolved(node: ast.expr, scopes: tuple[_Scope, ...], bindings: _Bindings) -> set[str]:
    """The dotted targets *node* may name through the imports visible where it stands.

    The innermost scope importing the name decides, as in Python's lookup (a
    function does not see the class body it is nested in); when that scope
    imports the name twice, either import may be the one in effect.
    """
    root = node
    while isinstance(root, ast.Attribute):
        root = root.value
    if not isinstance(root, ast.Name):
        return set()
    head, _, rest = _qualified_name(node).partition(".")
    for depth, scope in enumerate(reversed(scopes)):
        if depth and isinstance(scope, ast.ClassDef):
            continue
        targets = bindings.get(id(scope), {}).get(head)
        if targets:
            return {f"{target}.{rest}" if rest else target for target in targets}
    return set()


def _formats_value(call: ast.Call, *, positional_index: int | None) -> ast.expr | None:
    if positional_index is not None and len(call.args) > positional_index:
        return call.args[positional_index]
    return next((keyword.value for keyword in call.keywords if keyword.arg == "formats"), None)


def _names_formats(value: ast.expr | None) -> bool:
    return value is not None and not (isinstance(value, ast.Constant) and value.value is None)


def _pillow_decoder_sites(path: Path, source: str) -> Iterator[tuple[ast.expr, str, str, ast.AST | None]]:
    """Yield ``(node, target, qualname, parent)`` for each reference to ``Image.open`` or an any-format decoder."""
    tree = _tree(path, source)
    if not any(
        isinstance(node, ast.Import | ast.ImportFrom)
        and any(target == "PIL" or target.startswith("PIL.") for _name, target in _imported_names(node))
        for node in ast.walk(tree)
    ):
        return
    scoped = list(_scoped_nodes(tree, (tree,)))
    bindings: _Bindings = {}
    for node, scopes in scoped:
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = bindings.setdefault(id(scopes[-1]), {})
            for name, target in _imported_names(node):
                names.setdefault(name, set()).add(target)
    parents: dict[int, ast.AST] = {
        id(child): parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
    }
    for node, scopes in scoped:
        if not isinstance(node, ast.Name | ast.Attribute):
            continue
        for target in sorted(_resolved(node, scopes, bindings)):
            if target == _IMAGE_OPEN or target in _ANY_FORMAT_DECODERS:
                qualname = ".".join(scope.name for scope in scopes if not isinstance(scope, ast.Module))
                yield node, target, qualname, parents.get(id(node))


def _open_reference_names_formats(node: ast.expr, parent: ast.AST | None) -> bool:
    """Whether this ``Image.open`` reference is called with formats, or handed on together with them.

    ``asyncio.to_thread(Image.open, path, formats=...)`` and
    ``functools.partial(Image.open, formats=...)`` forward the keyword to it.
    """
    if not isinstance(parent, ast.Call):
        return False
    if parent.func is node:
        return _names_formats(_formats_value(parent, positional_index=2))
    return (
        bool(parent.args) and parent.args[0] is node and _names_formats(_formats_value(parent, positional_index=None))
    )


def _assert_pillow_decodes_only_named_formats(sources: dict[Path, str]) -> None:
    violations: list[str] = []
    for path, source in sources.items():
        for node, target, qualname, parent in _pillow_decoder_sites(path, source):
            if target == _IMAGE_OPEN:
                if not _open_reference_names_formats(node, parent):
                    violations.append(
                        f"{path.as_posix()}:{node.lineno}: Image.open must pass formats= naming the formats this "
                        "caller decodes (attached or viewed images: foundation/text/images.py::IMAGE_DECODE_FORMATS); "
                        "without it every Pillow decoder reads the bytes"
                    )
            elif f"{path.as_posix()}::{qualname}" not in _PILLOW_ANY_FORMAT_ALLOWLIST:
                violations.append(
                    f"{path.as_posix()}:{node.lineno}: {target} picks a decoder from the bytes alone and takes no "
                    "formats; read the bytes yourself and call Image.open(..., formats=...), or allowlist the "
                    "function in _PILLOW_ANY_FORMAT_ALLOWLIST with the reason its input is safe"
                )
    assert not violations, "\n".join(violations)


# Sibling functions importing different ``Image`` names: neither import reaches the other function.
_PILLOW_LOADER = "def load_raster(data):\n    from PIL import Image\n\n    return Image.open(data)\n"
_SLIDE_LOADER = "def load_slide_image(data):\n    from pptx.parts.image import Image\n\n    return Image.open(data)\n"
_NAMED_PILLOW_LOADER = _PILLOW_LOADER.replace("Image.open(data)", "Image.open(data, formats=FORMATS)")


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("from PIL import Image\nImage.open(data)\n", "Image.open must pass formats="),
        ("from PIL import Image as PILImage\nPILImage.open(data, formats=None)\n", "Image.open must pass formats="),
        ("import PIL.Image\nPIL.Image.open(data)\n", "Image.open must pass formats="),
        ("import PIL.Image as pil_image\npil_image.open(data, 'r')\n", "Image.open must pass formats="),
        ("from PIL.Image import open as open_image\nopen_image(data)\n", "Image.open must pass formats="),
        (
            "def load(path):\n    from PIL import Image\n\n    with Image.open(path) as opened:\n        return opened.size\n",
            "Image.open must pass formats=",
        ),
        ("from PIL import Image\nImage.open(data, **options)\n", "Image.open must pass formats="),
        (
            (
                "import asyncio\nfrom PIL import Image\n\n\nasync def load(path):\n"
                "    return await asyncio.to_thread(Image.open, path)\n"
            ),
            "Image.open must pass formats=",
        ),
        (
            "import functools\nfrom PIL import Image\nload = functools.partial(Image.open)\n",
            "Image.open must pass formats=",
        ),
        (
            "from PIL import Image\nopen_image = Image.open\nopen_image(data, formats=FORMATS)\n",
            "Image.open must pass formats=",
        ),
        ("from PIL import Image\n\n\ndef load(data):\n    return Image.open(data)\n", "Image.open must pass formats="),
        (_PILLOW_LOADER + "\n\n" + _SLIDE_LOADER, "Image.open must pass formats="),
        (_SLIDE_LOADER + "\n\n" + _PILLOW_LOADER, "Image.open must pass formats="),
        (
            (
                "def load(data):\n    from PIL import Image\n    from pptx.parts.image import Image\n\n"
                "    return Image.open(data)\n"
            ),
            "Image.open must pass formats=",
        ),
        (
            (
                "from PIL import Image\n\n\nclass Slides:\n    from pptx.parts.image import Image\n\n"
                "    def load(self, data):\n        return Image.open(data)\n"
            ),
            "Image.open must pass formats=",
        ),
        ("class Loader:\n    from PIL import Image\n\n    probe = Image.open(DATA)\n", "Image.open must pass formats="),
        ("from PIL import ImageFile\nparser = ImageFile.Parser()\n", "PIL.ImageFile.Parser picks a decoder"),
        ("from PIL.ImageFile import Parser\nparser = Parser()\n", "PIL.ImageFile.Parser picks a decoder"),
        (
            "def grab():\n    from PIL import ImageGrab\n\n    return ImageGrab.grabclipboard()\n",
            "PIL.ImageGrab.grabclipboard picks a decoder",
        ),
        ("import PIL.ImageGrab\nPIL.ImageGrab.grabclipboard()\n", "PIL.ImageGrab.grabclipboard picks a decoder"),
    ],
)
def test_pillow_formats_guard_rejects_decoding_without_named_formats(source: str, message: str) -> None:
    with pytest.raises(AssertionError, match=rf"src/example\.py:\d+: {re.escape(message)}"):
        _assert_pillow_decodes_only_named_formats({Path("src/example.py"): source})


def test_pillow_formats_guard_allows_named_formats_and_unrelated_opens() -> None:
    _assert_pillow_decodes_only_named_formats(
        {
            Path("src/example.py"): """import asyncio
import functools

from PIL import Image

Image.open(data, formats=("PNG",))
Image.open(data, "r", FORMATS)
open(path, encoding="utf-8")
path.open("rb")
load = functools.partial(Image.open, formats=FORMATS)


async def load_later(path):
    return await asyncio.to_thread(Image.open, path, formats=FORMATS)
""",
            Path("src/unrelated.py"): "import zipfile\nzipfile.ZipFile(path).open(name)\n",
            Path("src/pillow_then_slides.py"): _NAMED_PILLOW_LOADER + "\n\n" + _SLIDE_LOADER,
            Path("src/slides_then_pillow.py"): _SLIDE_LOADER + "\n\n" + _NAMED_PILLOW_LOADER,
            Path("src/class_body.py"): (
                "from PIL import Image\n\n\nclass Slides:\n    from pptx.parts.image import Image\n\n"
                "    probe = Image.open(DATA)\n"
            ),
            Path("src/shadowed.py"): (
                "from PIL import Image\n\n\ndef load_slide_image(data):\n"
                "    from pptx.parts.image import Image\n\n    return Image.open(data)\n"
            ),
        }
    )


def test_pillow_formats_guard_allows_an_allowlisted_any_format_decoder() -> None:
    path, qualname = next(iter(_PILLOW_ANY_FORMAT_ALLOWLIST)).split("::")
    source = f"def {qualname}():\n    from PIL import ImageGrab\n\n    return ImageGrab.grabclipboard()\n"

    _assert_pillow_decodes_only_named_formats({Path(path): source})
    with pytest.raises(AssertionError, match=re.escape("PIL.ImageGrab.grabclipboard picks a decoder")):
        _assert_pillow_decodes_only_named_formats({Path(path): source.replace(qualname, f"{qualname}_elsewhere")})


@_pins_allowlist("_PILLOW_ANY_FORMAT_ALLOWLIST")
def test_pillow_any_format_allowlist_entries_are_live() -> None:
    """Each entry must still name a function that reaches an any-format decoder: a stale key would
    silently vouch for a new call added later under that name."""
    live = {
        f"{path.as_posix()}::{qualname}"
        for path, source in _src_sources().items()
        for _node, target, qualname, _parent in _pillow_decoder_sites(path, source)
        if target in _ANY_FORMAT_DECODERS
    }
    problems = [
        _meta_guard_problem(
            "_PILLOW_ANY_FORMAT_ALLOWLIST",
            f"entry {key} names no function that reaches an any-format Pillow decoder",
            "remove the stale entry or re-key it to the function that now reaches the decoder",
        )
        for key in sorted(set(_PILLOW_ANY_FORMAT_ALLOWLIST) - live)
    ]
    assert problems == [], "\n".join(problems)

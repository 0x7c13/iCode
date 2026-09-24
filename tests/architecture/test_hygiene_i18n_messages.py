# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Src rule: msg() message construction must follow the canonical i18n shape, plus its proofs."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path

import pytest

from tests.architecture._hygiene_core import _qualified_name, _resolved_import_module, _tree
from tests.support.ci import CI_LINUX_ONLY

# Platform-independent source analysis: the Linux CI job covers it.
pytestmark = CI_LINUX_ONLY


_I18N_MESSAGES_PATH = Path("src/chrys/foundation/i18n/messages.py")


_I18N_INIT_PATH = Path("src/chrys/foundation/i18n/__init__.py")


def _assert_i18n_message_construction_is_canonical(sources: Mapping[Path, str]) -> None:
    """Keep every extractable message on the one AST-visible construction path."""
    violations: list[str] = []
    for path, source in sources.items():
        if path == _I18N_MESSAGES_PATH:
            continue
        tree = _tree(path, source)
        parents = {id(child): parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        # Ordinals, not line numbers: semicolon-joined statements share a
        # line while still executing strictly in statement order.
        top_level_ordinal = {
            id(descendant): index for index, statement in enumerate(tree.body) for descendant in ast.walk(statement)
        }
        canonical_import_ordinal: int | None = None
        i18n_msg_import = False
        i18n_module_refs: set[str] = set()
        i18n_module_bindings: set[str] = set()
        local_message_definitions: set[str] = set()
        dataclasses_module_names = {"dataclasses"}
        replace_function_names = {"replace"}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # An unaliased plain import binds its ROOT component, so
                    # ``import msg.submodule`` rebinds msg.
                    if (alias.asname or alias.name.split(".", maxsplit=1)[0]) == "msg":
                        violations.append(f"{path}:{node.lineno}: msg may not be locally defined or rebound")
                    elif alias.name == "chrys.foundation.i18n" or alias.name.startswith("chrys.foundation.i18n."):
                        if alias.asname is not None:
                            i18n_module_refs.add(alias.asname)
                            i18n_module_bindings.add(alias.asname)
                        else:
                            # An unaliased deep import makes every prefix
                            # module a live dotted value too.
                            name = alias.name
                            while name.startswith("chrys.foundation.i18n"):
                                i18n_module_refs.add(name)
                                name = name.rsplit(".", maxsplit=1)[0]
                    elif alias.name == "dataclasses" and alias.asname is not None:
                        dataclasses_module_names.add(alias.asname)
            elif isinstance(node, ast.ImportFrom):
                module = _resolved_import_module(path, node)
                i18n_family = module is not None and module.startswith("chrys.foundation.i18n")
                for alias in node.names:
                    # Any import binding the name msg other than an i18n
                    # msg-member import (e.g. MessageDef as msg) is a rebind,
                    # whatever the source module.
                    if (alias.asname or alias.name) == "msg" and not (i18n_family and alias.name in {"msg", "*"}):
                        violations.append(f"{path}:{node.lineno}: msg may not be locally defined or rebound")
                        continue
                    if module == "chrys.foundation":
                        if alias.name == "i18n":
                            i18n_module_refs.add(alias.asname or alias.name)
                            i18n_module_bindings.add(alias.asname or alias.name)
                    elif module == "dataclasses":
                        if alias.name == "replace":
                            replace_function_names.add(alias.asname or alias.name)
                    elif i18n_family:
                        if alias.name not in {"msg", "*"}:
                            # Submodule and member aliases become owner refs so
                            # qualified constructor calls through them (e.g.
                            # messages.msg) are caught below.
                            i18n_module_refs.add(alias.asname or alias.name)
                            continue
                        package_reexport = (
                            path == _I18N_INIT_PATH and node.level == 0 and module == "chrys.foundation.i18n.messages"
                        )
                        # Only an unconditional top-level import is canonical:
                        # a conditional one can lose to a rogue same-name
                        # binding at runtime.
                        canonical = (
                            node.level == 0
                            and module == "chrys.foundation.i18n"
                            and alias.asname is None
                            and node in tree.body
                        )
                        if package_reexport and alias.asname is None:
                            continue
                        i18n_msg_import = True
                        if canonical and alias.name == "msg":
                            ordinal = top_level_ordinal[id(node)]
                            if canonical_import_ordinal is None or ordinal < canonical_import_ordinal:
                                canonical_import_ordinal = ordinal
                        else:
                            violations.append(
                                f"{path}:{node.lineno}: import msg only with 'from chrys.foundation.i18n import msg'"
                            )

        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets = node.targets
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
                value = node.value
            else:
                continue
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "msg"
                and len(targets) == 1
                and isinstance(targets[0], ast.Name)
            ):
                local_message_definitions.add(targets[0].id)

        for node in ast.walk(tree):
            if (
                (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.MatchAs, ast.MatchStar))
                    and node.name == "msg"
                )
                or (isinstance(node, ast.MatchMapping) and node.rest == "msg")
                or (
                    # ExceptHandler.name is a plain string, invisible to Name
                    # Store/Del checks.
                    isinstance(node, ast.ExceptHandler) and node.name == "msg"
                )
            ):
                violations.append(f"{path}:{node.lineno}: msg may not be locally defined or rebound")
            if i18n_msg_import and isinstance(node, ast.arg) and node.arg == "msg":
                violations.append(f"{path}:{node.lineno}: msg may not be passed or rebound as a local name")
            if isinstance(node, ast.Name) and node.id in i18n_module_bindings:
                # Module bindings may only serve qualified attribute access;
                # aliasing one to a new name would launder later .msg calls.
                parent = parents.get(id(node))
                if not (isinstance(parent, ast.Attribute) and parent.value is node):
                    violations.append(
                        f"{path}:{node.lineno}: i18n module references may not be aliased or passed as values"
                    )
            if isinstance(node, ast.Attribute):
                # The dotted spelling of an unaliased plain import launders
                # the same way a name binding would; deep imports record every
                # i18n-namespace prefix as a ref, while member tails
                # (….messages.MessageDef) stay legal for annotations.
                qualified_ref = _qualified_name(node)
                if "." in qualified_ref and qualified_ref in i18n_module_refs:
                    parent = parents.get(id(node))
                    if not (isinstance(parent, ast.Attribute) and parent.value is node):
                        violations.append(
                            f"{path}:{node.lineno}: i18n module references may not be aliased or passed as values"
                        )

            if isinstance(node, ast.Attribute) and node.attr == "msg":
                # Any access — not just a direct call — or the attribute can
                # be aliased to a local factory and called untracked.
                owner = _qualified_name(node.value)
                if _is_i18n_owner(owner, i18n_module_refs):
                    violations.append(
                        f"{path}:{node.lineno}: call msg as a bare name, never through a module or object"
                    )

            if not isinstance(node, ast.Call):
                continue
            qualified = _qualified_name(node.func)
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "msg"
                and (
                    canonical_import_ordinal is None
                    # A call in a statement before the import raises NameError
                    # at runtime while extraction would still record it.
                    or top_level_ordinal.get(id(node), -1) < canonical_import_ordinal
                    or not _is_module_level_message_assignment(node, parents, tree)
                )
            ):
                violations.append(
                    f"{path}:{node.lineno}: msg() must follow the canonical import and be a direct "
                    "module-level assignment value"
                )
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in {"msg", "MessageDef", "MessageRef"}
            ):
                owner = _qualified_name(node.args[0])
                if _is_i18n_owner(owner, i18n_module_refs):
                    violations.append(
                        f"{path}:{node.lineno}: access message constructors as imported names, never via getattr"
                    )
            if qualified.rsplit(".", maxsplit=1)[-1] in {"MessageDef", "MessageRef"}:
                violations.append(f"{path}:{node.lineno}: construct messages only through msg() and MessageDef.bind()")
            is_replace_call = (isinstance(node.func, ast.Name) and node.func.id in replace_function_names) or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "replace"
                and _qualified_name(node.func.value) in dataclasses_module_names
            )
            if is_replace_call and any(
                isinstance(descendant, (ast.Name, ast.Attribute))
                and (
                    _qualified_name(descendant).rsplit(".", maxsplit=1)[-1] in {"MessageDef", "MessageRef"}
                    or (isinstance(descendant, ast.Name) and descendant.id in local_message_definitions)
                )
                for descendant in ast.walk(node)
            ):
                violations.append(
                    f"{path}:{node.lineno}: dataclasses.replace must not construct MessageDef or MessageRef variants"
                )

        for node in ast.walk(tree):
            if not i18n_msg_import or not isinstance(node, ast.Name) or node.id != "msg":
                continue
            parent = parents.get(id(node))
            if isinstance(parent, ast.Call) and parent.func is node:
                continue
            violations.append(f"{path}:{node.lineno}: msg is a construction function, not a first-class value")
    assert violations == [], "\n".join(violations)


def _is_i18n_owner(owner: str, module_refs: set[str]) -> bool:
    # An owner reaches the i18n namespace when any dotted prefix of it is a
    # recorded module ref (i18n.messages through `from chrys.foundation
    # import i18n`) or any segment is spelled i18n — nested submodule access
    # launders the same way single-step access does.
    parts = owner.split(".")
    if "i18n" in parts:
        return True
    prefix = parts[0]
    if prefix in module_refs:
        return True
    for part in parts[1:]:
        prefix = f"{prefix}.{part}"
        if prefix in module_refs:
            return True
    return False


def _is_module_level_message_assignment(
    call: ast.Call,
    parents: Mapping[int, ast.AST],
    tree: ast.Module,
) -> bool:
    parent = parents.get(id(call))
    if isinstance(parent, ast.Assign):
        return (
            parent.value is call
            and len(parent.targets) == 1
            and isinstance(parent.targets[0], ast.Name)
            and parent in tree.body
        )
    if isinstance(parent, ast.AnnAssign):
        return parent.value is call and isinstance(parent.target, ast.Name) and parent in tree.body
    return False


def test_i18n_message_guard_accepts_canonical_definition_and_type_imports() -> None:
    source = (
        "from chrys.foundation.i18n import MessageDef, MessageRef, msg\n"
        "\n"
        "MESSAGE: MessageDef = msg('dialog.close', fallback='Close')\n"
        "\n"
        "def render(reference: MessageRef) -> MessageRef:\n"
        "    return reference\n"
    )
    _assert_i18n_message_construction_is_canonical({Path("src/chrys/app/good.py"): source})


def test_i18n_message_guard_accepts_fully_qualified_member_annotations() -> None:
    # Deep imports record prefix MODULES as banned values, but terminal
    # member references through them are ordinary qualified type names.
    source = (
        "import chrys.foundation.i18n.messages\n"
        "\n"
        "def render(definition: chrys.foundation.i18n.messages.MessageDef) -> None:\n"
        "    pass\n"
        "\n"
        "def resolve(reference: chrys.foundation.i18n.MessageRef) -> None:\n"
        "    pass\n"
    )
    _assert_i18n_message_construction_is_canonical({Path("src/chrys/app/good.py"): source})


def test_i18n_message_guard_accepts_unrelated_msg_usage_without_import() -> None:
    # ``msg`` stays a legal parameter and attribute name in files that never
    # import the constructor; only calls and shadowing definitions are
    # reserved, or the sweep would flag half the transcript helpers.
    source = (
        "def opens_turn(msg):\n    return msg.role == 'user'\n\ndef label(event):\n    return getattr(event, 'msg')\n"
    )
    _assert_i18n_message_construction_is_canonical({Path("src/chrys/app/good.py"): source})


@pytest.mark.parametrize(
    ("source", "match"),
    [
        (
            "from chrys.foundation.i18n import msg\ndef build():\n    local = msg('dialog.close', fallback='Close')\n",
            "module-level assignment",
        ),
        (
            "from chrys.foundation.i18n import msg\nif enabled:\n    MESSAGE = msg('dialog.close', fallback='Close')\n",
            "module-level assignment",
        ),
        (
            "from chrys.foundation.i18n import msg\nREFERENCE = msg('dialog.close', fallback='Close').bind()\n",
            "module-level assignment",
        ),
        (
            (
                "from chrys.foundation.i18n import msg as translated\n"
                "MESSAGE = translated('dialog.close', fallback='Close')\n"
            ),
            "import msg only",
        ),
        (
            "import chrys.foundation.i18n as i18n\nMESSAGE = i18n.msg('dialog.close', fallback='Close')\n",
            "bare name",
        ),
        (
            "from ..foundation import i18n\nMESSAGE = i18n.msg('dialog.close', fallback='Close')\n",
            "bare name",
        ),
        (
            "from ..foundation.i18n import msg\nMESSAGE = msg('dialog.close', fallback='Close')\n",
            "import msg only",
        ),
        (
            "from chrys.foundation.i18n import msg\nfactory = msg\n",
            "not a first-class value",
        ),
        (
            "from chrys.foundation.i18n import msg\nconsume(msg)\n",
            "not a first-class value",
        ),
        (
            "from chrys.foundation.i18n import msg\ndef factory():\n    return msg\n",
            "not a first-class value",
        ),
        (
            "from chrys.foundation.i18n import msg\ndef factory():\n    return msg('dialog.close', fallback='Close')\n",
            "module-level assignment",
        ),
        (
            (
                "from chrys.foundation.i18n import MessageDef\n"
                "MESSAGE = MessageDef(key='dialog.close', fallback='Close')\n"
            ),
            "construct messages only",
        ),
        (
            "from chrys.foundation.i18n import MessageRef\nREFERENCE = MessageRef(definition=definition)\n",
            "construct messages only",
        ),
        (
            (
                "import dataclasses\n"
                "from chrys.foundation.i18n import msg\n"
                "MESSAGE = msg('dialog.close', fallback='Close')\n"
                "VARIANT = dataclasses.replace(MESSAGE, fallback='Variant')\n"
            ),
            "dataclasses.replace",
        ),
        (
            (
                "def msg(key, fallback=None):\n"
                "    return (key, fallback)\n"
                "\n"
                "MESSAGE = msg('rogue.key', fallback='Rogue')\n"
            ),
            "locally defined",
        ),
        (
            "MESSAGE = msg('rogue.key', fallback='Rogue')\n",
            "canonical import",
        ),
        (
            "from chrys.foundation import i18n\nMESSAGE = getattr(i18n, 'msg')('dialog.close', fallback='Close')\n",
            "never via getattr",
        ),
        (
            "from chrys.foundation.i18n import messages\nMESSAGE = messages.msg('dialog.close', fallback='Close')\n",
            "bare name",
        ),
        (
            "from ..foundation.i18n import messages\nMESSAGE = messages.msg('dialog.close', fallback='Close')\n",
            "bare name",
        ),
        (
            (
                "import chrys.foundation.i18n.messages\n"
                "MESSAGE = chrys.foundation.i18n.messages.msg('dialog.close', fallback='Close')\n"
            ),
            "bare name",
        ),
        (
            (
                "from chrys.foundation import i18n\n"
                "factory = i18n.msg\n"
                "MESSAGE = factory('dialog.close', fallback='Close')\n"
            ),
            "bare name",
        ),
        (
            "if enabled:\n    from chrys.foundation.i18n import msg\nMESSAGE = msg('dialog.close', fallback='Close')\n",
            "import msg only",
        ),
        (
            (
                "from chrys.foundation.i18n import msg\n"
                "from rogue_module import msg\n"
                "MESSAGE = msg('dialog.close', fallback='Close')\n"
            ),
            "rebound",
        ),
        (
            (
                "from chrys.foundation import i18n\n"
                "translated = i18n\n"
                "MESSAGE = translated.msg('dialog.close', fallback='Close')\n"
            ),
            "aliased or passed",
        ),
        (
            (
                "from chrys.foundation.i18n import msg\n"
                "class msg:\n"
                "    pass\n"
                "MESSAGE = msg('dialog.close', fallback='Close')\n"
            ),
            "locally defined",
        ),
        (
            (
                "import chrys.foundation.i18n\n"
                "translated = chrys.foundation.i18n\n"
                "MESSAGE = translated.msg('dialog.close', fallback='Close')\n"
            ),
            "aliased or passed",
        ),
        (
            (
                "from chrys.foundation.i18n import msg\n"
                "MESSAGE = msg('dialog.close', fallback='Close')\n"
                "def handle(value):\n"
                "    match value:\n"
                "        case [*msg]:\n"
                "            pass\n"
            ),
            "locally defined",
        ),
        (
            (
                "from chrys.foundation.i18n import msg\n"
                "MESSAGE = msg('dialog.close', fallback='Close')\n"
                "try:\n"
                "    pass\n"
                "except RuntimeError as msg:\n"
                "    pass\n"
            ),
            "locally defined",
        ),
        (
            (
                "from chrys.foundation.i18n import msg\n"
                "from chrys.foundation.i18n import MessageDef as msg\n"
                "MESSAGE = msg('dialog.close', fallback='Close')\n"
            ),
            "rebound",
        ),
        (
            (
                "from chrys.foundation.i18n import msg\n"
                "from chrys.foundation import i18n as msg\n"
                "MESSAGE = msg('dialog.close', fallback='Close')\n"
            ),
            "rebound",
        ),
        (
            (
                "from chrys.foundation.i18n import msg\n"
                "import chrys.foundation.i18n as msg\n"
                "MESSAGE = msg('dialog.close', fallback='Close')\n"
            ),
            "rebound",
        ),
        (
            (
                "import chrys.foundation.i18n.messages\n"
                "translated = chrys.foundation.i18n\n"
                "MESSAGE = translated.msg('dialog.close', fallback='Close')\n"
            ),
            "aliased or passed",
        ),
        (
            "MESSAGE = msg('dialog.close', fallback='Close')\nfrom chrys.foundation.i18n import msg\n",
            "follow the canonical import",
        ),
        (
            "MESSAGE = msg('dialog.close', fallback='Close'); from chrys.foundation.i18n import msg\n",
            "follow the canonical import",
        ),
        (
            "from chrys.foundation import i18n\nMESSAGE = i18n.messages.msg('dialog.hidden', fallback='Hidden')\n",
            "bare name",
        ),
        (
            (
                "from chrys.foundation.i18n import msg\n"
                "import msg.submodule\n"
                "MESSAGE = msg('dialog.close', fallback='Close')\n"
            ),
            "rebound",
        ),
        (
            (
                "from dataclasses import replace as clone\n"
                "from chrys.foundation.i18n import msg\n"
                "MESSAGE = msg('dialog.close', fallback='Close')\n"
                "VARIANT = clone(MESSAGE, fallback='Variant')\n"
            ),
            "dataclasses.replace",
        ),
        (
            (
                "import dataclasses as dc\n"
                "from chrys.foundation.i18n import msg\n"
                "MESSAGE = msg('dialog.close', fallback='Close')\n"
                "VARIANT = dc.replace(MESSAGE, fallback='Variant')\n"
            ),
            "dataclasses.replace",
        ),
    ],
    ids=[
        "function-local",
        "conditional",
        "inline-bind",
        "aliased-import",
        "module-qualified",
        "relative-module-qualified",
        "relative-import",
        "rebound",
        "passed",
        "returned",
        "wrapper",
        "direct-definition",
        "direct-reference",
        "dataclass-replace",
        "local-def-shadow",
        "unimported-call",
        "getattr-indirection",
        "submodule-qualified",
        "relative-submodule-qualified",
        "plain-import-submodule",
        "aliased-attribute",
        "conditional-canonical-import",
        "rogue-import-rebind",
        "module-alias-first-class",
        "class-shadow",
        "module-alias-dotted",
        "match-star-capture",
        "except-handler-shadow",
        "member-import-rebind",
        "module-member-import-rebind",
        "plain-import-rebind",
        "deep-import-prefix-launder",
        "use-before-import",
        "same-line-use-before-import",
        "nested-submodule-qualified",
        "dotted-plain-import-root-rebind",
        "replace-function-alias",
        "replace-module-alias",
    ],
)
def test_i18n_message_guard_rejects_noncanonical_shapes(source: str, match: str) -> None:
    with pytest.raises(AssertionError, match=match):
        _assert_i18n_message_construction_is_canonical({Path("src/chrys/app/bad.py"): source})

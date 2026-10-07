# Copyright (c) 2021 Will McGugan
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Textual (MIT License; see NOTICE).

"""Patch: avoid repeated ``inspect.signature`` calls for Textual callbacks.

Textual wraps ``MessagePump.call_later`` / ``call_next`` callbacks in fresh
``functools.partial`` objects.  Textual 8.2.7 tries to cache callback arity on
the callable itself, but the partial branch calls ``_count_parameters`` on the
wrapped function directly, bypassing the existing ``count_parameters`` cache.

During chat-panel scrolling this repeatedly inspects signatures for the same
callback functions.  Cache the underlying callable recursively and subtract the
partial arguments from the cached value.
"""

from __future__ import annotations

from contextlib import suppress
from functools import partial
from inspect import isawaitable
from typing import Any

from chrys.foundation.patches.patcher import FilePatch, register

_RUNTIME_PATCH_MARKER = "_chrys_callback_cache_patched"

# The runtime wrapper caches arity under its OWN attribute rather than Textual's
# ``_param_count``.  When the wrapper is installed over the *unpatched* upstream
# function (e.g. before the file patch lands, or in tests against vanilla
# Textual), upstream may already have cached values with different semantics —
# vanilla stores the post-subtraction *bound* count on the *unbound* function,
# which the wrapper's bound-method branch would subtract from a second time.  A
# private attribute keeps the two caches from ever reading each other.
_PARAM_COUNT_ATTR = "_chrys_param_count"


def apply_runtime_patch() -> None:
    """Install the callback arity cache fix in the current process."""
    try:
        import textual._callback as callback_mod
        import textual.reactive as reactive_mod
    except ImportError:
        return

    existing_count_parameters = callback_mod.count_parameters
    if not getattr(existing_count_parameters, _RUNTIME_PATCH_MARKER, False):
        original_count_parameters = existing_count_parameters
        raw_count_parameters = callback_mod._count_parameters

        def count_parameters(func: Any) -> int:
            """Count callback parameters while reusing cached underlying arity.

            Caches under :data:`_PARAM_COUNT_ATTR` rather than Textual's
            ``_param_count`` so this wrapper never trusts a value the unpatched
            upstream cached with different semantics (see the constant's note).
            """
            cache_target = func
            if isinstance(func, partial):
                cached = getattr(func, _PARAM_COUNT_ATTR, None)
                if cached is not None:
                    return cached
                keyword_count = len(func.keywords or {})
                param_count = count_parameters(func.func) - (len(func.args) + keyword_count)
            elif hasattr(func, "__self__") and hasattr(func, "__func__"):
                cache_target = func.__func__
                return count_parameters(cache_target) - 1
            else:
                cached = getattr(func, _PARAM_COUNT_ATTR, None)
                if cached is not None:
                    return cached
                param_count = raw_count_parameters(func)

            with suppress(AttributeError, TypeError):
                setattr(cache_target, _PARAM_COUNT_ATTR, param_count)
            return param_count

        setattr(count_parameters, _RUNTIME_PATCH_MARKER, True)
        count_parameters._original = original_count_parameters
        callback_mod.count_parameters = count_parameters

    # Reactive imported this function by value before bootstrap. Repair that
    # consumer even on reapplication, when the defining module is already fixed.
    reactive_mod.count_parameters = callback_mod.count_parameters

    existing_invoke = callback_mod._invoke
    if not getattr(existing_invoke, _RUNTIME_PATCH_MARKER, False):
        original_invoke = existing_invoke

        async def _invoke(callback: Any, *params: Any) -> Any:
            """Invoke callbacks without arity introspection when no params are supplied."""
            _rich_traceback_guard = True
            if params:
                parameter_count = callback_mod.count_parameters(callback)
                result = callback(*params[:parameter_count])
            else:
                result = callback()
            if isawaitable(result):
                result = await result
            return result

        setattr(_invoke, _RUNTIME_PATCH_MARKER, True)
        _invoke._original = original_invoke
        callback_mod._invoke = _invoke


_OLD = '''def count_parameters(func: Callable) -> int:
    """Count the number of parameters in a callable"""
    try:
        return func._param_count
    except AttributeError:
        pass
    if isinstance(func, partial):
        param_count = _count_parameters(func.func) - (
            len(func.args) + len(func.keywords)
        )
    elif hasattr(func, "__self__"):
        # Bound method
        func = func.__func__  # type: ignore
        param_count = _count_parameters(func) - 1
    else:
        param_count = _count_parameters(func)
    try:
        func._param_count = param_count
    except TypeError:
        pass
    return param_count
'''

_NEW = '''def count_parameters(func: Callable) -> int:
    """Count the number of parameters in a callable"""
    cache_target = func
    if isinstance(func, partial):
        try:
            return func._param_count
        except AttributeError:
            pass
        param_count = count_parameters(func.func) - (
            len(func.args) + len(func.keywords or {})
        )
    elif hasattr(func, "__self__") and hasattr(func, "__func__"):
        # Bound method
        cache_target = func.__func__  # type: ignore
        return count_parameters(cache_target) - 1
    else:
        try:
            return func._param_count
        except AttributeError:
            pass
        param_count = _count_parameters(func)
    try:
        cache_target._param_count = param_count
    except (AttributeError, TypeError):
        pass
    return param_count
'''

_BUGGY_NEW = '''def count_parameters(func: Callable) -> int:
    """Count the number of parameters in a callable"""
    try:
        return func._param_count
    except AttributeError:
        pass
    cache_target = func
    if isinstance(func, partial):
        param_count = count_parameters(func.func) - (
            len(func.args) + len(func.keywords or {})
        )
    elif hasattr(func, "__self__"):
        # Bound method
        cache_target = func.__func__  # type: ignore
        param_count = count_parameters(cache_target) - 1
    else:
        param_count = _count_parameters(func)
    try:
        cache_target._param_count = param_count
    except (AttributeError, TypeError):
        pass
    return param_count
'''

register(
    FilePatch(
        package="textual",
        module_file="_callback.py",
        old_fragment=_OLD,
        new_fragment=_NEW,
        description="Cache Textual callback arity through partial and bound-method wrappers.",
        equivalent_fragments=(_BUGGY_NEW,),
    )
)

register(
    FilePatch(
        package="textual",
        module_file="_callback.py",
        old_fragment=_BUGGY_NEW,
        new_fragment=_NEW,
        description="Upgrade Textual callback arity cache patch for bound methods.",
    )
)

_OLD_INVOKE = '''async def _invoke(callback: Callable, *params: object) -> Any:
    """Invoke a callback with an arbitrary number of parameters.

    Args:
        callback: The callable to be invoked.

    Returns:
        The return value of the invoked callable.
    """
    _rich_traceback_guard = True
    parameter_count = count_parameters(callback)
    result = callback(*params[:parameter_count])
    if isawaitable(result):
        result = await result
    return result
'''

_NEW_INVOKE = '''async def _invoke(callback: Callable, *params: object) -> Any:
    """Invoke a callback with an arbitrary number of parameters.

    Args:
        callback: The callable to be invoked.

    Returns:
        The return value of the invoked callable.
    """
    _rich_traceback_guard = True
    if params:
        parameter_count = count_parameters(callback)
        result = callback(*params[:parameter_count])
    else:
        result = callback()
    if isawaitable(result):
        result = await result
    return result
'''

register(
    FilePatch(
        package="textual",
        module_file="_callback.py",
        old_fragment=_OLD_INVOKE,
        new_fragment=_NEW_INVOKE,
        description="Skip Textual callback arity inspection when no params are supplied.",
    )
)

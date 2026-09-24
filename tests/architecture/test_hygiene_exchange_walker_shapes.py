# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shape-by-shape red/green proofs for the hand-rolled exchange-walker guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.architecture.test_hygiene_exchange_walkers import _assert_no_hand_rolled_exchange_walkers
from tests.support.ci import CI_LINUX_ONLY

# Platform-independent source analysis: the Linux CI job covers it.
pytestmark = CI_LINUX_ONLY


def test_exchange_walker_guard_rejects_marker_forgetting_walker() -> None:
    # The most dangerous new walker FORGETS markers entirely, so no
    # marker-reference trigger can catch it — the walking constructs
    # themselves (role loop + type classification + pending state) must.
    source = (
        "def collect_unanswered(messages):\n"
        "    pending = {}\n"
        "    for message in messages:\n"
        '        if message.get("role") == "assistant":\n'
        '            for content in message.get("contents", []):\n'
        '                if content.get("type") == "function_call":\n'
        '                    pending[content.get("call_id")] = content\n'
        '        elif message.get("role") == "tool":\n'
        '            for content in message.get("contents", []):\n'
        '                if content.get("type") == "function_result":\n'
        '                    pending.pop(content.get("call_id"), None)\n'
        "    return pending\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_helper_indirected_walker() -> None:
    # A cursor walk that classifies through a local helper never reads the
    # type sets in its own loop; the helper must not launder it.
    source = (
        "def _is_result_block(message):\n"
        '    return message.role == "tool" or any(\n'
        '        content.type == "function_result" for content in message.contents\n'
        "    )\n"
        "\n"
        "def find_block_end(messages, start):\n"
        "    index = start\n"
        "    while index < len(messages) and messages[index].role != 'user':\n"
        "        if not _is_result_block(messages[index]):\n"
        "            break\n"
        "        index += 1\n"
        "    return index\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_aliased_cursor_walker() -> None:
    # ``message = messages[index]`` inside a cursor walk is the ordinary
    # walker spelling; the alias must not hide the role read.
    source = (
        "def find_output_end(messages, start):\n"
        "    index = start\n"
        "    while index < len(messages):\n"
        "        message = messages[index]\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        index += 1\n"
        "    return index\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_loop_target_alias() -> None:
    # Rebinding the loop target to a fresh name walks the same items.
    source = (
        "def collect_calls(messages):\n"
        "    pending = {}\n"
        "    for item in messages:\n"
        "        message = item\n"
        '        if message.role == "assistant":\n'
        "            for content in message.contents:\n"
        '                if content.type == "function_call":\n'
        "                    pending[content.call_id] = content\n"
        "    return pending\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_serialized_subscript_walker() -> None:
    # A dict-transcript walker reads role and call_id through subscripts.
    source = (
        "def collect_unanswered(messages):\n"
        "    pending = {}\n"
        "    for message in messages:\n"
        '        if message["role"] == "assistant":\n'
        '            for content in message["contents"]:\n'
        '                if content["type"] == "function_call":\n'
        '                    pending[content["call_id"]] = content\n'
        '        elif message["role"] == "tool":\n'
        '            for content in message["contents"]:\n'
        '                if content["type"] == "function_result":\n'
        '                    pending.pop(content["call_id"], None)\n'
        "    return pending\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_dead_grammar_name_reference() -> None:
    # Merely naming iter_exchanges is not consuming it; only a call is.
    source = (
        "def collect_unanswered(messages):\n"
        "    unused = iter_exchanges\n"
        "    pending = {}\n"
        "    for message in messages:\n"
        '        if message.role == "assistant":\n'
        "            for content in message.contents:\n"
        '                if content.type == "function_call":\n'
        "                    pending[content.call_id] = content\n"
        '        elif message.role == "tool":\n'
        "            for content in message.contents:\n"
        '                if content.type == "function_result":\n'
        "                    pending.pop(content.call_id, None)\n"
        "    return pending\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_role_helper_walker() -> None:
    # Factoring the role read into a local helper is ordinary refactoring,
    # not sanitization; calling it on a walked item is still a role read.
    source = (
        "def _role(message):\n"
        "    return message.role\n"
        "\n"
        "def collect_calls(messages):\n"
        "    pending = {}\n"
        "    for message in messages:\n"
        '        if _role(message) == "assistant":\n'
        "            for content in message.contents:\n"
        '                if content.type == "function_call":\n'
        "                    pending[content.call_id] = content\n"
        "    return pending\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_id_helper_walker() -> None:
    # The pairing-state read hides inside a local key helper; the walk in
    # the caller is still keyed on call identity.
    source = (
        "def _call_key(content):\n"
        "    return content.call_id\n"
        "\n"
        "def collect_unanswered(messages):\n"
        "    pending = {}\n"
        "    for message in messages:\n"
        '        if message.role != "assistant":\n'
        "            continue\n"
        "        for content in message.contents:\n"
        '            if content.type == "function_call":\n'
        "                pending[_call_key(content)] = content\n"
        "    return pending\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_range_cursor_walker() -> None:
    # ``for index in range(...)`` over a subscripted sequence is the same
    # cursor walk as a while loop and must count as boundary state.
    source = (
        "def find_block_end(messages, start):\n"
        "    end = start\n"
        "    for index in range(start, len(messages)):\n"
        "        message = messages[index]\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        end = index\n"
        "    return end\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_enumerate_cursor_walker() -> None:
    # ``for index, message in enumerate(...)`` doing index arithmetic is the
    # same cursor walk as a subscripted range loop — the exclusive-end
    # bookkeeping is boundary state even without a subscript.
    source = (
        "def find_block_end(messages):\n"
        "    end = 0\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        end = index + 1\n"
        "    return end\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_enumerate_index_alias_arithmetic() -> None:
    # Copying the enumerate index into a fresh name before the arithmetic
    # is the same cursor walk.
    source = (
        "def find_block_end(messages):\n"
        "    end = 0\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        end = index\n"
        "    return end + 1\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_one_based_enumerate_cursor_walker() -> None:
    # ``enumerate(messages, 1)`` folds the ``+1`` into the start argument;
    # returning the tracked index is the same exclusive-end bookkeeping.
    source = (
        "def find_block_end(messages):\n"
        "    end = 0\n"
        "    for index, message in enumerate(messages, 1):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        end = index\n"
        "    return end\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_returned_enumerate_index_walker() -> None:
    # Returning the zero-based tracked index is the inclusive-end twin of
    # the one-based walk — the arithmetic just moves to the caller.
    source = (
        "def find_last_result_index(messages):\n"
        "    end = 0\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        end = index\n"
        "    return end\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_max_tracked_enumerate_index() -> None:
    # ``end = max(end, index)`` derives boundary state from the index
    # without any BinOp or direct copy — still a cursor walk.
    source = (
        "def find_last_result_index(messages):\n"
        "    end = 0\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        end = max(end, index)\n"
        "    return end\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_conditional_enumerate_index_capture() -> None:
    # A conditional-expression update is the same scalar boundary capture.
    source = (
        "def find_last_result_index(messages):\n"
        "    end = 0\n"
        "    for index, message in enumerate(messages):\n"
        '        end = index if message.role == "tool" else end\n'
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "    return end\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_attribute_stored_enumerate_index() -> None:
    # Storing the index on a result object is structured boundary capture.
    source = (
        "def find_block_bounds(messages, bounds):\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        bounds.end = index\n"
        "    return bounds\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_accepts_one_based_enumerate_reporter() -> None:
    # An explicit enumerate start adds no pairing state by itself: a
    # human-facing one-based reporter must stay out of reach.
    source = (
        "def call_message_positions(messages):\n"
        "    positions = []\n"
        "    for position, message in enumerate(messages, 1):\n"
        '        if message.role != "assistant":\n'
        "            continue\n"
        '        if any(content.type == "function_call" for content in message.contents):\n'
        "            positions.append(position)\n"
        "    return positions\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


def test_exchange_walker_guard_accepts_enumerate_membership_test() -> None:
    # A membership test consumes the index as a question, not a value —
    # the sidecar-suppression shape in the TUI merge walk. The boolean
    # answer carries no cursor state.
    source = (
        "def suppress_sidecars(messages, sidecars):\n"
        "    kept = []\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "assistant":\n'
        "            continue\n"
        '        if any(content.type == "function_call" for content in message.contents):\n'
        "            text = None if index in sidecars else message.text\n"
        "            kept.append(text)\n"
        "    return kept\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


def test_exchange_walker_guard_rejects_index_boundary_comparison_walker() -> None:
    # ``index == len(messages) - 1`` never stores a cursor — the boolean
    # itself controls the exchange boundary. Equality/order comparisons on
    # the index are boundary logic; only membership tests stay exempt.
    source = (
        "def last_message_if_result_tail(messages):\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        if index == len(messages) - 1:\n"
        "            return message\n"
        "    return None\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_tuple_unpacked_enumerate_index() -> None:
    # Parallel bounds updates are ordinary Python; the tuple target must
    # not hide the scalar capture.
    source = (
        "def find_block_end(messages):\n"
        "    previous_end = end = 0\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        previous_end, end = end, index\n"
        "    return end\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_fixed_key_bounds_store() -> None:
    # A fixed-key bounds dictionary is serialized cursor state, not a
    # dynamic coordinate/reporter map.
    source = (
        "def find_block_bounds(messages, bounds):\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        '        bounds["end"] = index\n'
        "    return bounds\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_accepts_formatted_label_reporter() -> None:
    # Formatting the index into display text consumes it without capturing
    # cursor state — the refactored twin of the direct append reporter.
    source = (
        "def call_message_labels(messages):\n"
        "    labels = []\n"
        "    for position, message in enumerate(messages, 1):\n"
        '        if message.role != "assistant":\n'
        "            continue\n"
        '        if any(content.type == "function_call" for content in message.contents):\n'
        '            label = f"message {position}"\n'
        "            labels.append(label)\n"
        "    return labels\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


def test_exchange_walker_guard_rejects_index_truthiness_boundary_walker() -> None:
    # ``if not index`` is the first-boundary check spelled as truthiness —
    # the test gates the walk without storing or comparing explicitly.
    source = (
        "def first_message_if_result_head(messages):\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        if not index:\n"
        "            return message\n"
        "    return None\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_match_case_boundary_walker() -> None:
    # ``match index: case 0`` is the same first-boundary gate in pattern
    # form.
    source = (
        "def first_message_if_result_head(messages):\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        match index:\n"
        "            case 0:\n"
        "                return message\n"
        "    return None\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


@pytest.mark.parametrize("key", ["-1", '("end", 0)'])
def test_exchange_walker_guard_rejects_literal_key_bounds_store(key: str) -> None:
    # Fixed keys come in more AST shapes than a bare constant: a negative
    # index and a tuple of literals serialize cursor state all the same.
    source = (
        "def find_block_bounds(messages, bounds):\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        f"        bounds[{key}] = index\n"
        "    return bounds\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_keyword_argument_bounds_store() -> None:
    # ``bounds.update(end=index)`` is fixed-field boundary storage through
    # a mutation call, not a dynamic reporter map.
    source = (
        "def find_block_bounds(messages, bounds):\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        bounds.update(end=index)\n"
        "    return bounds\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_accepts_inline_formatted_comparison() -> None:
    # An f-string compared inline is the one-statement twin of the
    # accepted formatted-label reporter — rendering stays exempt inside a
    # comparison too.
    source = (
        "def find_selected_label(messages, selected_label):\n"
        "    labels = []\n"
        "    for position, message in enumerate(messages, 1):\n"
        '        if message.role != "assistant":\n'
        "            continue\n"
        '        if any(content.type == "function_call" for content in message.contents):\n'
        '            if f"message {position}" == selected_label:\n'
        "                labels.append(selected_label)\n"
        "    return labels\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


@pytest.mark.parametrize(
    "store",
    [
        'bounds.update({"end": index})',
        'bounds.setdefault("end", index)',
        'setattr(bounds, "end", index)',
        "bounds.update(dict(end=index))",
        'bounds.update([("end", index)])',
        'bounds.update({"start": 0}, end=index)',
        'bounds.update({"start": 0} | defaults, end=index)',
    ],
)
def test_exchange_walker_guard_rejects_positional_store_calls(store: str) -> None:
    # update/setdefault/setattr are the call spellings of a fixed-key
    # store — the same cursor state ``bounds["end"] = index`` serializes.
    source = (
        "def find_block_bounds(messages, bounds):\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        f"        {store}\n"
        "    return bounds\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


@pytest.mark.parametrize(
    "report",
    ["reporter.emit(position=position)", 'logger.info("call site", extra={"position": position})'],
)
def test_exchange_walker_guard_accepts_keyword_reporting_calls(report: str) -> None:
    # Handing the position to another component by keyword is reporting,
    # not storage — only fixed-field store callees write cursor state.
    source = (
        "def report_call_positions(messages, reporter, logger):\n"
        "    for position, message in enumerate(messages, 1):\n"
        '        if message.role != "assistant":\n'
        "            continue\n"
        '        if any(content.type == "function_call" for content in message.contents):\n'
        f"            {report}\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


def test_exchange_walker_guard_accepts_dynamic_key_store_call() -> None:
    # A dynamic key keeps the coordinate-map exemption through the call
    # spelling of a store too.
    source = (
        "def positions_by_message(messages, positions):\n"
        "    for position, message in enumerate(messages, 1):\n"
        '        if message.role != "assistant":\n'
        "            continue\n"
        '        if any(content.type == "function_call" for content in message.contents):\n'
        "            positions.setdefault(message.id, position)\n"
        "    return positions\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


def test_exchange_walker_guard_rejects_match_guard_boundary_walker() -> None:
    # A case guard gates the boundary exactly like an if test — the same
    # cursor question moved from the subject into the guard.
    source = (
        "def first_message_if_result_head(messages):\n"
        "    for index, message in enumerate(messages):\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        match message.role:\n"
        '            case "tool" if not index:\n'
        "                return message\n"
        "    return None\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


@pytest.mark.parametrize("gate", ["not index", "index == len(messages) - 1"])
def test_exchange_walker_guard_rejects_comprehension_boundary_gate(gate: str) -> None:
    # Comprehension filters gate the walk exactly like statement-loop
    # tests; an enumerate cursor does not launder through comprehension
    # form.
    source = (
        "def result_heads(messages):\n"
        "    return [\n"
        "        message\n"
        "        for index, message in enumerate(messages)\n"
        '        if message.role == "tool"\n'
        '        if any(content.type == "function_result" for content in message.contents)\n'
        f"        if {gate}\n"
        "    ]\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_accepts_comprehension_position_reporter() -> None:
    # A comprehension whose value slot collects positions is the
    # comprehension spelling of the append reporter.
    source = (
        "def call_positions(messages):\n"
        "    return [\n"
        "        position\n"
        "        for position, message in enumerate(messages, 1)\n"
        '        if message.role == "assistant"\n'
        '        if any(content.type == "function_call" for content in message.contents)\n'
        "    ]\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


def test_exchange_walker_guard_rejects_collapsed_comprehension_cursor() -> None:
    # max() over a comprehension of indices collapses the container into
    # scalar cursor state — the reporter exemption stops at collapse.
    source = (
        "def last_result_index(messages):\n"
        "    end = max(\n"
        "        index\n"
        "        for index, message in enumerate(messages)\n"
        '        if message.role == "tool"\n'
        '        if any(content.type == "function_result" for content in message.contents)\n'
        "    )\n"
        "    return end\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_accepts_positional_update_reporter() -> None:
    # ``progress.update(task_id, completed=position)`` carries a positional
    # handle argument — the reporter-API shape, not the keyword-only or
    # mapping-shaped dict.update store idioms.
    source = (
        "def report_call_positions(messages, progress, task_id):\n"
        "    for position, message in enumerate(messages, 1):\n"
        '        if message.role != "assistant":\n'
        "            continue\n"
        '        if any(content.type == "function_call" for content in message.contents):\n'
        "            progress.update(task_id, completed=position)\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


def test_exchange_walker_guard_rejects_subscript_collapsed_comprehension() -> None:
    # Subscripting a comprehension of indices extracts scalar cursor state
    # in one expression — the reporter exemption stops at collapse.
    source = (
        "def last_result_index(messages):\n"
        "    return [\n"
        "        index\n"
        "        for index, message in enumerate(messages)\n"
        '        if message.role == "tool"\n'
        '        if any(content.type == "function_result" for content in message.contents)\n'
        "    ][-1]\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_accepts_sliced_comprehension_report() -> None:
    # A slice keeps the container a container — a truncated positions
    # report, not a scalar extraction.
    source = (
        "def first_result_positions(messages):\n"
        "    return [\n"
        "        index\n"
        "        for index, message in enumerate(messages)\n"
        '        if message.role == "tool"\n'
        '        if any(content.type == "function_result" for content in message.contents)\n'
        "    ][:3]\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


def test_exchange_walker_guard_accepts_outer_name_shadowed_by_comprehension_index() -> None:
    # Python scopes a comprehension target to the comprehension; an outer
    # parameter that happens to share its name is not an enumerate cursor.
    source = (
        "def report_call_positions(messages, index):\n"
        "    positions = [\n"
        "        index\n"
        "        for index, message in enumerate(messages)\n"
        '        if message.role == "assistant"\n'
        '        if any(content.type == "function_call" for content in message.contents)\n'
        "    ]\n"
        "    if index:\n"
        "        return positions\n"
        "    return []\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


def test_exchange_walker_guard_rejects_lambda_comprehension_walker() -> None:
    # A lambda is a scope of its own and gets its own evaluation — a
    # factory-returned selector cannot launder a comprehension walk.
    source = (
        "def result_head_selector():\n"
        "    return lambda messages: [\n"
        "        message\n"
        "        for index, message in enumerate(messages)\n"
        '        if message.role == "tool"\n'
        '        if any(content.type == "function_result" for content in message.contents)\n'
        "        if not index\n"
        "    ][0]\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_shadowed_grammar_helper_name() -> None:
    # A walker's own nested helper SHADOWS a module-level grammar helper
    # of the same name — the call resolves to the nearest binding, which
    # consumes no grammar.
    source = (
        "def block_pairs(messages):\n"
        "    return list(pair_results(messages))\n"
        "\n"
        "\n"
        "def find_last_result(messages):\n"
        "    def block_pairs():\n"
        "        return []\n"
        "    pairs = block_pairs()\n"
        "    end = 0\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        end = index\n"
        "    return pairs, end\n"
    )
    with pytest.raises(AssertionError, match="find_last_result hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_cross_class_self_helper() -> None:
    # ``self.block_pairs`` resolves within the calling method's OWN class;
    # a grammar-backed method of another class does not exempt it.
    source = (
        "class GrammarConsumer:\n"
        "    def block_pairs(self, messages):\n"
        "        return list(pair_results(messages))\n"
        "\n"
        "\n"
        "class Walker:\n"
        "    def block_pairs(self, messages):\n"
        "        return []\n"
        "\n"
        "    def find_last_result(self, messages):\n"
        "        pairs = self.block_pairs(messages)\n"
        "        end = 0\n"
        "        for index, message in enumerate(messages):\n"
        '            if message.role != "tool":\n'
        "                break\n"
        '            if not any(content.type == "function_result" for content in message.contents):\n'
        "                break\n"
        "            end = index\n"
        "        return pairs, end\n"
    )
    with pytest.raises(AssertionError, match=r"Walker\.find_last_result hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_scopes_allowlist_entries_to_their_class() -> None:
    # An allowlist entry vouches for ONE scope; a same-named method on a
    # different class in the same file is not covered.
    source = (
        "class NonVisionImageStubMiddleware:\n"
        "    def process(self, messages):\n"
        "        return messages\n"
        "\n"
        "\n"
        "class NewWalker:\n"
        "    def process(self, messages):\n"
        "        end = 0\n"
        "        for index, message in enumerate(messages):\n"
        '            if message.role != "tool":\n'
        "                break\n"
        '            if not any(content.type == "function_result" for content in message.contents):\n'
        "                break\n"
        "            end = index\n"
        "        return end\n"
    )
    with pytest.raises(AssertionError, match=r"NewWalker\.process hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/vision.py"): source})


def test_exchange_walker_guard_rejects_lambda_classifier_helper() -> None:
    # A name-bound lambda is a helper like any def — classification
    # through it propagates to the calling walker.
    source = (
        'is_result = lambda content: content.type == "function_result"\n'
        "\n"
        "\n"
        "def find_last_result(messages):\n"
        "    end = 0\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        "        if not any(is_result(content) for content in message.contents):\n"
        "            break\n"
        "        end = index\n"
        "    return end\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_lambda_role_helper() -> None:
    # Role reads through a name-bound lambda count the same as inline.
    source = (
        "read_role = lambda message: message.role\n"
        "\n"
        "\n"
        "def find_last_result(messages):\n"
        "    end = 0\n"
        "    for index, message in enumerate(messages):\n"
        '        if read_role(message) != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        end = index\n"
        "    return end\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_singleton_unpacked_comprehension() -> None:
    # ``end, = [...]`` destructures the container away in one statement —
    # scalar extraction, not a positions report.
    source = (
        "def sole_result_index(messages):\n"
        "    end, = [\n"
        "        index\n"
        "        for index, message in enumerate(messages)\n"
        '        if message.role == "tool"\n'
        '        if any(content.type == "function_result" for content in message.contents)\n'
        "    ]\n"
        "    return end\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_parameter_shadowed_grammar_helper() -> None:
    # A parameter shadows a same-named module grammar helper — the walker
    # calls its CALLBACK, not the helper, and earns no exemption.
    source = (
        "def block_pairs(messages):\n"
        "    return list(pair_results(messages))\n"
        "\n"
        "\n"
        "def find_last_result(messages, block_pairs):\n"
        "    pairs = block_pairs(messages)\n"
        "    end = 0\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        end = index\n"
        "    return pairs, end\n"
    )
    with pytest.raises(AssertionError, match="find_last_result hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_cross_class_attribute_grammar_exemption() -> None:
    # ``consumer.block_pairs`` cannot borrow another class's grammar-backed
    # method — with mixed same-named candidates the coarse attribute
    # fallback refuses to exempt.
    source = (
        "class GrammarConsumer:\n"
        "    def block_pairs(self, messages):\n"
        "        return list(pair_results(messages))\n"
        "\n"
        "\n"
        "class LocalConsumer:\n"
        "    def block_pairs(self, messages):\n"
        "        return []\n"
        "\n"
        "\n"
        "def find_last_result(messages, consumer: LocalConsumer):\n"
        "    pairs = consumer.block_pairs(messages)\n"
        "    end = 0\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        end = index\n"
        "    return pairs, end\n"
    )
    with pytest.raises(AssertionError, match="find_last_result hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_accepts_sole_attribute_grammar_helper() -> None:
    # When every module binding of the called name is grammar-backed, an
    # attribute call through a collaborator keeps the exemption.
    source = (
        "class GrammarConsumer:\n"
        "    def block_pairs(self, messages):\n"
        "        return list(pair_results(messages))\n"
        "\n"
        "\n"
        "def find_last_result(messages, consumer):\n"
        "    pairs = consumer.block_pairs(messages)\n"
        "    end = 0\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        end = index\n"
        "    return pairs, end\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


def test_exchange_walker_guard_rejects_inherited_role_helper() -> None:
    # ``self.read_role`` inherited from a same-module base class reaches
    # the walker — resolution climbs the local class lineage.
    source = (
        "class RoleReader:\n"
        "    def read_role(self, message):\n"
        "        return message.role\n"
        "\n"
        "\n"
        "class Walker(RoleReader):\n"
        "    def find_last_result(self, messages):\n"
        "        end = 0\n"
        "        for index, message in enumerate(messages):\n"
        '            if self.read_role(message) != "tool":\n'
        "                break\n"
        '            if not any(content.type == "function_result" for content in message.contents):\n'
        "                break\n"
        "            end = index\n"
        "        return end\n"
    )
    with pytest.raises(AssertionError, match=r"Walker\.find_last_result hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_generic_base_inherited_role_helper() -> None:
    # A subscripted (generic) local base is still the same class — lineage
    # climbing unwraps the subscript.
    source = (
        "class RoleReader:\n"
        "    def read_role(self, message):\n"
        "        return message.role\n"
        "\n"
        "\n"
        "class Walker(RoleReader[Message]):\n"
        "    def find_last_result(self, messages):\n"
        "        end = 0\n"
        "        for index, message in enumerate(messages):\n"
        '            if self.read_role(message) != "tool":\n'
        "                break\n"
        '            if not any(content.type == "function_result" for content in message.contents):\n'
        "                break\n"
        "            end = index\n"
        "        return end\n"
    )
    with pytest.raises(AssertionError, match=r"Walker\.find_last_result hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_reversed_enumerate_walker() -> None:
    # ``reversed(list(enumerate(...)))`` is the natural backwards scan —
    # wrapper calls don't hide the enumerate cursor.
    source = (
        "def find_last_result(messages):\n"
        "    for index, message in reversed(list(enumerate(messages))):\n"
        '        if message.role != "tool":\n'
        "            continue\n"
        '        if any(content.type == "function_result" for content in message.contents):\n'
        "            return index\n"
        "    return None\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_popped_comprehension_cursor() -> None:
    # ``[...].pop()`` collapses the container in the same statement —
    # the method-call spelling of the max/subscript collapse.
    source = (
        "def last_result_index(messages):\n"
        "    return [\n"
        "        index\n"
        "        for index, message in enumerate(messages)\n"
        '        if message.role == "tool"\n'
        '        if any(content.type == "function_result" for content in message.contents)\n'
        "    ].pop()\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_cross_scope_helper_name_collision() -> None:
    # A grammar-backed helper nested in one function must not exempt a
    # walker in another function calling its OWN same-named helper —
    # bare-name propagation respects lexical visibility.
    source = (
        "def grammar_consumer(messages):\n"
        "    def block_pairs():\n"
        "        return list(pair_results(messages))\n"
        "    return block_pairs()\n"
        "\n"
        "\n"
        "def find_last_result(messages):\n"
        "    def block_pairs():\n"
        "        return []\n"
        "    pairs = block_pairs()\n"
        "    end = 0\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "tool":\n'
        "            break\n"
        '        if not any(content.type == "function_result" for content in message.contents):\n'
        "            break\n"
        "        end = index\n"
        "    return pairs, end\n"
    )
    with pytest.raises(AssertionError, match="find_last_result hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_accepts_formatter_with_nested_indexed_helper() -> None:
    # A nested helper is its own scope: its enumerate cursor gets its own
    # guard evaluation and must not be attributed to the enclosing
    # formatter.
    source = (
        "def render_transcript(messages):\n"
        "    def numbered_lines(lines):\n"
        "        for index, line in enumerate(lines):\n"
        "            if index:\n"
        "                yield line\n"
        "    rendered = []\n"
        "    for message in messages:\n"
        '        if message.role != "assistant":\n'
        "            continue\n"
        '        if any(content.type == "function_call" for content in message.contents):\n'
        "            rendered.extend(numbered_lines(message.text.splitlines()))\n"
        "    return rendered\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


def test_exchange_walker_guard_rejects_keyword_role_helper_walker() -> None:
    # Passing the walked item by keyword is the same helper role read.
    source = (
        "def _role(message):\n"
        "    return message.role\n"
        "\n"
        "def collect_calls(messages):\n"
        "    pending = {}\n"
        "    for message in messages:\n"
        '        if _role(message=message) == "assistant":\n'
        "            for content in message.contents:\n"
        '                if content.type == "function_call":\n'
        "                    pending[content.call_id] = content\n"
        "    return pending\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_rejects_discarded_grammar_call() -> None:
    # A grammar call whose result is thrown away consumes nothing; the
    # hand-rolled walk beside it is still the transcript authority.
    source = (
        "def collect_unanswered(messages, accessor):\n"
        "    iter_exchanges(messages, accessor)\n"
        "    pending = {}\n"
        "    for message in messages:\n"
        '        if message.role == "assistant":\n'
        "            for content in message.contents:\n"
        '                if content.type == "function_call":\n'
        "                    pending[content.call_id] = content\n"
        '        elif message.role == "tool":\n'
        "            for content in message.contents:\n"
        '                if content.type == "function_result":\n'
        "                    pending.pop(content.call_id, None)\n"
        "    return pending\n"
    )
    with pytest.raises(AssertionError, match="hand-rolls an exchange walk"):
        _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/bad.py"): source})


def test_exchange_walker_guard_accepts_grammar_based_consumer() -> None:
    # A consumer on the shared grammar may still run a role-reading skeleton
    # walk (trim's backward walk) — the grammar reference is the tell.
    source = (
        "def trim(messages):\n"
        "    owners = {e.response_indices[0]: e for e in iter_exchanges(messages, ACCESSOR)}\n"
        "    i = len(messages) - 1\n"
        "    while i >= 0:\n"
        '        if messages[i].role != "assistant":\n'
        "            i -= 1\n"
        "            continue\n"
        '        if any(c.type == "function_call" and c.call_id for c in messages[i].contents):\n'
        "            break\n"
        "        i -= 1\n"
        "    return i\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


def test_exchange_walker_guard_accepts_per_message_serializer() -> None:
    # A wire serializer reads role off its parameter and loops one message's
    # contents; it walks no transcript and must stay out of reach.
    source = (
        "def prepare_message(message):\n"
        "    items = []\n"
        "    for content in message.contents:\n"
        '        if message.role == "assistant" and content.type == "function_call":\n'
        '            items.append({"id": content.call_id})\n'
        "    return items\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})


def test_exchange_walker_guard_accepts_enumerate_without_index_arithmetic() -> None:
    # Enumerate alone is not pairing state: a reporter that collects indices
    # without arithmetic, subscripting, or id reads stays out of reach.
    source = (
        "def call_message_indices(messages):\n"
        "    indices = []\n"
        "    for index, message in enumerate(messages):\n"
        '        if message.role != "assistant":\n'
        "            continue\n"
        '        if any(content.type == "function_call" for content in message.contents):\n'
        "            indices.append(index)\n"
        "    return indices\n"
    )
    _assert_no_hand_rolled_exchange_walkers({Path("src/chrys/service/good.py"): source})

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for token calibration, trigger math and the compaction admission protocol."""

import pytest

from chrys.kernel import (
    CompactionAdmissionState,
    CompactionCallContext,
    Message,
    annotate_message_groups,
    included_token_count,
)
from chrys.kernel import compaction as chrys_compaction
from chrys.kernel.client import _clamp_output_cap_for_context
from chrys.service.session.runtime_metadata import SessionRuntimeMetadata
from tests.service.context.compaction._compaction_helpers import (
    _assistant_text,
    _assistant_tool_call,
    _build_multi_turn,
    _build_single_turn,
    _build_tool_group,
    _estimate_tokens,
    _make_strategy,
    _tool_result,
    _user,
)


async def test_last_included_tokens_tracked():
    """Strategy tracks its own included_token_count for calibration."""
    messages = _build_single_turn(4, result_size=1000)

    strategy = _make_strategy(max_context_tokens=1_000_000)
    assert strategy.last_included_tokens == 0

    await strategy(messages)
    assert strategy.last_included_tokens > 0

    expected = _estimate_tokens(messages)
    assert strategy.last_included_tokens == expected


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param([(1.1, 1.1)], id="api_above_estimate"),
        pytest.param([(0.9, 0.9)], id="hot_estimator"),
        pytest.param([(10.0, 1.5), (1.2, 1.2)], id="capped_then_within_cap"),
    ],
)
async def test_calibrate_learns_ratio(steps: list[tuple[float, float]]) -> None:
    """calibrate() learns overhead on the first call, then ratio = api / (local + overhead)
    on subsequent calls: below 1.0 when our estimate runs hot, and capped at
    _MAX_CALIBRATION_RATIO (1.5) so an API spike cannot poison later estimates."""
    messages = _build_single_turn(4, result_size=4000)

    strategy = _make_strategy(max_context_tokens=1_000_000)
    assert strategy.calibration_ratio == 1.0

    await strategy(messages)
    local = strategy.last_included_tokens
    assert local > 0

    # First calibrate: learns overhead, skips ratio (circular)
    overhead = 5000
    strategy.calibrate(local + overhead)
    assert strategy.system_overhead_tokens == overhead
    assert strategy.calibration_ratio == 1.0  # unchanged on first call

    for factor, expected_ratio in steps:
        api_tokens = int((local + overhead) * factor)
        strategy.calibrate(api_tokens)
        if factor > 1.5:
            # A spike is clamped to the published 1.5 boundary. Pin it exactly rather
            # than within the tolerance below, which would accept a cap of 1.495.
            assert strategy.calibration_ratio == 1.5
        else:
            assert abs(strategy.calibration_ratio - expected_ratio) < 0.01


async def test_restored_old_estimator_session_is_migrated_before_admission():
    """UnifiedContextStrategy discards both artifacts from the old
    (``ensure_ascii=True``) estimator on restore — the
    prepare pass recomputes unversioned token counts and the v1 calibration
    record is rejected — so output-cap admission never mixes stale artifacts
    with new-estimator semantics before the first provider response.
    """
    # A restored CJK conversation carrying escape-era inflated counts with no
    # estimator-version stamp (exactly what pre-migration sessions persist).
    messages = [_user("中文内容" * 100), _assistant_text("日本語のかな" * 100)]
    annotate_message_groups(messages)
    for message in messages:
        annotation = message.additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY]
        annotation[chrys_compaction.GROUP_TOKEN_COUNT_KEY] = 45_000
        assert chrys_compaction.GROUP_TOKEN_ESTIMATOR_VERSION_KEY not in annotation

    strategy = _make_strategy(max_context_tokens=100_000)
    metadata = SessionRuntimeMetadata(
        context_calibration={
            "v": 1,
            "system_overhead_tokens": 5_000,
            "calibration_ratio": 0.4,
            "model_profile_fingerprint": "fp-model",
            "agent_profile_fingerprint": "fp-agent",
        }
    )
    # Fingerprints match, but the v1 record must be rejected wholesale.
    assert not metadata.restore_context_calibration(
        strategy,
        model_profile_fingerprint="fp-model",
        agent_profile_fingerprint="fp-agent",
    )
    assert strategy.calibration_ratio == 1.0
    assert strategy.system_overhead_tokens == 0

    # The real prepare pass, before any provider response arrives.
    await strategy(messages)

    recomputed = strategy.last_included_tokens
    assert recomputed == included_token_count(messages)
    assert recomputed < 5_000  # stale 90k total discarded, not reused
    for message in messages:
        annotation = message.additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY]
        assert annotation[chrys_compaction.GROUP_TOKEN_ESTIMATOR_VERSION_KEY] == (
            chrys_compaction.TOKEN_ESTIMATOR_VERSION
        )

    # Admission with migrated state: recomputed input at ratio 1.0 leaves
    # ample room, so the cap survives untouched. Stale counts (room = 10k) or
    # the stale 0.4 ratio over stale counts (room = 62k) would clamp it.
    admitted = _clamp_output_cap_for_context(
        {"max_tokens": 90_000},
        strategy=strategy,
        request_overhead_tokens=0,
        client_kwargs={},
    )
    assert admitted["max_tokens"] == 90_000


async def test_calibrate_first_call_learns_overhead_only():
    """First calibrate() learns overhead and skips ratio (circular).

    This prevents extreme calibration ratios from the first API call where
    overhead = api_input - local would make ratio = 1.0 trivially.
    """
    # Small conversation: 2 tool groups with tiny results
    messages = _build_single_turn(2, result_size=100)

    strategy = _make_strategy(max_context_tokens=1_000_000)
    assert strategy.calibration_ratio == 1.0

    await strategy(messages)
    our_estimate = strategy.last_included_tokens
    assert our_estimate > 0

    # First calibrate learns overhead, keeps ratio at 1.0
    strategy.calibrate(50_000)
    assert strategy.system_overhead_tokens == 50_000 - our_estimate
    assert strategy.calibration_ratio == 1.0, "First call should skip ratio calibration"


async def test_calibration_stable_across_approval_resubmission():
    """Approval re-submission doesn't inflate calibration when __call__ re-runs.

    During approval loops, the executor restores history and re-submits with
    approval messages appended.  The strategy's ``__call__`` runs again on
    the full message set (including approval messages), updating
    ``_last_included_tokens`` to match what the API sees.  The calibration
    ratio therefore stays stable.
    """
    messages = _build_multi_turn(3, groups_per_turn=5, result_size=5000)

    strategy = _make_strategy(max_context_tokens=1_000_000)
    await strategy(messages)
    local = strategy.last_included_tokens
    assert local >= 10_000, f"Need substantial local tokens for test, got {local}"

    # First calibrate: learn overhead
    overhead = 5000
    strategy.calibrate(local + overhead)
    assert strategy.system_overhead_tokens == overhead

    # Second calibrate: learn ratio with slight tokenizer drift (1.05x)
    api_tokens = int((local + overhead) * 1.05)
    strategy.calibrate(api_tokens)
    normal_ratio = strategy.calibration_ratio

    # Simulate re-submission: add approval messages, re-run __call__
    approval_msgs = []
    for i in range(3):
        approval_msgs.extend(_build_tool_group(f"approval_{i}", "tool_0", "approved"))
    messages.extend(approval_msgs)

    # __call__ re-runs with ALL messages (including approval), updating
    # _last_included_tokens to include the approval message tokens.
    await strategy(messages)
    local_with_approval = strategy.last_included_tokens
    assert local_with_approval > local

    # Calibrate with the API count that also includes approval tokens.
    api_with_approval = int((local_with_approval + overhead) * 1.05)
    strategy.calibrate(api_with_approval)
    resubmit_ratio = strategy.calibration_ratio

    # Ratio stays stable because both API and local include the same messages.
    assert abs(resubmit_ratio - normal_ratio) < 0.05, (
        f"Re-submission ratio ({resubmit_ratio:.2f}) should ≈ normal ({normal_ratio:.2f})"
    )


async def test_calibration_accurate_for_tool_heavy_sessions():
    """Tool-heavy sessions calibrate correctly using _last_included_tokens.

    The calibration uses ``_last_included_tokens`` (set by ``__call__`` via
    ``kernel.compaction.included_token_count``) as the local baseline. Since
    ``_usage_pct`` multiplies the same ``included_token_count`` by the ratio,
    the estimated API usage tracks the real API token count regardless of
    content type (text vs function_call vs function_result).
    """
    # Build a tool-heavy conversation: multiple read_file + write_file calls
    # with large file content in function_results.  Very little plain text.
    file_body = "x" * 8000  # ~8k chars per file read
    messages: list[Message] = [_user("Read and update the config files")]
    for i in range(6):
        cid_r = f"read_{i}"
        messages.append(_assistant_tool_call(cid_r, "read_file", {"path": f"/src/file_{i}.py"}))
        messages.append(_tool_result(cid_r, file_body))
        cid_w = f"write_{i}"
        messages.append(_assistant_tool_call(cid_w, "write_file", {"path": f"/src/file_{i}.py", "content": file_body}))
        messages.append(_tool_result(cid_w, "File written successfully."))
    messages.append(_assistant_text("Done updating files."))

    strategy = _make_strategy(max_context_tokens=1_000_000)
    await strategy(messages)
    local = strategy.last_included_tokens
    assert local > 10_000, f"Need substantial tokens for tool-heavy test, got {local}"

    # First calibrate: learn overhead
    system_overhead = 5000
    strategy.calibrate(local + system_overhead)
    assert strategy.system_overhead_tokens == system_overhead

    # Second calibrate: ratio should be ~1.0 (no tokenizer drift)
    strategy.calibrate(local + system_overhead)
    assert abs(strategy.calibration_ratio - 1.0) < 0.01, (
        f"Calibration ratio ({strategy.calibration_ratio:.3f}) should be ~1.0 with no drift"
    )

    # _usage_pct should estimate API tokens accurately
    estimated_pct = strategy._usage_pct(local)
    real_pct = (local + system_overhead) / 1_000_000
    assert abs(estimated_pct - real_pct) < 0.001, f"_usage_pct ({estimated_pct:.4f}) should match real ({real_pct:.4f})"


# ---------------------------------------------------------------------------
# Trigger math and admission
# ---------------------------------------------------------------------------


async def test_ratio_affects_trigger():
    """High calibration ratio makes the trigger fire earlier."""
    messages = _build_single_turn(4, result_size=1000)
    total = _estimate_tokens(messages)

    # Without calibration (ratio=1.0): messages fill ~50% of context → no trigger
    strategy = _make_strategy(max_context_tokens=total * 2, trigger_pct=0.85, target_pct=0.30)
    changed = await strategy(messages)
    assert not changed

    # Ratio of 2.0 pushes effective usage to ~100% → triggers compaction
    strategy._calibration_ratio = 2.0
    changed = await strategy(messages)
    assert changed


async def test_request_overhead_floor_triggers_fresh_strategy_earlier() -> None:
    messages = _build_multi_turn(2, groups_per_turn=2, result_size=500)
    total = _estimate_tokens(messages)
    without_floor = _make_strategy(max_context_tokens=total * 2, trigger_pct=0.75, target_pct=0.50)
    with_floor = _make_strategy(max_context_tokens=total * 2, trigger_pct=0.75, target_pct=0.50)

    assert not await without_floor(messages)
    assert await with_floor(messages, CompactionCallContext(request_overhead_tokens=total))


async def test_legacy_tool_definition_tokens_still_raise_the_overhead_floor() -> None:
    """External constructors may populate only the retained legacy field.

    The strategy folds it with ``max()``; reading the new field exclusively
    would size admission as though tools cost zero for such callers.
    """
    messages = _build_multi_turn(2, groups_per_turn=2, result_size=500)
    total = _estimate_tokens(messages)
    strategy = _make_strategy(max_context_tokens=total * 2, trigger_pct=0.75, target_pct=0.50)

    assert await strategy(messages, CompactionCallContext(tool_definition_tokens=total))


def test_compaction_call_context_keeps_legacy_tool_definition_field() -> None:
    context = CompactionCallContext(tool_definition_tokens=123)

    assert context.tool_definition_tokens == 123
    assert context.request_overhead_tokens == 0


def test_strategy_satisfies_admission_protocol() -> None:
    """The runtime protocol gates the outbound clamp: if the real strategy
    stops conforming, the clamp silently no-ops in production while the
    double-based kernel tests keep passing."""
    assert isinstance(_make_strategy(max_context_tokens=100), CompactionAdmissionState)


@pytest.mark.parametrize(
    ("overhead", "ratio"),
    [
        (-1, 1.0),
        (100, 1.0),
        (0, 0.0),
        (0, 1.500_001),
        (0, float("nan")),
        (0, float("inf")),
        (True, 1.0),
        (0, False),
    ],
)
def test_restore_calibration_rejects_invalid_values(overhead: object, ratio: object) -> None:
    strategy = _make_strategy(max_context_tokens=100)

    assert not strategy.restore_calibration(overhead, ratio)
    assert not strategy.calibration_initialized


def test_restore_calibration_hydrates_valid_values() -> None:
    strategy = _make_strategy(max_context_tokens=100)

    assert strategy.restore_calibration(20, 1.25)
    assert strategy.calibration_initialized
    assert strategy.system_overhead_tokens == 20
    assert strategy.calibration_ratio == 1.25

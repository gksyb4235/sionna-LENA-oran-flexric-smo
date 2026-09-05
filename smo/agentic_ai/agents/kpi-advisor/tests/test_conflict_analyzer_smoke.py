"""Smoke tests for ``conflict_analyzer.analyze`` (task 8.1, Requirement 9).

Lightweight sanity coverage only -- Hypothesis property tests for Properties
53/54/55/58 are separate, dedicated tasks (8.2-8.4 and beyond) that exercise
this module far more thoroughly. This module just confirms the main
behaviors work end to end:

- 9.1: clamp to [-100.0, 100.0] and round to 1 decimal place.
- 9.2/9.4: boundary-inclusive Indirect_Conflict detection and exclusion.
- 9.5/9.6/9.9: Conflict_Threshold defaulting/validation.
- 9.7: deterministic sort order.
- 9.8: missing Control_Parameter effect exclusion, rest of the analysis continues.
- 9.10: fewer than 2 valid Control_Parameters -> empty list + reason.

Follows ``test_contract.py``/``test_agent_baseline_and_probe_plan.py``'s manual
``sys.path`` setup convention since ``kpi-advisor`` is not installed as a
package during tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(AGENT_DIR))

from smo.aimlfw.common.models import MarginalEffectRecord  # noqa: E402

from conflict_analyzer import analyze  # noqa: E402


def _record(cell_id: str, control_parameter: str, target_kpi: str, value_percent: float | None) -> MarginalEffectRecord:
    return MarginalEffectRecord(
        cell_id=cell_id, control_parameter=control_parameter, target_kpi=target_kpi, value_percent=value_percent
    )


def test_boundary_inclusive_conflict_is_detected() -> None:
    """9.2/9.4: exactly +threshold and -threshold is still a conflict (boundary inclusive)."""
    effects = [
        _record("gNB_5G", "tx_power_dbm", "cell_goodput_mbps", 5.0),
        _record("gNB_5G", "ret_tilt_deg", "cell_goodput_mbps", -5.0),
    ]
    result = analyze(effects, conflict_threshold=5.0)

    assert len(result.indirect_conflicts) == 1
    conflict = result.indirect_conflicts[0]
    assert conflict.cell_id == "gNB_5G"
    assert conflict.control_parameter_a == "ret_tilt_deg"
    assert conflict.control_parameter_b == "tx_power_dbm"
    assert conflict.marginal_effect_a == -5.0
    assert conflict.marginal_effect_b == 5.0
    assert conflict.target_kpi == "cell_goodput_mbps"
    assert conflict.conflict_threshold == 5.0
    assert result.insufficient_parameters == []
    assert result.unavailable_control_parameters == []


def test_same_sign_and_below_threshold_are_excluded() -> None:
    """9.4: same-sign pairs and pairs where neither/only one exceeds threshold are excluded."""
    effects = [
        # same sign, both above threshold in magnitude -> excluded
        _record("gNB_5G", "tx_power_dbm", "cell_goodput_mbps", 10.0),
        _record("gNB_5G", "ret_tilt_deg", "cell_goodput_mbps", 8.0),
        # opposite sign but both below threshold magnitude -> excluded
        _record("gNB_5G", "cio_bias_db", "cell_goodput_mbps", 2.0),
        _record("gNB_5G", "hysteresis_db", "cell_goodput_mbps", -2.0),
    ]
    result = analyze(effects, conflict_threshold=5.0)

    assert result.indirect_conflicts == []


def test_one_sided_exceedance_is_excluded() -> None:
    """9.4: only one side exceeding the threshold magnitude is excluded."""
    effects = [
        _record("gNB_5G", "tx_power_dbm", "cell_goodput_mbps", 20.0),
        _record("gNB_5G", "ret_tilt_deg", "cell_goodput_mbps", -1.0),
    ]
    result = analyze(effects, conflict_threshold=5.0)

    assert result.indirect_conflicts == []


def test_clamp_and_round_marginal_effect() -> None:
    """9.1: Conflict_Analyzer's own output is clamped to [-100.0, 100.0] and rounded to 1dp."""
    effects = [
        _record("gNB_5G", "tx_power_dbm", "cell_goodput_mbps", 7.26),
        _record("gNB_5G", "ret_tilt_deg", "cell_goodput_mbps", -7.24),
    ]
    result = analyze(effects, conflict_threshold=5.0)

    assert len(result.indirect_conflicts) == 1
    conflict = result.indirect_conflicts[0]
    assert conflict.marginal_effect_a == -7.2
    assert conflict.marginal_effect_b == 7.3


def test_missing_effect_excludes_only_pairs_with_that_parameter() -> None:
    """9.8: a missing (value_percent=None) parameter is excluded, other pairs still analyzed."""
    effects = [
        _record("gNB_5G", "tx_power_dbm", "cell_goodput_mbps", None),
        _record("gNB_5G", "ret_tilt_deg", "cell_goodput_mbps", 6.0),
        _record("gNB_5G", "cio_bias_db", "cell_goodput_mbps", -6.0),
    ]
    result = analyze(effects, conflict_threshold=5.0)

    assert ("gNB_5G", "tx_power_dbm", "cell_goodput_mbps") in result.unavailable_control_parameters
    assert len(result.indirect_conflicts) == 1
    conflict = result.indirect_conflicts[0]
    assert {conflict.control_parameter_a, conflict.control_parameter_b} == {"ret_tilt_deg", "cio_bias_db"}


def test_fewer_than_two_valid_parameters_yields_empty_list_with_reason() -> None:
    """9.10: fewer than 2 usable Control_Parameters for a Target_KPI -> empty list + reason."""
    effects = [_record("gNB_5G", "tx_power_dbm", "cell_goodput_mbps", 6.0)]
    result = analyze(effects, conflict_threshold=5.0)

    assert result.indirect_conflicts == []
    assert len(result.insufficient_parameters) == 1
    reason_entry = result.insufficient_parameters[0]
    assert reason_entry.cell_id == "gNB_5G"
    assert reason_entry.target_kpi == "cell_goodput_mbps"
    assert reason_entry.valid_control_parameter_count == 1
    assert reason_entry.reason


def test_threshold_defaults_when_unset_without_warning() -> None:
    """9.6: unset Conflict_Threshold defaults to 5.0 without a warning."""
    result = analyze([], conflict_threshold=None)

    assert result.conflict_threshold == 5.0
    assert result.threshold_warning is None


def test_threshold_defaults_with_warning_when_out_of_range() -> None:
    """9.9: an out-of-range Conflict_Threshold defaults to 5.0 with a warning."""
    result = analyze([], conflict_threshold=500.0)

    assert result.conflict_threshold == 5.0
    assert result.threshold_warning is not None


def test_valid_threshold_is_used_as_is() -> None:
    """9.5: a Conflict_Threshold within [0.1, 100.0] is used as specified."""
    result = analyze([], conflict_threshold=10.0)

    assert result.conflict_threshold == 10.0
    assert result.threshold_warning is None


def test_deterministic_sort_order() -> None:
    """9.7: sorted by Target_KPI ascending, then the two Control_Parameter names ascending."""
    effects = [
        # target_kpi "sinr_p50_db", pair (ret_tilt_deg, tx_power_dbm)
        _record("gNB_5G", "tx_power_dbm", "sinr_p50_db", 6.0),
        _record("gNB_5G", "ret_tilt_deg", "sinr_p50_db", -6.0),
        # target_kpi "cell_goodput_mbps", pair (cio_bias_db, hysteresis_db)
        _record("gNB_5G", "hysteresis_db", "cell_goodput_mbps", 6.0),
        _record("gNB_5G", "cio_bias_db", "cell_goodput_mbps", -6.0),
    ]
    result = analyze(effects, conflict_threshold=5.0)

    assert [(c.target_kpi, c.control_parameter_a, c.control_parameter_b) for c in result.indirect_conflicts] == [
        ("cell_goodput_mbps", "cio_bias_db", "hysteresis_db"),
        ("sinr_p50_db", "ret_tilt_deg", "tx_power_dbm"),
    ]

    # Re-running with the same input and threshold reproduces the identical order (idempotence).
    repeat_result = analyze(effects, conflict_threshold=5.0)
    assert result.indirect_conflicts == repeat_result.indirect_conflicts

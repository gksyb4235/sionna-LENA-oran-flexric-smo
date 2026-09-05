"""Task 7.3 unit tests: Threshold_KPI judgment (Requirement 8) and ``run_advisor`` wiring.

Follows ``test_agent_baseline_and_probe_plan.py`` (task 7.2)'s manual
``sys.path`` setup convention and fake/injectable Feature_Store pattern.

Task 7.3 has no dedicated Property test of its own (Properties 48-52 for
Requirement 8 belong to the separate tasks 7.11-7.15), so this module is
plain focused unit-test coverage, mirroring task 7.2's own test module.

Requirement 8.1/8.10 (임계값 설정 검증):
    설정 파일에서 Threshold_KPI 정의를 로드하고 각 Target_KPI 에 하한/상한/개선
    방향 중 하나 이상이 정의되어 있는지 확인한다. 로드/검증이 실패하면 판정을
    수행하지 않고 오류를 반환한다.

Requirement 8.2/8.3 (경계 포함 판정):
    모든 Target_KPI 가 허용 범위 내(경계값 포함)이면 acceptable, 하나 이상
    위반이면 degrading (위반된 Target_KPI 최대 10개 보고).

Requirement 8.4/8.12 (unknown 판정):
    신뢰 구간 정보가 없으면(오늘날의 실제 Inference_Service 응답) unknown.

Requirement 8.5-8.8 (추천 후보 필터링과 결정론적 정렬):
    Threshold_KPI 를 모두 만족하고 주 Target_KPI 가 1.0% 이상 개선된 variant만
    후보. 2개 이상이면 개선폭 내림차순(동률은 식별자 오름차순) 정렬, 상위 5개.
    정확히 1개면 최상위 추천. 0개면 제외 사유를 포함한 빈 목록.

Requirement 8.11 (누락 예측값의 부분 배제):
    일부 Probe_Variant 의 예측값이 누락되면 해당 variant만 제외하고 나머지는
    계속 판정한다.

Requirement 7.9/12.6-12.8:
    ``run_advisor`` 는 Policy_Manager/A1 변경 경로를 호출하지 않고, 한국어
    의도 입력에 한국어 근거를 반환한다.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

AGENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(AGENT_DIR))

from smo.aimlfw.common.config import (  # noqa: E402
    ObjectivePrimaryKpiConfig,
    ObjectivePrimaryKpiEntry,
)
from smo.aimlfw.common.constants import TARGET_KPIS  # noqa: E402
from smo.aimlfw.common.models import (  # noqa: E402
    CellParameters,
    FeatureRecord,
    ParameterSet,
    SampleCount,
    ThresholdKpi,
)

import agent  # noqa: E402
from schemas import InvokeRequest, PartialCellParameters, XAppParameterRequest  # noqa: E402

_BASE_CONTROL_PARAMETERS: dict[str, float | int] = {
    "tx_power_dbm": 40.0,
    "ret_tilt_deg": 5.0,
    "cio_bias_db": 0.0,
    "hysteresis_db": 2.0,
    "ttt_ms": 160,
}


def _cell_parameters(**overrides: float | int) -> dict[str, float | int]:
    values = dict(_BASE_CONTROL_PARAMETERS)
    values.update(overrides)
    return values


def _all_acceptable_target_kpi(**overrides: float) -> dict[str, float]:
    """A Target_KPI value map that satisfies every kpi_thresholds.json bound."""
    values = {
        "cell_goodput_mbps": 100.0,
        "avg_ue_goodput_mbps": 50.0,
        "ue_goodput_p5_mbps": 10.0,
        "sinr_p50_db": 10.0,
        "prb_utilization_pct": 50.0,
        "delay_p95_ms": 100.0,
        "ho_failure_count": 0.0,
        "pingpong_count": 0.0,
        "rlf_count": 0.0,
        "interval_energy_j": 10.0,
    }
    assert set(values) == set(TARGET_KPIS)
    values.update(overrides)
    return values


def _threshold_kpi(name: str, *, lower: float | None, upper: float | None, direction: str) -> ThresholdKpi:
    return ThresholdKpi(target_kpi=name, lower_bound=lower, upper_bound=upper, improve_direction=direction)


def _default_kpi_threshold_config() -> agent.KpiThresholdConfig:
    """A minimal, valid KpiThresholdConfig covering every canonical Target_KPI."""
    thresholds = {
        "cell_goodput_mbps": _threshold_kpi("cell_goodput_mbps", lower=0.0, upper=None, direction="higher_is_better"),
        "avg_ue_goodput_mbps": _threshold_kpi(
            "avg_ue_goodput_mbps", lower=0.0, upper=None, direction="higher_is_better"
        ),
        "ue_goodput_p5_mbps": _threshold_kpi(
            "ue_goodput_p5_mbps", lower=0.0, upper=None, direction="higher_is_better"
        ),
        "sinr_p50_db": _threshold_kpi("sinr_p50_db", lower=-20.0, upper=None, direction="higher_is_better"),
        "prb_utilization_pct": _threshold_kpi(
            "prb_utilization_pct", lower=0.0, upper=100.0, direction="lower_is_better"
        ),
        "delay_p95_ms": _threshold_kpi("delay_p95_ms", lower=None, upper=1000.0, direction="lower_is_better"),
        "ho_failure_count": _threshold_kpi("ho_failure_count", lower=0.0, upper=None, direction="lower_is_better"),
        "pingpong_count": _threshold_kpi("pingpong_count", lower=0.0, upper=None, direction="lower_is_better"),
        "rlf_count": _threshold_kpi("rlf_count", lower=0.0, upper=None, direction="lower_is_better"),
        "interval_energy_j": _threshold_kpi(
            "interval_energy_j", lower=0.0, upper=None, direction="lower_is_better"
        ),
    }
    return agent.KpiThresholdConfig(thresholds=thresholds)


def _default_objective_primary_kpi_config() -> ObjectivePrimaryKpiConfig:
    return ObjectivePrimaryKpiConfig(
        objectives={
            "energy_saving": ObjectivePrimaryKpiEntry(
                target_kpi="interval_energy_j", improve_direction="lower_is_better"
            ),
            "throughput_maximization": ObjectivePrimaryKpiEntry(
                target_kpi="cell_goodput_mbps", improve_direction="higher_is_better"
            ),
            "mobility_robustness": ObjectivePrimaryKpiEntry(
                target_kpi="ho_failure_count", improve_direction="lower_is_better"
            ),
        }
    )


def _prediction(**overrides: float) -> agent.ProbePredictionResult:
    return agent.ProbePredictionResult(
        target_kpi=_all_acceptable_target_kpi(**overrides), cell_kpi={"gNB_5G": _all_acceptable_target_kpi(**overrides)}
    )


def _simple_baseline(cell_id: str = "gNB_5G") -> ParameterSet:
    return ParameterSet(cells={cell_id: CellParameters.model_validate(_cell_parameters())})


def _variant(baseline: ParameterSet, *, cell_id: str, parameter_name: str, direction: str, variant_id: str) -> Any:
    from smo.aimlfw.common.models import ProbeVariant

    new_value = agent.step_value(parameter_name, getattr(baseline.cells[cell_id], parameter_name), direction)
    updated_cells = dict(baseline.cells)
    updated_cells[cell_id] = baseline.cells[cell_id].model_copy(update={parameter_name: new_value})
    return ProbeVariant(
        variant_id=variant_id,
        base_parameter_set_id="baseline-1",
        cell_id=cell_id,
        parameter_name=parameter_name,
        direction=direction,
        parameter_set=ParameterSet(cells=updated_cells),
    )


def _variant_result(variant: Any, *, cell_goodput_mbps: float, improvement_percent: float) -> agent.ProbeVariantResult:
    return agent.ProbeVariantResult(
        variant=variant,
        prediction=_prediction(cell_goodput_mbps=cell_goodput_mbps),
        percent_change={"cell_goodput_mbps": improvement_percent},
    )


def _recommend(
    results: list[agent.ProbeVariantResult],
    thresholds: agent.KpiThresholdConfig,
    objective_primary_kpi: ObjectivePrimaryKpiConfig,
    *,
    xapp_objective: str | None = "throughput_maximization",
) -> agent.RecommendationResult:
    return agent.build_recommendations(
        results, thresholds, xapp_objective=xapp_objective, objective_primary_kpi=objective_primary_kpi
    )


# ---------------------------------------------------------------------------
# Requirement 8.1/8.10: Threshold_KPI configuration validation.
# ---------------------------------------------------------------------------


def test_load_threshold_config_returns_the_real_configuration_file_successfully() -> None:
    # smo/aimlfw/config/kpi_thresholds.json is a real, already-valid file (task 1.1) --
    # this confirms load_threshold_config() reuses load_kpi_thresholds() end-to-end.
    thresholds = agent.load_threshold_config()
    assert set(thresholds.thresholds) == set(TARGET_KPIS)


def test_load_threshold_config_wraps_configuration_error_as_probe_execution_error(tmp_path: Path) -> None:
    bad_path = tmp_path / "kpi_thresholds.json"
    bad_path.write_text("{}", encoding="utf-8")  # missing every Target_KPI entry.

    with pytest.raises(agent.ProbeExecutionError) as excinfo:
        agent.load_threshold_config(bad_path)
    assert excinfo.value.error_code == "threshold_configuration_invalid"
    assert "reason" in excinfo.value.details


def test_load_threshold_config_reports_lower_bound_exceeding_upper_bound(tmp_path: Path) -> None:
    payload = {
        name: {"lower_bound": 10.0, "upper_bound": 0.0, "improve_direction": "higher_is_better"}
        for name in TARGET_KPIS
    }
    bad_path = tmp_path / "kpi_thresholds.json"
    bad_path.write_text(__import__("json").dumps(payload), encoding="utf-8")

    with pytest.raises(agent.ProbeExecutionError) as excinfo:
        agent.load_threshold_config(bad_path)
    assert excinfo.value.error_code == "threshold_configuration_invalid"
    assert excinfo.value.details["target_kpi"] in TARGET_KPIS


def test_load_objective_primary_kpi_config_returns_the_real_configuration_file_successfully() -> None:
    config = agent.load_objective_primary_kpi_config()
    assert set(config.objectives) == {"energy_saving", "throughput_maximization", "mobility_robustness"}


# ---------------------------------------------------------------------------
# Requirement 8.2/8.3: acceptable/degrading, boundary-inclusive.
# ---------------------------------------------------------------------------


def test_judge_degradation_verdict_is_acceptable_when_every_bound_is_satisfied_inclusive_of_boundaries() -> None:
    thresholds = _default_kpi_threshold_config()
    # Exactly at the boundary values for every bounded Target_KPI.
    values = _all_acceptable_target_kpi(
        cell_goodput_mbps=0.0,
        avg_ue_goodput_mbps=0.0,
        ue_goodput_p5_mbps=0.0,
        sinr_p50_db=-20.0,
        prb_utilization_pct=100.0,
        delay_p95_ms=1000.0,
        ho_failure_count=0.0,
        pingpong_count=0.0,
        rlf_count=0.0,
        interval_energy_j=0.0,
    )
    # A narrow, present confidence interval is required for "acceptable" to actually
    # surface as the final verdict: per 8.12, an *absent* CI (the default from
    # _prediction()) always yields "unknown" once there are 0 violations, so this
    # test must supply CI data to observe 8.2's acceptable branch in isolation.
    prediction = agent.ProbePredictionResult(
        target_kpi=values,
        cell_kpi={"gNB_5G": values},
        confidence_interval_width={name: 0.1 * abs(value) if value else 0.1 for name, value in values.items()},
    )

    judgment = agent.judge_degradation_verdict(prediction, thresholds)

    assert judgment.verdict == "acceptable"
    assert judgment.violations == []


def test_judge_degradation_verdict_is_degrading_when_one_target_kpi_is_below_lower_bound() -> None:
    thresholds = _default_kpi_threshold_config()
    prediction = _prediction(cell_goodput_mbps=-1.0)

    judgment = agent.judge_degradation_verdict(prediction, thresholds)

    assert judgment.verdict == "degrading"
    assert len(judgment.violations) == 1
    violation = judgment.violations[0]
    assert violation.target_kpi == "cell_goodput_mbps"
    assert violation.predicted_value == -1.0
    assert violation.direction == "below_lower"


def test_judge_degradation_verdict_is_degrading_when_one_target_kpi_is_above_upper_bound() -> None:
    thresholds = _default_kpi_threshold_config()
    prediction = _prediction(delay_p95_ms=1000.5)

    judgment = agent.judge_degradation_verdict(prediction, thresholds)

    assert judgment.verdict == "degrading"
    assert judgment.violations[0].direction == "above_upper"


def test_judge_degradation_verdict_reports_at_most_ten_violations() -> None:
    thresholds = _default_kpi_threshold_config()
    # Every one of the 10 canonical Target_KPIs violates its lower bound (all bounded below by 0 or -20).
    prediction = _prediction(
        cell_goodput_mbps=-1.0,
        avg_ue_goodput_mbps=-1.0,
        ue_goodput_p5_mbps=-1.0,
        sinr_p50_db=-21.0,
        ho_failure_count=-1.0,
        pingpong_count=-1.0,
        rlf_count=-1.0,
        interval_energy_j=-1.0,
        prb_utilization_pct=-1.0,
        delay_p95_ms=1000.5,
    )

    judgment = agent.judge_degradation_verdict(prediction, thresholds)

    assert judgment.verdict == "degrading"
    assert len(judgment.violations) == len(TARGET_KPIS) == 10
    assert len(judgment.violations) <= agent.MAX_REPORTED_VIOLATIONS


# ---------------------------------------------------------------------------
# Requirement 8.4/8.12: unknown verdict.
# ---------------------------------------------------------------------------


def test_judge_degradation_verdict_is_unknown_when_no_confidence_interval_information_is_present() -> None:
    """8.12: today's real Inference_Service never returns CI info -> always unknown when no violations."""
    thresholds = _default_kpi_threshold_config()
    prediction = _prediction()  # confidence_interval_width defaults to None.

    judgment = agent.judge_degradation_verdict(prediction, thresholds)

    assert judgment.verdict == "unknown"
    assert len(judgment.unknown_reasons) == len(TARGET_KPIS)
    assert all(reason.reason == "confidence_interval_absent" for reason in judgment.unknown_reasons)


def test_judge_degradation_verdict_is_acceptable_when_confidence_interval_is_present_and_narrow() -> None:
    """8.4's CI-present branch: a narrow CI (<=50% of predicted value) does not trigger unknown."""
    thresholds = _default_kpi_threshold_config()
    values = _all_acceptable_target_kpi()
    prediction = agent.ProbePredictionResult(
        target_kpi=values,
        cell_kpi={"gNB_5G": values},
        confidence_interval_width={name: 0.1 * abs(value) if value else 0.1 for name, value in values.items()},
    )

    judgment = agent.judge_degradation_verdict(prediction, thresholds)

    assert judgment.verdict == "acceptable"


def test_judge_degradation_verdict_is_unknown_when_confidence_interval_is_wide_and_no_actual_evidence() -> None:
    """8.4's CI-present branch: a >50%-of-value-wide CI with no matching Feature_Store evidence -> unknown."""
    thresholds = _default_kpi_threshold_config()
    values = _all_acceptable_target_kpi()
    prediction = agent.ProbePredictionResult(
        target_kpi=values,
        cell_kpi={"gNB_5G": values},
        confidence_interval_width={name: 10.0 * abs(value) if value else 10.0 for name, value in values.items()},
    )

    class _EmptyFeatureStore:
        def records(self, *_args: Any, **_kwargs: Any) -> list[FeatureRecord]:
            return []

    judgment = agent.judge_degradation_verdict(
        prediction,
        thresholds,
        feature_store=_EmptyFeatureStore(),
        feature_group="default",
        time_step=0,
        target_cells=["gNB_5G"],
    )

    assert judgment.verdict == "unknown"
    assert all(reason.reason == "confidence_interval_too_wide" for reason in judgment.unknown_reasons)


def test_judge_degradation_verdict_degrading_takes_precedence_over_unknown() -> None:
    """8.2/8.3 gate 8.4/8.12: a violation always yields degrading, never unknown, regardless of CI absence."""
    thresholds = _default_kpi_threshold_config()
    prediction = _prediction(cell_goodput_mbps=-5.0)  # violation + no CI info at all.

    judgment = agent.judge_degradation_verdict(prediction, thresholds)

    assert judgment.verdict == "degrading"


# ---------------------------------------------------------------------------
# Requirement 8.5-8.8: recommendation candidacy, ranking, and exclusion.
# ---------------------------------------------------------------------------


def test_build_recommendations_returns_a_single_candidate_as_the_sole_recommendation() -> None:
    thresholds = _default_kpi_threshold_config()
    objective_primary_kpi = _default_objective_primary_kpi_config()
    baseline = _simple_baseline()
    variant = _variant(baseline, cell_id="gNB_5G", parameter_name="tx_power_dbm", direction="+1", variant_id="v1")
    variant_result = _variant_result(variant, cell_goodput_mbps=120.0, improvement_percent=20.0)

    result = _recommend([variant_result], thresholds, objective_primary_kpi)

    assert len(result.recommendations) == 1
    assert result.recommendations[0].variant_id == "v1"
    assert result.recommendations[0].improvement_percent == 20.0
    assert result.excluded == []


def test_build_recommendations_ranks_two_or_more_candidates_by_improvement_descending() -> None:
    thresholds = _default_kpi_threshold_config()
    objective_primary_kpi = _default_objective_primary_kpi_config()
    baseline = _simple_baseline()
    variant_a = _variant(baseline, cell_id="gNB_5G", parameter_name="tx_power_dbm", direction="+1", variant_id="v-a")
    variant_b = _variant(baseline, cell_id="gNB_5G", parameter_name="ret_tilt_deg", direction="+1", variant_id="v-b")
    results = [
        _variant_result(variant_a, cell_goodput_mbps=105.0, improvement_percent=5.0),
        _variant_result(variant_b, cell_goodput_mbps=130.0, improvement_percent=30.0),
    ]

    result = _recommend(results, thresholds, objective_primary_kpi)

    assert [recommendation.variant_id for recommendation in result.recommendations] == ["v-b", "v-a"]


def test_build_recommendations_breaks_ties_within_0_1_percent_by_variant_id_ascending() -> None:
    thresholds = _default_kpi_threshold_config()
    objective_primary_kpi = _default_objective_primary_kpi_config()
    baseline = _simple_baseline()
    variant_z = _variant(
        baseline, cell_id="gNB_5G", parameter_name="tx_power_dbm", direction="+1", variant_id="z-variant"
    )
    variant_a = _variant(
        baseline, cell_id="gNB_5G", parameter_name="ret_tilt_deg", direction="+1", variant_id="a-variant"
    )
    # Both improve by exactly 10.0% (already-rounded percent change) -- an exact tie.
    results = [
        _variant_result(variant_z, cell_goodput_mbps=110.0, improvement_percent=10.0),
        _variant_result(variant_a, cell_goodput_mbps=110.0, improvement_percent=10.0),
    ]

    result = _recommend(results, thresholds, objective_primary_kpi)

    assert [recommendation.variant_id for recommendation in result.recommendations] == ["a-variant", "z-variant"]


def test_build_recommendations_returns_top_five_of_more_than_five_candidates() -> None:
    thresholds = _default_kpi_threshold_config()
    objective_primary_kpi = _default_objective_primary_kpi_config()
    baseline = _simple_baseline()
    results = []
    for index, improvement in enumerate((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)):
        variant = agent.ProbeVariant(
            variant_id=f"v-{index}",
            base_parameter_set_id="baseline-1",
            cell_id="gNB_5G",
            parameter_name="tx_power_dbm",
            direction="+1",
            parameter_set=baseline,
        )
        results.append(
            _variant_result(
                variant, cell_goodput_mbps=100.0 * (1 + improvement / 100), improvement_percent=improvement
            )
        )

    result = _recommend(results, thresholds, objective_primary_kpi)

    assert len(result.recommendations) == agent.MAX_RECOMMENDATIONS == 5
    improvements = [recommendation.improvement_percent for recommendation in result.recommendations]
    assert improvements == [7.0, 6.0, 5.0, 4.0, 3.0]


def test_build_recommendations_excludes_variant_that_violates_a_threshold() -> None:
    thresholds = _default_kpi_threshold_config()
    objective_primary_kpi = _default_objective_primary_kpi_config()
    baseline = _simple_baseline()
    variant = _variant(baseline, cell_id="gNB_5G", parameter_name="tx_power_dbm", direction="+1", variant_id="v1")
    variant_result = agent.ProbeVariantResult(
        variant=variant,
        prediction=_prediction(cell_goodput_mbps=120.0, rlf_count=-1.0),  # violates rlf_count >= 0.
        percent_change={"cell_goodput_mbps": 20.0},
    )

    result = _recommend([variant_result], thresholds, objective_primary_kpi)

    assert result.recommendations == []
    assert len(result.excluded) == 1
    assert result.excluded[0].reason == "threshold_violation"


def test_build_recommendations_excludes_variant_below_the_1_percent_improvement_threshold() -> None:
    thresholds = _default_kpi_threshold_config()
    objective_primary_kpi = _default_objective_primary_kpi_config()
    baseline = _simple_baseline()
    variant = _variant(baseline, cell_id="gNB_5G", parameter_name="tx_power_dbm", direction="+1", variant_id="v1")
    # improvement_percent=0.5 is below the 1.0% minimum.
    variant_result = _variant_result(variant, cell_goodput_mbps=100.5, improvement_percent=0.5)

    result = _recommend([variant_result], thresholds, objective_primary_kpi)

    assert result.recommendations == []
    assert result.excluded[0].reason == "primary_kpi_improvement_below_threshold"


def test_build_recommendations_returns_empty_with_no_objective_specified() -> None:
    """Design decision: Requirement 8.5's conjunction requires an xApp_Objective for candidacy at all."""
    thresholds = _default_kpi_threshold_config()
    objective_primary_kpi = _default_objective_primary_kpi_config()
    baseline = _simple_baseline()
    variant = _variant(baseline, cell_id="gNB_5G", parameter_name="tx_power_dbm", direction="+1", variant_id="v1")
    variant_result = _variant_result(variant, cell_goodput_mbps=200.0, improvement_percent=100.0)

    result = _recommend([variant_result], thresholds, objective_primary_kpi, xapp_objective=None)

    assert result.recommendations == []
    assert result.excluded[0].reason == "no_objective_specified"

    result_unspecified = _recommend([variant_result], thresholds, objective_primary_kpi, xapp_objective="unspecified")
    assert result_unspecified.recommendations == []
    assert result_unspecified.excluded[0].reason == "no_objective_specified"


# ---------------------------------------------------------------------------
# Requirement 8.11: missing prediction excludes only that variant.
# ---------------------------------------------------------------------------


def test_build_recommendations_excludes_only_the_variant_with_a_missing_target_kpi_prediction() -> None:
    thresholds = _default_kpi_threshold_config()
    objective_primary_kpi = _default_objective_primary_kpi_config()
    baseline = _simple_baseline()

    good_variant = _variant(
        baseline, cell_id="gNB_5G", parameter_name="tx_power_dbm", direction="+1", variant_id="v-good"
    )
    incomplete_prediction_values = _all_acceptable_target_kpi(cell_goodput_mbps=150.0)
    del incomplete_prediction_values["rlf_count"]  # simulate a missing Target_KPI prediction.
    broken_variant = _variant(
        baseline, cell_id="gNB_5G", parameter_name="ret_tilt_deg", direction="+1", variant_id="v-broken"
    )

    results = [
        agent.ProbeVariantResult(
            variant=good_variant,
            prediction=_prediction(cell_goodput_mbps=150.0),
            percent_change={"cell_goodput_mbps": 50.0},
        ),
        agent.ProbeVariantResult(
            variant=broken_variant,
            prediction=agent.ProbePredictionResult(
                target_kpi=incomplete_prediction_values, cell_kpi={"gNB_5G": incomplete_prediction_values}
            ),
            percent_change={"cell_goodput_mbps": 50.0},
        ),
    ]

    result = agent.build_recommendations(
        results, thresholds, xapp_objective="throughput_maximization", objective_primary_kpi=objective_primary_kpi
    )

    assert [recommendation.variant_id for recommendation in result.recommendations] == ["v-good"]
    assert len(result.excluded) == 1
    assert result.excluded[0].variant_id == "v-broken"
    assert result.excluded[0].reason == "missing_prediction"


# ---------------------------------------------------------------------------
# End-to-end run_advisor: baseline_parameter_set path, xapp_request path, and
# Korean rationale generation.
# ---------------------------------------------------------------------------


def _feature_record(
    *, cell_id: str, time_step: int, control_parameters: dict[str, float | int]
) -> FeatureRecord:
    kpi_names = TARGET_KPIS
    return FeatureRecord(
        feature_group="default",
        time_step=time_step,
        cell_id=cell_id,
        control_parameters=CellParameters.model_validate(control_parameters),
        features=dict.fromkeys(kpi_names, 1.0),
        sample_counts={name: SampleCount(valid=1, excluded=0) for name in kpi_names},
        data_quality="complete",
        source_dir="/tmp/results/example",
        seed=0,
    )


class _FakeFeatureStore:
    """Duck-typed Feature_Store stand-in: only implements ``.records(...)``."""

    def __init__(self, records: list[FeatureRecord]) -> None:
        self._records = records

    def records(
        self, feature_group: str, *, cell_id: str | None = None, time_step: int | None = None
    ) -> list[FeatureRecord]:
        return [
            record
            for record in self._records
            if record.feature_group == feature_group
            and (cell_id is None or record.cell_id == cell_id)
            and (time_step is None or record.time_step == time_step)
        ]


def _acceptable_batch_response(*, parameter_sets: list[dict[str, Any]], **_kwargs: Any) -> dict[str, Any]:
    values = _all_acceptable_target_kpi()
    return {
        "model_name": "gnn",
        "model_version": 1,
        "applied_time_step": 0,
        "predictions": [
            {"index": index, "target_kpi": values, "cell_kpi": {"gNB_5G": values}}
            for index in range(len(parameter_sets))
        ],
    }


@pytest.fixture(autouse=True)
def _ready_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KPI_ADVISOR_MODEL_NAME", "ran-gnn")
    monkeypatch.setenv("KPI_ADVISOR_MODEL_VERSION", "1")
    # Task 7 tests isolate judgment from Task 12's real MongoDB boundary.
    monkeypatch.setattr(agent, "persist_evidence", lambda record: record)


def _patch_predict_batch(monkeypatch: pytest.MonkeyPatch, stub: Any) -> None:
    """Swap the ``predict_batch`` callable used by ``execute_probe_plan``/``build_and_execute_probe_plan``.

    Both functions declare ``predict_batch: Callable[...] = mcp_server.predict_batch``
    as a *keyword-only* default -- bound once, to the function object itself, at
    ``agent.py`` import time. ``run_advisor`` calls both without passing
    ``predict_batch`` explicitly, so simply reassigning the ``mcp_server.predict_batch``
    module attribute afterward has no effect on either function's already-bound
    default; the actual default lives in each function's ``__kwdefaults__`` and
    must be patched there instead.
    """
    monkeypatch.setitem(agent.execute_probe_plan.__kwdefaults__, "predict_batch", stub)
    monkeypatch.setitem(agent.build_and_execute_probe_plan.__kwdefaults__, "predict_batch", stub)


def test_run_advisor_end_to_end_with_baseline_parameter_set_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "get_feature_store", lambda: _FakeFeatureStore([]))
    _patch_predict_batch(monkeypatch, _acceptable_batch_response)

    request = InvokeRequest(
        intent="Evaluate the KPI impact of this parameter combination.",
        baseline_parameter_set=_simple_baseline(),
        time_step=1,
    )

    response = asyncio.run(agent.run_advisor(request))

    assert response.degradation_verdict == "unknown"  # no CI info from the stub predictor (8.12).
    assert response.evidence_record_id
    assert 1 <= len(response.rationale_summary) <= 2000


def test_run_advisor_end_to_end_with_xapp_request_path(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _FakeFeatureStore(
        [_feature_record(cell_id="gNB_5G", time_step=0, control_parameters=_cell_parameters())]
    )
    monkeypatch.setattr(agent, "get_feature_store", lambda: store)
    _patch_predict_batch(monkeypatch, _acceptable_batch_response)

    request = InvokeRequest(
        intent="Evaluate this xApp parameter request.",
        xapp_request=XAppParameterRequest(
            target_cells=["gNB_5G"],
            parameter_overrides={"gNB_5G": PartialCellParameters(tx_power_dbm=41.0)},
        ),
    )

    response = asyncio.run(agent.run_advisor(request))

    assert response.degradation_verdict == "unknown"
    assert response.evidence_record_id


def test_run_advisor_returns_korean_rationale_for_korean_intent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "get_feature_store", lambda: _FakeFeatureStore([]))
    _patch_predict_batch(monkeypatch, _acceptable_batch_response)

    request = InvokeRequest(
        intent="이 파라미터 조합의 KPI 영향을 평가해 주세요.",
        baseline_parameter_set=_simple_baseline(),
    )

    response = asyncio.run(agent.run_advisor(request))

    from schemas import intent_is_korean, validate_response_for_intent

    assert intent_is_korean(request.intent)
    validate_response_for_intent(response, request.intent)  # raises if rationale is not Korean.


def test_run_advisor_raises_advisor_not_ready_when_no_model_is_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KPI_ADVISOR_MODEL_NAME", raising=False)
    monkeypatch.delenv("KPI_ADVISOR_MODEL_VERSION", raising=False)

    request = InvokeRequest(intent="평가", baseline_parameter_set=_simple_baseline())

    with pytest.raises(agent.AdvisorNotReadyError):
        asyncio.run(agent.run_advisor(request))


def test_run_advisor_never_imports_a_policy_manager_or_a1_mediator_module() -> None:
    """Requirement 7.9/12.6-12.8: this module must never reach a Policy_Manager/A1 mutation path.

    Checks actual imports rather than searching the whole module source: the
    module's own docstrings and comments legitimately *mention* Policy_Manager
    by name (documenting why it is deliberately not called), which a plain
    substring search over ``inspect.getsource`` would misflag.
    """
    imported_names = {
        getattr(value, "__name__", "") for value in vars(agent).values() if getattr(value, "__name__", None)
    }
    assert not any("policy_manager" in name.lower() for name in imported_names)
    assert not any("a1_mediator" in name.lower() for name in imported_names)
    # Independent Policy Manager tests may have loaded that module into the
    # shared pytest process. The read-only invariant belongs to ``agent``'s
    # own namespace, not the process-wide import cache.
    assert "policy_manager" not in vars(agent)

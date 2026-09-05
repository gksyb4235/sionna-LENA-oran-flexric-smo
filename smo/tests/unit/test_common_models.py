from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from smo.aimlfw.common.errors import ErrorResponse
from smo.aimlfw.common.models import (
    CellParameters,
    EvidenceRecord,
    FeatureRecord,
    MarginalEffectRecord,
    ModelVersionRecord,
    ParameterSet,
    ProbePlan,
    ProbeVariant,
    SampleCount,
    StageRecord,
    TemporalPlan,
    ThresholdKpi,
    TimeStepAssignment,
    TrainingJob,
)


def cell_parameters(**changes: object) -> CellParameters:
    values = {
        "tx_power_dbm": 43.0,
        "ret_tilt_deg": 5.0,
        "cio_bias_db": 0.5,
        "hysteresis_db": 2.5,
        "ttt_ms": 160,
    }
    values.update(changes)
    return CellParameters.model_validate(values)


def parameter_set() -> ParameterSet:
    return ParameterSet(cells={"gNB_5G": cell_parameters()})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tx_power_dbm", 29),
        ("ret_tilt_deg", 1.5),
        ("cio_bias_db", 0.25),
        ("hysteresis_db", 10.5),
        ("ttt_ms", 200),
    ],
)
def test_control_parameter_range_step_and_allowed_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        cell_parameters(**{field: value})


def test_parameter_set_requires_a_complete_nonempty_cell_map() -> None:
    with pytest.raises(ValidationError):
        ParameterSet(cells={})
    with pytest.raises(ValidationError):
        ParameterSet.model_validate({"cells": {"cell": {"tx_power_dbm": 43}}})


def test_feature_record_enforces_time_step_counts_and_quality() -> None:
    record = FeatureRecord(
        feature_group="default",
        time_step=4,
        cell_id="gNB_5G",
        control_parameters=cell_parameters(),
        features={"sinr_p50_db": None},
        sample_counts={"sinr_p50_db": SampleCount(valid=0, excluded=2)},
        data_quality="insufficient",
        source_dir="/tmp/result",
        seed=12,
    )
    assert record.data_quality == "insufficient"
    with pytest.raises(ValidationError):
        record.model_copy(update={"time_step": 5}, deep=True).__class__.model_validate(
            {**record.model_dump(), "time_step": 5}
        )
    with pytest.raises(ValidationError):
        FeatureRecord.model_validate({**record.model_dump(), "data_quality": "partial"})


def test_probe_plan_requires_consistent_baseline_references() -> None:
    baseline = parameter_set()
    variant = ProbeVariant(
        variant_id="v1",
        base_parameter_set_id="baseline",
        cell_id="gNB_5G",
        parameter_name="tx_power_dbm",
        direction="+1",
        parameter_set=ParameterSet(cells={"gNB_5G": cell_parameters(tx_power_dbm=44)}),
    )
    plan = ProbePlan(
        baseline_id="baseline", baseline=baseline, target_cells=["gNB_5G"], variants=[variant]
    )
    assert plan.variants[0].variant_id == "v1"
    with pytest.raises(ValidationError):
        ProbePlan(
            baseline_id="different", baseline=baseline, target_cells=["gNB_5G"], variants=[variant]
        )


def assignment(step: int, objective: str = "energy_saving") -> TimeStepAssignment:
    return TimeStepAssignment.model_validate(
        {
            "time_step": step,
            "xapp_objective": objective,
            "parameter_set": parameter_set().model_dump(),
            "predicted_kpis": {"interval_energy_j": {"gNB_5G": 10.0}},
            "model_name": "gnn",
            "model_version": 1,
            "threshold_violations": [],
        }
    )


def test_temporal_plan_requires_exact_order_and_transitions() -> None:
    assignments = [assignment(step, "energy_saving" if step < 3 else "mobility_robustness") for step in range(5)]
    plan = TemporalPlan(
        assignments=assignments,
        aggregate_kpis={"interval_energy_j": 50.0},
        transition_time_steps=[3],
    )
    assert len(plan.assignments) == 5
    with pytest.raises(ValidationError):
        TemporalPlan(assignments=assignments, aggregate_kpis={}, transition_time_steps=[])


def test_training_job_and_stage_status_contracts() -> None:
    now = datetime.now(timezone.utc)
    stage = StageRecord(
        stage="extract_features", started_at=now, ended_at=now, result="succeeded"
    )
    job = TrainingJob(
        job_id="job-1",
        feature_group="default",
        model_name="gnn",
        seed=0,
        status="pending",
        stage_history=[stage],
    )
    assert job.failure_type is None
    with pytest.raises(ValidationError):
        TrainingJob(
            job_id="job-2",
            feature_group="default",
            model_name="gnn",
            seed=0,
            status="failed",
        )


def test_model_version_requires_utc_and_known_finite_metrics() -> None:
    record = ModelVersionRecord(
        model_name="gnn",
        version=1,
        created_at=datetime.now(timezone.utc),
        feature_group="default",
        metrics={"sinr_p50_db": 4.25},
        artifact_uri="models/gnn/1/model.pt",
    )
    assert record.version == 1
    with pytest.raises(ValidationError):
        ModelVersionRecord.model_validate(
            {**record.model_dump(), "metrics": {"unknown": 1.0}}
        )


def test_evidence_requires_predictions_and_complete_conflict_context() -> None:
    plan = ProbePlan(
        baseline_id="baseline", baseline=parameter_set(), target_cells=["gNB_5G"], variants=[]
    )
    evidence = EvidenceRecord(
        evidence_id="e-1",
        model_name="gnn",
        model_version=1,
        probe_plan=plan,
        predictions=[{"parameter_set_index": 0, "target_kpi": {"sinr_p50_db": 5.0}}],
        applied_thresholds=[
            ThresholdKpi(
                target_kpi="sinr_p50_db", lower_bound=-20.0, improve_direction="higher_is_better"
            )
        ],
        degradation_verdict="acceptable",
        marginal_effects=[
            MarginalEffectRecord(
                cell_id="gNB_5G",
                control_parameter="tx_power_dbm",
                target_kpi="sinr_p50_db",
                value_percent=1.0,
            )
        ],
        conflict_threshold=5.0,
        created_at=datetime.now(timezone.utc),
    )
    assert evidence.evidence_id == "e-1"
    with pytest.raises(ValidationError):
        EvidenceRecord.model_validate({**evidence.model_dump(), "predictions": []})


def test_common_error_response_is_strict_and_has_details_default() -> None:
    response = ErrorResponse(error_code="invalid_time_step", message="bad input")
    assert response.model_dump() == {
        "error_code": "invalid_time_step",
        "message": "bad input",
        "details": {},
    }
    with pytest.raises(ValidationError):
        ErrorResponse(error_code="x", message="x", unexpected=True)

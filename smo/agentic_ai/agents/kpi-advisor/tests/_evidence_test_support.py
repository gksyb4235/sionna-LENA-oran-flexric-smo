"""Shared complete evidence fixture and in-memory Mongo collection contract."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

TEST_DIR = Path(__file__).resolve().parent
AGENT_DIR = TEST_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(REPO_ROOT), str(AGENT_DIR), str(TEST_DIR)]

import test_agent_baseline_and_probe_plan as probe_cases  # noqa: E402
import test_agent_threshold_judgment_and_recommendations as judgment_cases  # noqa: E402
import test_temporal_scheduler_smoke as temporal_cases  # noqa: E402
from smo.aimlfw.common.models import MarginalEffectRecord  # noqa: E402

import agent  # noqa: E402
import conflict_analyzer  # noqa: E402
import evidence_store  # noqa: E402


def components(*, with_temporal: bool = True) -> dict[str, Any]:
    plan = probe_cases._simple_plan()
    baseline = judgment_cases._prediction()
    first_variant = plan.variants[0]
    variant_result = agent.ProbeVariantResult(
        variant=first_variant,
        prediction=judgment_cases._prediction(cell_goodput_mbps=110.0),
        percent_change={"cell_goodput_mbps": 10.0},
    )
    probe_result = agent.ProbeExecutionResult(
        plan=plan,
        model_name="gnn",
        model_version=1,
        baseline_prediction=baseline,
        variant_results=[variant_result],
    )
    thresholds = judgment_cases._default_kpi_threshold_config()
    judgment = agent.judge_degradation_verdict(baseline, thresholds)
    recommendations = judgment_cases._recommend(
        [variant_result],
        thresholds,
        judgment_cases._default_objective_primary_kpi_config(),
    )
    marginal_effects = [
        MarginalEffectRecord(
            cell_id="gNB_5G",
            control_parameter="tx_power_dbm",
            target_kpi="cell_goodput_mbps",
            value_percent=10.0,
        ),
        MarginalEffectRecord(
            cell_id="gNB_5G",
            control_parameter="ret_tilt_deg",
            target_kpi="cell_goodput_mbps",
            value_percent=-10.0,
        ),
    ]
    conflict = conflict_analyzer.analyze(marginal_effects, 5.0)
    temporal = None
    if with_temporal:
        temporal = temporal_cases._build(
            objectives=["throughput_maximization"],
            candidates_by_objective={"throughput_maximization": [temporal_cases._parameter_set()]},
            predict_batch=temporal_cases._make_predict_batch(lambda _cells, _step: temporal_cases._default_kpis()),
        )
    return {
        "probe_result": probe_result,
        "thresholds": thresholds,
        "judgment": judgment,
        "recommendation_result": recommendations,
        "marginal_effects": marginal_effects,
        "conflict_result": conflict,
        "temporal_plan": temporal,
    }


def record(*, with_temporal: bool = True):
    return evidence_store.assemble_evidence_record(**components(with_temporal=with_temporal))


class Collection:
    def __init__(self, *, fail_after_insert: bool = False) -> None:
        self.documents: dict[str, dict[str, Any]] = {}
        self.fail_after_insert = fail_after_insert

    def create_index(self, *_args: Any, **_kwargs: Any) -> str:
        return "evidence_id_unique"

    def insert_one(self, document: dict[str, Any]) -> object:
        self.documents[document["_id"]] = dict(document)
        if self.fail_after_insert:
            raise RuntimeError("injected acknowledgement failure")
        return object()

    def delete_one(self, query: dict[str, Any]) -> object:
        self.documents.pop(query["_id"], None)
        return object()

    def find_one(self, query: dict[str, Any], *_args: Any, **_kwargs: Any) -> dict[str, Any] | None:
        return self.documents.get(query["_id"])

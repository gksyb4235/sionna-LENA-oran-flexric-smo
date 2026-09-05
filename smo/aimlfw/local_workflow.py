"""CLI helpers that exercise the local stack through its public boundaries."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from smo.aimlfw.common.models import ParameterSet

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
KPI_ADVISOR_ROOT = REPOSITORY_ROOT / "smo" / "agentic_ai" / "agents" / "kpi-advisor"
if str(KPI_ADVISOR_ROOT) not in sys.path:
    sys.path.insert(0, str(KPI_ADVISOR_ROOT))

from policy_manager import (  # noqa: E402
    ApprovalRecord,
    PendingApproval,
    PolicyInstanceResult,
    publish_parameter_set,
    register_policy_type,
)


def _json_response(response: httpx.Response) -> dict[str, Any]:
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"{response.request.method} {response.request.url} returned a non-object")
    return payload


def bootstrap_model(
    result_dir: str,
    *,
    feature_group: str = "default",
    model_name: str = "gnn",
    seed: int = 0,
    timeout_seconds: float = 3600.0,
) -> dict[str, Any]:
    """Extract a real ns-3 result, train/register, then explicitly load version 1+."""
    deadline = time.monotonic() + timeout_seconds
    with httpx.Client(timeout=30.0) as client:
        extraction = _json_response(
            client.post(
                "http://127.0.0.1:8101/extract",
                json={"result_dir": result_dir, "feature_group": feature_group},
            )
        )
        created = _json_response(
            client.post(
                "http://127.0.0.1:8103/jobs",
                json={"feature_group": feature_group, "model_name": model_name, "seed": seed},
            )
        )
        job_id = str(created["job_id"])
        while True:
            job = _json_response(client.get(f"http://127.0.0.1:8103/jobs/{job_id}"))
            if job.get("status") == "completed":
                break
            if job.get("status") == "failed":
                raise RuntimeError(f"training failed: {job.get('failure_reason')}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"training job {job_id} exceeded {timeout_seconds:.0f}s")
            time.sleep(1.0)
        summary = job.get("training_summary")
        version = summary.get("model_version") if isinstance(summary, dict) else None
        if not isinstance(version, int):
            raise RuntimeError("completed training job has no registered model_version")
        inference = _json_response(
            client.put(
                "http://127.0.0.1:8105/serving-version",
                json={"model_name": model_name, "model_version": version},
                timeout=60.0,
            )
        )
        registry = _json_response(
            client.put(
                f"http://127.0.0.1:8104/models/{model_name}/serving-version",
                json={"version": version},
            )
        )
    return {
        "extraction": extraction,
        "training_job_id": job_id,
        "model_name": model_name,
        "model_version": version,
        "inference": inference,
        "registry": registry,
    }


def advise_and_publish(request_body: dict[str, Any], *, approved_by: str) -> dict[str, Any]:
    """Obtain persisted evidence, then publish the first recommendation through A1."""
    with httpx.Client(timeout=65.0) as client:
        advice = _json_response(client.post("http://127.0.0.1:8110/invoke", json=request_body))
    evidence_id = advice.get("evidence_record_id")
    if not isinstance(evidence_id, str) or not evidence_id:
        raise RuntimeError("KPI Advisor returned no persisted Evidence_Record identifier")
    recommendations = advice.get("recommendations")
    if not isinstance(recommendations, list) or not recommendations:
        return {
            "evidence_record_id": evidence_id,
            "degradation_verdict": advice.get("degradation_verdict"),
            "policy_status": "not_published",
            "reason": "no_recommended_parameter_set",
        }
    recommendation = recommendations[0]
    parameter_set = ParameterSet.model_validate(recommendation["parameter_set"])
    register_policy_type()
    result = publish_parameter_set(
        parameter_set,
        list(parameter_set.cells),
        ApprovalRecord(approver_id=approved_by, approved_at=datetime.now(UTC)),
        degradation_verdict=advice["degradation_verdict"],
    )
    if isinstance(result, PendingApproval):
        policy: dict[str, Any] = {"status": result.status}
    elif isinstance(result, PolicyInstanceResult):
        policy = {
            "policy_type_id": result.policy_type_id,
            "policy_instance_id": result.policy_instance_id,
            "status": result.status,
        }
    else:
        raise RuntimeError("Policy Manager returned an unsupported result")
    return {
        "evidence_record_id": evidence_id,
        "degradation_verdict": advice["degradation_verdict"],
        "policy": policy,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("request JSON must contain an object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    bootstrap = subparsers.add_parser("bootstrap-model")
    bootstrap.add_argument("result_dir")
    bootstrap.add_argument("--feature-group", default="default")
    bootstrap.add_argument("--model-name", default="gnn")
    bootstrap.add_argument("--seed", type=int, default=0)
    advise = subparsers.add_parser("advise-and-publish")
    advise.add_argument("request_json", type=Path)
    advise.add_argument("--approved-by", required=True)
    args = parser.parse_args()
    if args.command == "bootstrap-model":
        result = bootstrap_model(
            args.result_dir,
            feature_group=args.feature_group,
            model_name=args.model_name,
            seed=args.seed,
        )
    else:
        result = advise_and_publish(
            _read_json_object(args.request_json),
            approved_by=args.approved_by,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

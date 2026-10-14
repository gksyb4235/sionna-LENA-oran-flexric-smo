"""Deterministic operator shortlist; never relax the estimator's constraints."""

from copy import deepcopy
import hashlib
import json
import math
from .applications import application_for


RANKING = {
    "version": "operator-review-v2-app-role",
    "order": [
        "fully_evaluable_feasible_first",
        "fully_evaluable_before_unknown",
        "threshold_normalized_violation_sum",
        "violation_count",
        "goal_achievement_desc",
        "request_deviation",
        "candidate_id",
    ],
    "violation_formula": "sum(max(violation,0) / max(abs(effective_threshold),1))",
    "note": "Unweighted dimensionless comparison score, not a probability. Threshold scales are fixed across candidates. No constraint is removed or relaxed.",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def review_score(row):
    unknown, violations, score = 0, 0, 0.0
    for entry in row.get("constraints", []):
        if (
            entry.get("evaluation_status") == "insufficient_evidence"
            or not finite(entry.get("predicted_value"))
            or not finite(entry.get("effective_threshold"))
            or not finite(entry.get("violation"))
        ):
            unknown += 1
            continue
        violations += not entry["satisfied"]
        score += max(0.0, entry["violation"]) / max(abs(entry["effective_threshold"]), 1.0)
    unknown_goals = sum(
        e.get("evaluation_status") == "insufficient_evidence"
        or not finite(e.get("predicted_value"))
        for e in row.get("goals", [])
    )
    return {
        "normalized_violation_sum": score,
        "violation_count": violations,
        "unknown_constraints": unknown,
        "unknown_goals": unknown_goals,
        "fully_evaluable": not (unknown or unknown_goals),
    }


def shortlist(rows, limit=3, intent=None):
    unique = {}
    for row in rows:
        if row.get("status") == "invalid" or not row.get("parameters"):
            continue
        key = (tuple(sorted(row["parameters"].items())), row.get("delay_s"), row.get("horizon_s"))
        unique.setdefault(key, row)
    candidates = []
    for row in unique.values():
        item = deepcopy(row)
        item["operator_score"] = review_score(row)
        candidates.append(item)

    def order(row):
        score = row["operator_score"]
        tier = (0 if row["feasible"] else 1) if score["fully_evaluable"] else 2
        # Preserve goal-first ranking for feasible candidates.
        return (
            tier,
            score["normalized_violation_sum"] if tier else 0,
            score["violation_count"] if tier else 0,
            -row.get("goal_achievement", 0),
            row.get("request_deviation", 0),
            row["candidate_id"],
        )

    candidates.sort(key=order)
    selected = candidates[:limit]
    for rank, row in enumerate(selected, 1):
        families = {name.split("_")[0] for name in row["changes"]}
        row["operator_rank"] = rank
        row["suggested_ric"] = (
            "non_rt"
            if families <= {"txp", "ret"}
            else "near_rt"
            if families <= {"cio", "ttt", "hys"}
            else "split"
        )
        application = application_for(intent)
        if application is not None:
            row["application"] = application.describe()
            row["suggested_ric"] = application.target_ric
        row["requires_constraint_override"] = not row["feasible"]
        row["deployment_eligible"] = row["operator_score"]["fully_evaluable"]
        row["review_kind"] = (
            "insufficient_evidence"
            if not row["deployment_eligible"]
            else "feasible"
            if row["feasible"]
            else "least_violation"
        )
    result = {
        "candidates": selected,
        "ranking": deepcopy(RANKING),
        "evaluated_unique_count": len(candidates),
        "scope": "Evaluated candidates only; no claim of a global optimum or measured network improvement.",
    }
    result["review_sha256"] = digest(result)
    return result


def ensure_review(result, events):
    """Reconstruct old sessions from saved numerical evidence, with no new inference."""
    if (
        result is None
        or result.get("operator_review", {}).get("ranking", {}).get("version") == RANKING["version"]
    ):
        return result
    rows = []
    for event in events:
        if event.get("type") == "candidates":
            rows.extend((event.get("data") or {}).get("all_candidates", []))
    if not rows:
        for event in events:
            data = event.get("data") or {}
            if event.get("type") == "tool_result" and data.get("tool") == "estimate_batch":
                rows.extend((data.get("result") or {}).get("all_candidates", []))
    if not rows:
        rows = result.get("top_candidates", []) + result.get("infeasible_diagnostics", [])
        if result.get("original"):
            rows.append(result["original"])
    result["operator_review"] = shortlist(rows, intent=result.get("intent"))
    return result

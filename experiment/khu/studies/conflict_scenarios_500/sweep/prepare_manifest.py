"""Freeze a reviewed wide scenario CSV into a portable execution manifest."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sweep import PARAMETERS, SCHEMA, canonical_hash, sha256, validate_manifest


def prepare(source: Path, seed_start: int, experiment_id: str) -> dict:
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    jobs = []
    for index, row in enumerate(rows):
        initial = {p: float(row[p]) for p in PARAMETERS}
        state = initial.copy()
        events = []
        for second in (180, 360, 540, 720):
            changes = json.loads(row[f"t{second}_changes"])
            expected = {p: float(row[f"t{second}_{p}"]) for p in PARAMETERS}
            actual = {p: float(changes[p]) for p in PARAMETERS if p in changes}
            if set(changes) != set(actual):
                raise ValueError(f"Unknown changed parameter: {row['name']} at {second}")
            state.update(actual)
            if state != expected:
                raise ValueError(f"Changed fields and state columns disagree: {row['name']} at {second}")
            if actual:
                events.append({"time_s": second, "app": row[f"t{second}_app"], "updates": actual})
        seed = seed_start + index
        jobs.append({"job_id": row["name"], "trace_seed": seed,
                     "trace_path": f"scenarios/khu-real/ue_position/ue_positions_seed{seed}.csv",
                     "initial_app": row["initial_app"], "initial_parameters": initial,
                     "events": events, "notes_ko": row.get("비고", ""),
                     "schedule_kind": row["schedule_kind"],
                     "screening_tier": row.get("screening_tier", "unreviewed"),
                     "screening_priority": int(row.get("screening_priority", index + 1)),
                     "goal_supported_events": row.get("goal_supported_events"),
                     "tradeoff_supported_events": row.get("tradeoff_supported_events"),
                     "all_goals_supported": row.get("all_goals_supported")})
    result = {"schema": SCHEMA, "experiment_id": experiment_id,
              "duration_s": 900, "kpi_interval_s": 1, "n_ues": 300,
              "rng_seed": 1, "rng_run": 1,
              "source": {"name": source.name, "sha256": sha256(source)},
              "execution_mode": "fixed_absolute_parameters_no_DRL_no_RIC_no_LLM",
              "trace_assignment": {"rule": "CSV row index + seed_start", "seed_start": seed_start,
                                   "claimed_held_out": False,
                                   "note": "Trace identifiers may overlap older experiments. Split future datasets by whole trace and whole scenario, never by individual time windows."},
              "interpretation": "Screened conflict hypotheses, not measured conflicts or causal comparisons. Changes persist until overwritten; no automatic restore.",
              "jobs": jobs}
    validate_manifest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--experiment-id", default="agora_conflict_500_v1")
    args = parser.parse_args()
    if args.seed_start < 0:
        parser.error("--seed-start must be nonnegative")
    result = prepare(args.csv, args.seed_start, args.experiment_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "jobs": len(result["jobs"]),
                      "events": sum(len(j["events"]) for j in result["jobs"]),
                      "manifest_content_sha256": canonical_hash(result)}))


if __name__ == "__main__":
    main()

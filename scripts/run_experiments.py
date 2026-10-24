#!/usr/bin/env python3
"""Canonical entry point for KHU fixed and scheduled-parameter collection."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def command(kind: str, arguments: list[str]) -> list[str]:
    filename = {"fixed": "run_gnn_dataset_sweep.sh",
                "conflict": "run_conflict_dataset_sweep.sh"}[kind]
    return ["bash", str(ROOT / "experiment/khu/tools" / filename), *arguments]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Example: fixed 0 99 1 12 --dry-run; conflict 501 512 1 12 --dry-run. "
        "Trailing options are handled by the selected runner."))
    parser.add_argument("kind", choices=("fixed", "conflict"))
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int)
    parser.add_argument("cores_per_job", type=int)
    parser.add_argument("parallel_jobs", type=int)
    args, extra = parser.parse_known_args()
    if args.cores_per_job < 1 or args.parallel_jobs < 1 or args.end < args.start:
        parser.error("Invalid range or CPU/worker count")
    env = os.environ.copy()
    if args.kind == "fixed":
        # Standard collection is offline with 1-second KPI samples. Explicit
        # overrides remain possible and must be reported in the experiment.
        env.setdefault("GNN_SWEEP_KPI_REPORT_INTERVAL", "1")
        env.setdefault("GNN_SWEEP_INFLUX", "false")
        env.setdefault("GNN_SWEEP_LIVE_LOGS", "false")
    values = [str(args.start), str(args.end), str(args.cores_per_job), str(args.parallel_jobs)]
    result = subprocess.run(command(args.kind, values + extra), cwd=ROOT, env=env)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

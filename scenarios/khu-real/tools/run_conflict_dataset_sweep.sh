#!/usr/bin/env bash
set -euo pipefail

# Usage: run_conflict_dataset_sweep.sh FIRST LAST CPUS_PER_JOB PARALLEL_JOBS
# IDs are conflict_0001..conflict_0500, inclusive; they are not UE trace seeds.
# Example: ./scenarios/khu-real/tools/run_conflict_dataset_sweep.sh 1 100 1 12
# Add --dry-run to inspect, or --mode preflight to build/check without simulation.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
exec python3 "${repo_root}/scripts/conflict_sweep_range.py" "$@"

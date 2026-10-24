#!/usr/bin/env bash
set -euo pipefail

# Usage: run_conflict_dataset_sweep.sh FIRST LAST CPUS_PER_JOB PARALLEL_JOBS
# IDs are conflict_0001..conflict_0500 or reserve conflict_0501..conflict_0550.
# Each range selects its manifest automatically; do not span both datasets.
# Example: ./experiment/khu/tools/run_conflict_dataset_sweep.sh 501 512 1 12
# Add --dry-run to inspect, or --mode preflight to build/check without simulation.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd -P)"
exec python3 "${repo_root}/scripts/conflict_sweep_range.py" "$@"

#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"

export SWEEP_KIND=TTT
export SWEEP_VALUES="${TTT_VALUES:-128 160 256 320 480 512}"
export SWEEP_RESULTS_ROOT="${TTT_RESULTS_ROOT:-${repo_root}/scenarios/results/TTT_test}"
export SWEEP_NAME="${TTT_SWEEP_NAME:-sweep_$(date +%Y%m%d_%H%M%S)}"
export SWEEP_SIM_TIME="${TTT_SIM_TIME:-900}"
export SWEEP_MAX_PARALLEL="${TTT_MAX_PARALLEL:-3}"
export SWEEP_MEMORY_LIMIT_BYTES="${TTT_MEMORY_LIMIT_BYTES:-3221225472}"

exec "${script_dir}/run_cell_parameter_sweep.sh"

#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"

export SWEEP_KIND=TXP
export SWEEP_VALUES="${TXP_VALUES:-31 33 35 37 39 41 43}"
export SWEEP_RESULTS_ROOT="${TXP_RESULTS_ROOT:-${repo_root}/scenarios/results/TxP_test}"
export SWEEP_NAME="${TXP_SWEEP_NAME:-sweep_$(date +%Y%m%d_%H%M%S)}"
export SWEEP_SIM_TIME="${TXP_SIM_TIME:-900}"
export SWEEP_MAX_PARALLEL="${TXP_MAX_PARALLEL:-4}"
export SWEEP_MEMORY_LIMIT_BYTES="${TXP_MEMORY_LIMIT_BYTES:-3221225472}"

exec "${script_dir}/run_cell_parameter_sweep.sh"

#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"

export SWEEP_KIND=CIO
export SWEEP_VALUES="${CIO_VALUES:--6 -4 -2 0 2 4 6}"
export SWEEP_RESULTS_ROOT="${CIO_RESULTS_ROOT:-${repo_root}/scenarios/results/CIO_test}"
export SWEEP_NAME="${CIO_SWEEP_NAME:-sweep_$(date +%Y%m%d_%H%M%S)}"
export SWEEP_SIM_TIME="${CIO_SIM_TIME:-900}"
export SWEEP_MAX_PARALLEL="${CIO_MAX_PARALLEL:-3}"
export SWEEP_MEMORY_LIMIT_BYTES="${CIO_MEMORY_LIMIT_BYTES:-3221225472}"

exec "${script_dir}/run_cell_parameter_sweep.sh"

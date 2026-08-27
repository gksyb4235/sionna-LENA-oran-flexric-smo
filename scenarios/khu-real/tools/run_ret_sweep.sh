#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"

export SWEEP_KIND=RET
export SWEEP_VALUES="${RET_VALUES:-1 3 5 7 9 12}"
export SWEEP_RESULTS_ROOT="${RET_RESULTS_ROOT:-${repo_root}/scenarios/results/RET_test}"
export SWEEP_NAME="${RET_SWEEP_NAME:-sweep_$(date +%Y%m%d_%H%M%S)}"
export SWEEP_SIM_TIME="${RET_SIM_TIME:-900}"
export SWEEP_MAX_PARALLEL="${RET_MAX_PARALLEL:-3}"
export SWEEP_MEMORY_LIMIT_BYTES="${RET_MEMORY_LIMIT_BYTES:-3221225472}"

# Defaults to:
# scenarios/khu-real/ret_caches/tilt_<value>deg/
#   sionna_rt_cache_1p8ghz_2x2_1x1_lzf.h5
#   sionna_rt_cache_3p5ghz_2x2_1x1_lzf.h5
# Override RET_CACHE_ROOT or the RET_CACHE18_TEMPLATE/RET_CACHE35_TEMPLATE
# variables if the generated caches use a different layout. Templates support
# {value} (raw value) and {label} (filename-safe value, e.g. 2.5 -> 2p5).

exec "${script_dir}/run_cell_parameter_sweep.sh"

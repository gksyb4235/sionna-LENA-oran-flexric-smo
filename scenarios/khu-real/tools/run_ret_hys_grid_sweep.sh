#!/usr/bin/env bash
set -euo pipefail

# 2D grid sweep: every (RET tilt, HYS) combination, each cell held fixed while
# the other two loop. Baseline (TTT/TxP/CIO) matches the other single-axis
# sweep runners. Requires scenarios/khu-real/ret_caches/tilt_<value>deg/ for
# every RET_VALUES entry.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
binary="${repo_root}/build/scratch/ns3.48-khu-real-dual-nr-sionna-final-default"

results_root="${GRID_RESULTS_ROOT:-${repo_root}/scenarios/results/RET_HYS_grid_test}"
sweep_name="${GRID_SWEEP_NAME:-sweep_$(date +%Y%m%d_%H%M%S)}"
sweep_root="${results_root}/${sweep_name}"
sim_time="${GRID_SIM_TIME:-900}"
max_parallel="${GRID_MAX_PARALLEL:-6}"
memory_limit_bytes="${GRID_MEMORY_LIMIT_BYTES:-3221225472}"
ret_values_text="${RET_VALUES:-7 9 12}"
hys_values_text="${HYS_VALUES:-0.5 1 1.5 2 2.5 3}"
ret_cache_root="${RET_CACHE_ROOT:-${repo_root}/scenarios/khu-real/ret_caches}"

# One logical CPU per physical core (this host: 8 physical cores as thread 0
# of pairs (0,1),(2,3),...,(14,15); keeping the sibling unused avoids SMT
# contention). max_parallel must not exceed the number of entries here.
physical_core_cpus=(0 2 4 6 8 10 12 14)

read -r -a ret_values <<< "${ret_values_text}"
read -r -a hys_values <<< "${hys_values_text}"

safe_label()
{
  local label="${1//-/m}"
  printf '%s' "${label//./p}"
}

if [[ ! -x "${binary}" ]]; then
  printf 'Missing executable: %s\n' "${binary}" >&2
  exit 1
fi
if ((max_parallel < 1 || max_parallel > ${#physical_core_cpus[@]})); then
  printf 'GRID_MAX_PARALLEL must be between 1 and %d, got %s\n' \
    "${#physical_core_cpus[@]}" "${max_parallel}" >&2
  exit 1
fi
if ((${#ret_values[@]} == 0 || ${#hys_values[@]} == 0)); then
  printf 'RET_VALUES and HYS_VALUES must each contain at least one value\n' >&2
  exit 1
fi

# Build the full (ret, hys) job list and check every RET cache up front, so a
# missing cache aborts before any simulation starts.
job_ret=()
job_hys=()
missing_cache=0
for ret_value in "${ret_values[@]}"; do
  ret_label="$(safe_label "${ret_value}")"
  cache18="${ret_cache_root}/tilt_${ret_label}deg/sionna_rt_cache_1p8ghz_2x2_1x1_lzf.h5"
  cache35="${ret_cache_root}/tilt_${ret_label}deg/sionna_rt_cache_3p5ghz_2x2_1x1_lzf.h5"
  for cache in "${cache18}" "${cache35}"; do
    if [[ ! -f "${cache}" ]]; then
      printf 'Missing RET cache: %s\n' "${cache}" >&2
      missing_cache=1
    fi
  done
  for hys_value in "${hys_values[@]}"; do
    job_ret+=("${ret_value}")
    job_hys+=("${hys_value}")
  done
done
if ((missing_cache != 0)); then
  printf 'Create all listed RET caches first; no simulation was started.\n' >&2
  exit 1
fi

mkdir -p "${sweep_root}"

config_file="${sweep_root}/sweep_config.txt"
jobs_file="${sweep_root}/jobs.csv"
completion_file="${sweep_root}/completion.csv"

{
  printf 'created_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'repo_root=%s\n' "${repo_root}"
  printf 'binary=%s\n' "${binary}"
  printf 'sim_time_s=%s\n' "${sim_time}"
  printf 'max_parallel=%s\n' "${max_parallel}"
  printf 'cpu_ids='
  printf '%s ' "${physical_core_cpus[@]:0:max_parallel}"
  printf '\n'
  printf 'memory_address_space_limit_bytes=%s\n' "${memory_limit_bytes}"
  printf 'ret_values_deg=%s\n' "${ret_values_text}"
  printf 'hys_values_db=%s\n' "${hys_values_text}"
  printf 'total_jobs=%d\n' "${#job_ret[@]}"
  printf 'bandwidth18_hz=10000000\n'
  printf 'bandwidth35_hz=10000000\n'
  printf 'ttt_ms=256\n'
  printf 'baseline_cio_db=gNB_5G:0,gNB_4G_1:0,gNB_4G_2:0\n'
  printf 'baseline_tx_power_dbm=43\n'
  printf 'ret_bearing_deg=gNB_5G:20,gNB_4G_1:0,gNB_4G_2:10\n'
} > "${config_file}"

printf 'ret_tilt_deg,hys_db,run_tag,pid,cpu_id,launcher_log\n' > "${jobs_file}"
printf 'ret_tilt_deg,hys_db,run_tag,pid,cpu_id,exit_code,finished_at\n' > "${completion_file}"

active_pids=()
sweep_failed=0
terminate_active()
{
  if ((${#active_pids[@]} > 0)); then
    kill -TERM "${active_pids[@]}" 2>/dev/null || true
    wait "${active_pids[@]}" 2>/dev/null || true
  fi
}
trap terminate_active INT TERM

printf '[RET x HYS grid] output=%s jobs=%d parallel=%d simTime=%ss memoryLimit=%s bytes\n' \
  "${sweep_root}" "${#job_ret[@]}" "${max_parallel}" "${sim_time}" "${memory_limit_bytes}"

for ((wave_start = 0; wave_start < ${#job_ret[@]}; wave_start += max_parallel)); do
  active_pids=()
  active_ret=()
  active_hys=()
  active_tags=()
  active_cpus=()

  for ((slot = 0; slot < max_parallel; ++slot)); do
    index=$((wave_start + slot))
    if ((index >= ${#job_ret[@]})); then
      break
    fi

    ret_value="${job_ret[index]}"
    hys_value="${job_hys[index]}"
    ret_label="$(safe_label "${ret_value}")"
    hys_label="$(safe_label "${hys_value}")"
    cpu_id="${physical_core_cpus[slot]}"
    cache18="${ret_cache_root}/tilt_${ret_label}deg/sionna_rt_cache_1p8ghz_2x2_1x1_lzf.h5"
    cache35="${ret_cache_root}/tilt_${ret_label}deg/sionna_rt_cache_3p5ghz_2x2_1x1_lzf.h5"
    run_tag="RET_tilt_${ret_label}deg_HYS_${hys_label}dB_all3_seed12_bw10M_ttt256ms"
    launcher_log="${sweep_root}/${run_tag}.launcher.log"

    printf '[launch] RET=%sdeg HYS=%sdB cpu=%s tag=%s\n' \
      "${ret_value}" "${hys_value}" "${cpu_id}" "${run_tag}"
    (
      cd "${repo_root}"
      exec env \
        OMP_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 \
        NUMEXPR_NUM_THREADS=1 \
        taskset -c "${cpu_id}" \
        prlimit --as="${memory_limit_bytes}" \
        "${binary}" \
          --centralFrequency18=1.8e9 \
          --bandwidth18=10e6 \
          --centralFrequency35=3.5e9 \
          --bandwidth35=10e6 \
          --numerology=0 \
          --sionnaCacheFile18="${cache18}" \
          --sionnaCacheFile35="${cache35}" \
          --sionnaUpdatePeriod=1s \
          --udpBaseIntervalMs=200 \
          --trafficMultiplierEnabled=true \
          --simTime="${sim_time}" \
          --kpiReportInterval=1.0 \
          --sumoTrace=scenarios/khu-real/ue_positions_seed12.csv \
          --N_Ues=300 \
          --resultsRoot="${sweep_root}" \
          --runTag="${run_tag}" \
          --handoverTtt=256 \
          --handoverHysteresis="${hys_value}" \
          --cellCioDb=gNB_5G:0,gNB_4G_1:0,gNB_4G_2:0 \
          --cellTxPowerDbm=gNB_5G:43,gNB_4G_1:43,gNB_4G_2:43 \
          --cellHysteresisDb="gNB_5G:${hys_value},gNB_4G_1:${hys_value},gNB_4G_2:${hys_value}" \
          --cellTttMs=gNB_5G:256,gNB_4G_1:256,gNB_4G_2:256 \
          --cellRetBearingDeg=gNB_5G:20,gNB_4G_1:0,gNB_4G_2:10 \
          --cellRetTiltDeg="gNB_5G:${ret_value},gNB_4G_1:${ret_value},gNB_4G_2:${ret_value}"
    ) > "${launcher_log}" 2>&1 &

    pid=$!
    active_pids+=("${pid}")
    active_ret+=("${ret_value}")
    active_hys+=("${hys_value}")
    active_tags+=("${run_tag}")
    active_cpus+=("${cpu_id}")
    printf '%s,%s,%s,%s,%s,%s\n' \
      "${ret_value}" "${hys_value}" "${run_tag}" "${pid}" "${cpu_id}" "${launcher_log}" \
      >> "${jobs_file}"
  done

  for ((slot = 0; slot < ${#active_pids[@]}; ++slot)); do
    pid="${active_pids[slot]}"
    if wait "${pid}"; then
      exit_code=0
    else
      exit_code=$?
      sweep_failed=1
    fi
    printf '[complete] RET=%sdeg HYS=%sdB pid=%s exit=%s\n' \
      "${active_ret[slot]}" "${active_hys[slot]}" "${pid}" "${exit_code}"
    printf '%s,%s,%s,%s,%s,%s,%s\n' \
      "${active_ret[slot]}" "${active_hys[slot]}" "${active_tags[slot]}" "${pid}" \
      "${active_cpus[slot]}" "${exit_code}" "$(date --iso-8601=seconds)" >> "${completion_file}"
  done
done

active_pids=()
trap - INT TERM
if ((sweep_failed != 0)); then
  printf '[RET x HYS grid] one or more runs failed: %s\n' "${sweep_root}" >&2
  exit 1
fi
printf '[RET x HYS grid] all runs completed: %s\n' "${sweep_root}"

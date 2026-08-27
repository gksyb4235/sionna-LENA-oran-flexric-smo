#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
binary="${repo_root}/build/scratch/ns3.48-khu-real-dual-nr-sionna-final-default"

results_root="${HYS_RESULTS_ROOT:-${repo_root}/scenarios/results/HYS_test}"
sweep_name="${HYS_SWEEP_NAME:-sweep_$(date +%Y%m%d_%H%M%S)}"
sweep_root="${results_root}/${sweep_name}"
sim_time="${HYS_SIM_TIME:-900}"
max_parallel="${HYS_MAX_PARALLEL:-4}"
memory_limit_bytes="${HYS_MEMORY_LIMIT_BYTES:-3221225472}"
hys_values_text="${HYS_VALUES:-0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0}"

# One logical CPU from each physical core. On this host the sibling pairs are
# (0,1), (2,3), ..., (14,15). Keeping the sibling unused avoids SMT contention.
physical_core_cpus=(0 2 4 6 8 10 12 14)
read -r -a hys_values <<< "${hys_values_text}"

if [[ ! -x "${binary}" ]]; then
  printf 'Missing executable: %s\n' "${binary}" >&2
  exit 1
fi
if ((max_parallel < 1 || max_parallel > 8)); then
  printf 'HYS_MAX_PARALLEL must be between 1 and 8, got %s\n' "${max_parallel}" >&2
  exit 1
fi
if ((${#hys_values[@]} == 0)); then
  printf 'HYS_VALUES must contain at least one value\n' >&2
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
  printf 'hys_values_db=%s\n' "${hys_values_text}"
  printf 'bandwidth18_hz=10000000\n'
  printf 'bandwidth35_hz=10000000\n'
  printf 'ttt_ms=256\n'
  printf 'cio_db=gNB_5G:0,gNB_4G_1:0,gNB_4G_2:0\n'
  printf 'tx_power_dbm=gNB_5G:43,gNB_4G_1:43,gNB_4G_2:43\n'
  printf 'ret_bearing_deg=gNB_5G:20,gNB_4G_1:0,gNB_4G_2:10\n'
  printf 'ret_tilt_deg=gNB_5G:15,gNB_4G_1:15,gNB_4G_2:15\n'
} > "${config_file}"

printf 'hys_db,run_tag,pid,cpu_id,launcher_log\n' > "${jobs_file}"
printf 'hys_db,run_tag,pid,cpu_id,exit_code,finished_at\n' > "${completion_file}"

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

printf '[HYS sweep] output=%s runs=%d parallel=%d simTime=%ss memoryLimit=%s bytes\n' \
  "${sweep_root}" "${#hys_values[@]}" "${max_parallel}" "${sim_time}" \
  "${memory_limit_bytes}"

for ((wave_start = 0; wave_start < ${#hys_values[@]}; wave_start += max_parallel)); do
  active_pids=()
  active_hys=()
  active_tags=()
  active_cpus=()

  for ((slot = 0; slot < max_parallel; ++slot)); do
    index=$((wave_start + slot))
    if ((index >= ${#hys_values[@]})); then
      break
    fi

    hys_value="${hys_values[index]}"
    hys_label="${hys_value//./p}"
    cpu_id="${physical_core_cpus[slot]}"
    run_tag="HYS_${hys_label}dB_seed12_bw10M_ttt256ms"
    launcher_log="${sweep_root}/${run_tag}.launcher.log"

    printf '[launch] HYS=%s dB cpu=%s tag=%s\n' "${hys_value}" "${cpu_id}" "${run_tag}"
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
          --sionnaCacheFile18=scenarios/khu-real/sionna_rt_cache_1p8ghz_2x2_1x1_lzf.h5 \
          --sionnaCacheFile35=scenarios/khu-real/sionna_rt_cache_3p5ghz_2x2_1x1_lzf.h5 \
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
          --cellRetTiltDeg=gNB_5G:15,gNB_4G_1:15,gNB_4G_2:15
    ) > "${launcher_log}" 2>&1 &

    pid=$!
    active_pids+=("${pid}")
    active_hys+=("${hys_value}")
    active_tags+=("${run_tag}")
    active_cpus+=("${cpu_id}")
    printf '%s,%s,%s,%s,%s\n' \
      "${hys_value}" "${run_tag}" "${pid}" "${cpu_id}" "${launcher_log}" >> "${jobs_file}"
  done

  for ((slot = 0; slot < ${#active_pids[@]}; ++slot)); do
    pid="${active_pids[slot]}"
    if wait "${pid}"; then
      exit_code=0
    else
      exit_code=$?
      sweep_failed=1
    fi
    printf '[complete] HYS=%s dB pid=%s exit=%s\n' \
      "${active_hys[slot]}" "${pid}" "${exit_code}"
    printf '%s,%s,%s,%s,%s,%s\n' \
      "${active_hys[slot]}" "${active_tags[slot]}" "${pid}" \
      "${active_cpus[slot]}" "${exit_code}" "$(date --iso-8601=seconds)" >> "${completion_file}"
  done
done

active_pids=()
trap - INT TERM
if ((sweep_failed != 0)); then
  printf '[HYS sweep] one or more runs failed: %s\n' "${sweep_root}" >&2
  exit 1
fi
printf '[HYS sweep] all runs completed: %s\n' "${sweep_root}"

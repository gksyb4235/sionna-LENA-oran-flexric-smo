#!/usr/bin/env bash
set -euo pipefail

# Shared runner used by run_txp_sweep.sh, run_cio_sweep.sh, and
# run_ret_sweep.sh. Invoke one of those wrappers instead of this file.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
binary="${repo_root}/build/scratch/ns3.48-khu-real-dual-nr-sionna-final-default"

sweep_kind="${SWEEP_KIND:?SWEEP_KIND is required}"
sweep_values_text="${SWEEP_VALUES:?SWEEP_VALUES is required}"
results_root="${SWEEP_RESULTS_ROOT:?SWEEP_RESULTS_ROOT is required}"
sweep_name="${SWEEP_NAME:-sweep_$(date +%Y%m%d_%H%M%S)}"
sweep_root="${results_root}/${sweep_name}"
sim_time="${SWEEP_SIM_TIME:-900}"
max_parallel="${SWEEP_MAX_PARALLEL:-4}"
memory_limit_bytes="${SWEEP_MEMORY_LIMIT_BYTES:-3221225472}"

cache18_default="${repo_root}/scenarios/khu-real/sionna_rt_cache_1p8ghz_2x2_1x1_lzf.h5"
cache35_default="${repo_root}/scenarios/khu-real/sionna_rt_cache_3p5ghz_2x2_1x1_lzf.h5"
ret_cache_root="${RET_CACHE_ROOT:-${repo_root}/scenarios/khu-real/ret_caches}"
if [[ -n "${RET_CACHE18_TEMPLATE:-}" ]]; then
  ret_cache18_template="${RET_CACHE18_TEMPLATE}"
else
  ret_cache18_template="${ret_cache_root}/tilt_{label}deg/sionna_rt_cache_1p8ghz_2x2_1x1_lzf.h5"
fi
if [[ -n "${RET_CACHE35_TEMPLATE:-}" ]]; then
  ret_cache35_template="${RET_CACHE35_TEMPLATE}"
else
  ret_cache35_template="${ret_cache_root}/tilt_{label}deg/sionna_rt_cache_3p5ghz_2x2_1x1_lzf.h5"
fi

# One logical CPU from each physical core on this 8-core/16-thread host.
physical_core_cpus=(0 2 4 6 8 10 12 14)
read -r -a sweep_values <<< "${sweep_values_text}"

safe_label()
{
  local label="${1//-/m}"
  printf '%s' "${label//./p}"
}

expand_cache_template()
{
  local template="$1"
  local value="$2"
  local label="$3"
  template="${template//\{value\}/${value}}"
  printf '%s' "${template//\{label\}/${label}}"
}

if [[ ! -x "${binary}" ]]; then
  printf 'Missing executable: %s\n' "${binary}" >&2
  exit 1
fi
if ((max_parallel < 1 || max_parallel > 8)); then
  printf 'SWEEP_MAX_PARALLEL must be between 1 and 8, got %s\n' "${max_parallel}" >&2
  exit 1
fi
if ((${#sweep_values[@]} == 0)); then
  printf 'SWEEP_VALUES must contain at least one value\n' >&2
  exit 1
fi
if [[ "${sweep_kind}" != "TXP" && "${sweep_kind}" != "CIO" && "${sweep_kind}" != "RET" ]]; then
  printf 'Unsupported SWEEP_KIND: %s\n' "${sweep_kind}" >&2
  exit 1
fi

for value in "${sweep_values[@]}"; do
  if [[ ! "${value}" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
    printf '%s value is not numeric: %s\n' "${sweep_kind}" "${value}" >&2
    exit 1
  fi
done

if [[ "${sweep_kind}" == "CIO" ]]; then
  for value in "${sweep_values[@]}"; do
    if ! awk -v value="${value}" 'BEGIN { exit !(value >= -15 && value <= 15) }'; then
      printf 'CIO value must be in [-15,15] dB: %s\n' "${value}" >&2
      exit 1
    fi
  done
fi

# RET changes the actual ray-traced channel, so every tilt requires matching
# low- and mid-band caches. Check all files before launching any simulation.
if [[ "${sweep_kind}" == "RET" ]]; then
  missing_cache=0
  for value in "${sweep_values[@]}"; do
    label="$(safe_label "${value}")"
    cache18="$(expand_cache_template "${ret_cache18_template}" "${value}" "${label}")"
    cache35="$(expand_cache_template "${ret_cache35_template}" "${value}" "${label}")"
    for cache in "${cache18}" "${cache35}"; do
      if [[ ! -f "${cache}" ]]; then
        printf 'Missing RET cache: %s\n' "${cache}" >&2
        missing_cache=1
      fi
    done
  done
  if ((missing_cache != 0)); then
    printf 'Create all listed RET caches first; no simulation was started.\n' >&2
    exit 1
  fi
else
  for cache in "${cache18_default}" "${cache35_default}"; do
    if [[ ! -f "${cache}" ]]; then
      printf 'Missing Sionna cache: %s\n' "${cache}" >&2
      exit 1
    fi
  done
fi

mkdir -p "${sweep_root}"

config_file="${sweep_root}/sweep_config.txt"
jobs_file="${sweep_root}/jobs.csv"
completion_file="${sweep_root}/completion.csv"

{
  printf 'created_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'sweep_kind=%s\n' "${sweep_kind}"
  printf 'sweep_values=%s\n' "${sweep_values_text}"
  printf 'repo_root=%s\n' "${repo_root}"
  printf 'binary=%s\n' "${binary}"
  printf 'sim_time_s=%s\n' "${sim_time}"
  printf 'max_parallel=%s\n' "${max_parallel}"
  printf 'cpu_ids='
  printf '%s ' "${physical_core_cpus[@]:0:max_parallel}"
  printf '\n'
  printf 'memory_address_space_limit_bytes=%s\n' "${memory_limit_bytes}"
  printf 'bandwidth18_hz=10000000\n'
  printf 'bandwidth35_hz=10000000\n'
  printf 'ttt_ms=256\n'
  printf 'baseline_hys_db=2.5\n'
  printf 'baseline_tx_power_dbm=43\n'
  printf 'baseline_cio_db=gNB_5G:0,gNB_4G_1:0,gNB_4G_2:0\n'
  printf 'ret_bearing_deg=gNB_5G:20,gNB_4G_1:0,gNB_4G_2:10\n'
  printf 'baseline_ret_tilt_deg=15\n'
  if [[ "${sweep_kind}" == "CIO" ]]; then
    printf 'interpretation_warning=Equal CIO on serving and neighbor cells cancels in the A3 Ocn-Ocp comparison.\n'
  fi
  if [[ "${sweep_kind}" == "RET" ]]; then
    printf 'ret_cache18_template=%s\n' "${ret_cache18_template}"
    printf 'ret_cache35_template=%s\n' "${ret_cache35_template}"
  fi
} > "${config_file}"

printf 'parameter_value,run_tag,pid,cpu_id,launcher_log\n' > "${jobs_file}"
printf 'parameter_value,run_tag,pid,cpu_id,exit_code,finished_at\n' > "${completion_file}"

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

printf '[%s sweep] output=%s runs=%d parallel=%d simTime=%ss memoryLimit=%s bytes\n' \
  "${sweep_kind}" "${sweep_root}" "${#sweep_values[@]}" "${max_parallel}" \
  "${sim_time}" "${memory_limit_bytes}"

for ((wave_start = 0; wave_start < ${#sweep_values[@]}; wave_start += max_parallel)); do
  active_pids=()
  active_values=()
  active_tags=()
  active_cpus=()

  for ((slot = 0; slot < max_parallel; ++slot)); do
    index=$((wave_start + slot))
    if ((index >= ${#sweep_values[@]})); then
      break
    fi

    value="${sweep_values[index]}"
    label="$(safe_label "${value}")"
    cpu_id="${physical_core_cpus[slot]}"

    txp_map='gNB_5G:43,gNB_4G_1:43,gNB_4G_2:43'
    cio_map='gNB_5G:0,gNB_4G_1:0,gNB_4G_2:0'
    tilt_map='gNB_5G:15,gNB_4G_1:15,gNB_4G_2:15'
    cache18="${cache18_default}"
    cache35="${cache35_default}"

    case "${sweep_kind}" in
      TXP)
        txp_map="gNB_5G:${value},gNB_4G_1:${value},gNB_4G_2:${value}"
        run_tag="TxP_${label}dBm_all3_seed12_bw10M_ttt256ms_hys2p5dB"
        ;;
      CIO)
        cio_map="gNB_5G:${value},gNB_4G_1:${value},gNB_4G_2:${value}"
        run_tag="CIO_${label}dB_all3_seed12_bw10M_ttt256ms_hys2p5dB"
        ;;
      RET)
        tilt_map="gNB_5G:${value},gNB_4G_1:${value},gNB_4G_2:${value}"
        cache18="$(expand_cache_template "${ret_cache18_template}" "${value}" "${label}")"
        cache35="$(expand_cache_template "${ret_cache35_template}" "${value}" "${label}")"
        run_tag="RET_tilt_${label}deg_all3_seed12_bw10M_ttt256ms_hys2p5dB"
        ;;
    esac

    launcher_log="${sweep_root}/${run_tag}.launcher.log"
    printf '[launch] %s=%s cpu=%s tag=%s\n' "${sweep_kind}" "${value}" "${cpu_id}" "${run_tag}"
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
          --handoverHysteresis=2.5 \
          --cellCioDb="${cio_map}" \
          --cellTxPowerDbm="${txp_map}" \
          --cellHysteresisDb=gNB_5G:2.5,gNB_4G_1:2.5,gNB_4G_2:2.5 \
          --cellTttMs=gNB_5G:256,gNB_4G_1:256,gNB_4G_2:256 \
          --cellRetBearingDeg=gNB_5G:20,gNB_4G_1:0,gNB_4G_2:10 \
          --cellRetTiltDeg="${tilt_map}"
    ) > "${launcher_log}" 2>&1 &

    pid=$!
    active_pids+=("${pid}")
    active_values+=("${value}")
    active_tags+=("${run_tag}")
    active_cpus+=("${cpu_id}")
    printf '%s,%s,%s,%s,%s\n' \
      "${value}" "${run_tag}" "${pid}" "${cpu_id}" "${launcher_log}" >> "${jobs_file}"
  done

  for ((slot = 0; slot < ${#active_pids[@]}; ++slot)); do
    pid="${active_pids[slot]}"
    if wait "${pid}"; then
      exit_code=0
    else
      exit_code=$?
      sweep_failed=1
    fi
    printf '[complete] %s=%s pid=%s exit=%s\n' \
      "${sweep_kind}" "${active_values[slot]}" "${pid}" "${exit_code}"
    printf '%s,%s,%s,%s,%s,%s\n' \
      "${active_values[slot]}" "${active_tags[slot]}" "${pid}" \
      "${active_cpus[slot]}" "${exit_code}" "$(date --iso-8601=seconds)" >> "${completion_file}"
  done
done

active_pids=()
trap - INT TERM
if ((sweep_failed != 0)); then
  printf '[%s sweep] one or more runs failed: %s\n' "${sweep_kind}" "${sweep_root}" >&2
  exit 1
fi
printf '[%s sweep] all runs completed: %s\n' "${sweep_kind}" "${sweep_root}"

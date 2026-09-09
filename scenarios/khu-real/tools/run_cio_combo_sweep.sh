#!/usr/bin/env bash
set -euo pipefail

# Per-cell CIO combination sweep, holding RET/HYS/TTT/TxP fixed. The 5 candidate
# CIO values (0.5/1/1.5/2/2.5 dB) applied to 3 distinguishable cells give 5^3=125
# ordered combinations, but the A3 handover decision only depends on the
# *relative* CIO difference between cells (Ocn - Ocp), so any combination that
# is a uniform shift of another is physically equivalent. Collapsing those
# leaves 61 distinct combinations, generated below.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
binary="${repo_root}/build/scratch/ns3.48-khu-real-dual-nr-sionna-final-default"

results_root="${CIO_COMBO_RESULTS_ROOT:-${repo_root}/scenarios/results/CIO_combo_test}"
sweep_name="${CIO_COMBO_SWEEP_NAME:-sweep_$(date +%Y%m%d_%H%M%S)}"
sweep_root="${results_root}/${sweep_name}"
sim_time="${CIO_COMBO_SIM_TIME:-900}"
max_parallel="${CIO_COMBO_MAX_PARALLEL:-8}"
memory_limit_bytes="${CIO_COMBO_MEMORY_LIMIT_BYTES:-3221225472}"
ret_tilt="${CIO_COMBO_RET_TILT:-9}"
hys_db="${CIO_COMBO_HYS:-5.0}"
tx_power="${CIO_COMBO_TXP:-43}"
ret_cache_root="${RET_CACHE_ROOT:-${repo_root}/scenarios/khu-real/ret_caches}"

# One logical CPU per physical core (8 physical cores on this host: 0,2,4,...,14).
physical_core_cpus=(0 2 4 6 8 10 12 14)

if [[ ! -x "${binary}" ]]; then
  printf 'Missing executable: %s\n' "${binary}" >&2
  exit 1
fi
if ((max_parallel < 1 || max_parallel > ${#physical_core_cpus[@]})); then
  printf 'CIO_COMBO_MAX_PARALLEL must be between 1 and %d, got %s\n' \
    "${#physical_core_cpus[@]}" "${max_parallel}" >&2
  exit 1
fi

ret_label="${ret_tilt//./p}"
cache18="${ret_cache_root}/tilt_${ret_label}deg/sionna_rt_cache_1p8ghz_2x2_1x1_lzf.h5"
cache35="${ret_cache_root}/tilt_${ret_label}deg/sionna_rt_cache_3p5ghz_2x2_1x1_lzf.h5"
for cache in "${cache18}" "${cache35}"; do
  if [[ ! -f "${cache}" ]]; then
    printf 'Missing RET cache: %s\n' "${cache}" >&2
    exit 1
  fi
done

# Generate the 61 canonical (5G, 4G_1, 4G_2) index combinations (0..4, mapped
# to 0.5/1/1.5/2/2.5 dB), i.e. every combo whose minimum element is index 0.
combo_list="$(python3 - <<'PY'
import itertools
values = [0.5, 1.0, 1.5, 2.0, 2.5]
n = len(values)
seen = set()
for combo in itertools.product(range(n), repeat=3):
    if min(combo) != 0:
        continue
    print(" ".join(f"{values[i]}" for i in combo))
PY
)"

job_cio5g=()
job_cio4g1=()
job_cio4g2=()
while read -r c5g c4g1 c4g2; do
  job_cio5g+=("${c5g}")
  job_cio4g1+=("${c4g1}")
  job_cio4g2+=("${c4g2}")
done <<< "${combo_list}"

total_jobs="${#job_cio5g[@]}"

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
  printf 'total_jobs=%d\n' "${total_jobs}"
  printf 'baseline_ret_tilt_deg=%s\n' "${ret_tilt}"
  printf 'baseline_hys_db=%s\n' "${hys_db}"
  printf 'baseline_tx_power_dbm=%s\n' "${tx_power}"
  printf 'ttt_ms=256\n'
  printf 'seed=12\n'
} > "${config_file}"

printf 'cio_5g,cio_4g1,cio_4g2,run_tag,pid,cpu_id,exit_code,finished_at\n' > "${completion_file}"
printf 'cio_5g,cio_4g1,cio_4g2,run_tag,pid,cpu_id\n' > "${jobs_file}"

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

printf '[CIO combo sweep] output=%s jobs=%d parallel=%d simTime=%ss RET=%sdeg HYS=%sdB TxP=%sdBm\n' \
  "${sweep_root}" "${total_jobs}" "${max_parallel}" "${sim_time}" "${ret_tilt}" "${hys_db}" "${tx_power}"

for ((wave_start = 0; wave_start < total_jobs; wave_start += max_parallel)); do
  active_pids=()
  active_tags=()
  active_cios=()
  active_cpus=()

  for ((slot = 0; slot < max_parallel; ++slot)); do
    index=$((wave_start + slot))
    if ((index >= total_jobs)); then
      break
    fi

    c5g="${job_cio5g[index]}"
    c4g1="${job_cio4g1[index]}"
    c4g2="${job_cio4g2[index]}"
    l5g="${c5g//./p}"; l4g1="${c4g1//./p}"; l4g2="${c4g2//./p}"
    cpu_id="${physical_core_cpus[slot]}"
    run_tag="CIO_5g${l5g}_4g1${l4g1}_4g2${l4g2}_RET${ret_tilt}deg_HYS${hys_db//./p}dB_seed12"
    launcher_log="${sweep_root}/${run_tag}.launcher.log"

    printf '[launch] CIO=(%s,%s,%s) cpu=%s tag=%s\n' "${c5g}" "${c4g1}" "${c4g2}" "${cpu_id}" "${run_tag}"
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
          --handoverHysteresis="${hys_db}" \
          --cellCioDb="gNB_5G:${c5g},gNB_4G_1:${c4g1},gNB_4G_2:${c4g2}" \
          --cellTxPowerDbm="gNB_5G:${tx_power},gNB_4G_1:${tx_power},gNB_4G_2:${tx_power}" \
          --cellHysteresisDb="gNB_5G:${hys_db},gNB_4G_1:${hys_db},gNB_4G_2:${hys_db}" \
          --cellTttMs=gNB_5G:256,gNB_4G_1:256,gNB_4G_2:256 \
          --cellRetBearingDeg=gNB_5G:20,gNB_4G_1:0,gNB_4G_2:10 \
          --cellRetTiltDeg="gNB_5G:${ret_tilt},gNB_4G_1:${ret_tilt},gNB_4G_2:${ret_tilt}"
    ) > "${launcher_log}" 2>&1 &

    pid=$!
    active_pids+=("${pid}")
    active_tags+=("${run_tag}")
    active_cios+=("${c5g},${c4g1},${c4g2}")
    active_cpus+=("${cpu_id}")
    printf '%s,%s,%s,%s,%s,%s\n' "${c5g}" "${c4g1}" "${c4g2}" "${run_tag}" "${pid}" "${cpu_id}" >> "${jobs_file}"
  done

  for ((slot = 0; slot < ${#active_pids[@]}; ++slot)); do
    pid="${active_pids[slot]}"
    if wait "${pid}"; then
      exit_code=0
    else
      exit_code=$?
      sweep_failed=1
    fi
    printf '[complete] CIO=(%s) pid=%s exit=%s\n' "${active_cios[slot]}" "${pid}" "${exit_code}"
    printf '%s,%s,%s,%s,%s,%s\n' "${active_cios[slot]}" "${active_tags[slot]}" \
      "${pid}" "${active_cpus[slot]}" "${exit_code}" "$(date --iso-8601=seconds)" >> "${completion_file}"
  done
done

active_pids=()
trap - INT TERM
if ((sweep_failed != 0)); then
  printf '[CIO combo sweep] one or more runs failed: %s\n' "${sweep_root}" >&2
  exit 1
fi
printf '[CIO combo sweep] all runs completed: %s\n' "${sweep_root}"

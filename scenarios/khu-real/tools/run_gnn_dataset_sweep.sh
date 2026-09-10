#!/usr/bin/env bash
set -euo pipefail

# Distributed GNN-dataset sweep runner. Every contributor's machine runs the
# exact same command with a different (non-overlapping) row range from the
# shared scenarios/khu-real/gnn_dataset_params.csv, so results merge cleanly
# with no coordination beyond agreeing on ranges up front.
#
# Usage:
#   run_gnn_dataset_sweep.sh <start_idx> <end_idx> <cores_per_job> <parallel_jobs>
#
# start_idx/end_idx are the 0-based scenario indices from the "name" column
# (gnn_0000 .. gnn_0999), inclusive on both ends, matching the CSV's own
# seed numbering (row gnn_0042 uses ue_position/ue_positions_seed42.csv).
#
# Example: process gnn_0000..gnn_0099 (100 scenarios), 2 CPU cores per
# simulation, 5 simulations running concurrently (10 cores in use at once):
#   ./run_gnn_dataset_sweep.sh 0 99 2 5
#
# Two people splitting 300 scenarios in half, each using 2 cores/job and
# 5 parallel slots:
#   person A: ./run_gnn_dataset_sweep.sh 0   149 2 5
#   person B: ./run_gnn_dataset_sweep.sh 150 299 2 5
#
# Re-running the same range later skips rows that already completed cleanly
# (cell_kpi.csv exists and run.log has no FATAL/aborted marker), so an
# interrupted run can just be resubmitted as-is.

if (($# != 4)); then
  echo "Usage: $0 <start_idx> <end_idx> <cores_per_job> <parallel_jobs>" >&2
  exit 1
fi

start_idx="$1"
end_idx="$2"
cores_per_job="$3"
parallel_jobs="$4"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
binary="${repo_root}/build/scratch/ns3.48-khu-real-dual-nr-sionna-final-default"
params_csv="${repo_root}/scenarios/khu-real/gnn_dataset_params.csv"
ue_position_dir="${repo_root}/scenarios/khu-real/ue_position"
merge_tool="${script_dir}/build_merged_ret_cache.py"

sim_time="${GNN_SWEEP_SIM_TIME:-900}"
memory_limit_bytes="${GNN_SWEEP_MEMORY_LIMIT_BYTES:-3221225472}"
results_root="${GNN_SWEEP_RESULTS_ROOT:-${repo_root}/scenarios/results/GNN_dataset_sweep}"

if [[ ! -x "${binary}" ]]; then
  printf 'Missing executable: %s (run ./ns3 build khu-real-dual-nr-sionna-final first)\n' "${binary}" >&2
  exit 1
fi
if [[ ! -f "${params_csv}" ]]; then
  printf 'Missing param CSV: %s\n' "${params_csv}" >&2
  exit 1
fi
if ((start_idx < 0 || end_idx < start_idx || cores_per_job < 1 || parallel_jobs < 1)); then
  printf 'Invalid range/parallelism arguments\n' >&2
  exit 1
fi

nproc_avail="$(nproc)"
needed_cores=$((cores_per_job * parallel_jobs))
if ((needed_cores > nproc_avail)); then
  printf 'Warning: requesting %d cores (parallel_jobs*cores_per_job) but only %d logical CPUs available\n' \
    "${needed_cores}" "${nproc_avail}" >&2
fi

mkdir -p "${results_root}"
jobs_file="${results_root}/jobs_${start_idx}_${end_idx}.csv"
completion_file="${results_root}/completion_${start_idx}_${end_idx}.csv"
printf 'name,pid,cpu_range,exit_code,finished_at\n' > "${completion_file}"
printf 'name,cpu_range\n' > "${jobs_file}"

# ---------------------------------------------------------------------------
# Read the requested scenario-index range out of the CSV, matched by the
# "name" column's own 0-based index (gnn_0000 = index 0), not by physical
# CSV row position -- robust to the CSV ever being re-sorted or filtered.
# ---------------------------------------------------------------------------
mapfile -t rows < <(python3 - "${params_csv}" "${start_idx}" "${end_idx}" <<'PY'
import csv, sys
path, start, end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path, newline="") as f:
    reader = list(csv.DictReader(f))
by_index = {}
for r in reader:
    idx = int(r["name"].rsplit("_", 1)[-1])
    by_index[idx] = r
for i in range(start, end + 1):
    r = by_index.get(i)
    if r is None:
        continue
    print(",".join([
        r["name"], r["txp_5g"], r["txp_4g1"], r["txp_4g2"],
        r["ret_5g"], r["ret_4g1"], r["ret_4g2"],
        r["cio_5g"], r["cio_4g1"], r["cio_4g2"],
        r["ttt_ms"], r["hys_db"],
    ]))
PY
)

if ((${#rows[@]} == 0)); then
  printf 'No rows in range [%d, %d] (CSV has %d data rows)\n' \
    "${start_idx}" "${end_idx}" "$(($(wc -l < "${params_csv}") - 1))" >&2
  exit 1
fi
printf '[GNN sweep] rows=%d range=[%d,%d] cores_per_job=%d parallel_jobs=%d simTime=%ss\n' \
  "${#rows[@]}" "${start_idx}" "${end_idx}" "${cores_per_job}" "${parallel_jobs}" "${sim_time}"

# ---------------------------------------------------------------------------
# Pre-build every distinct merged 1.8GHz RET cache this range will need, up
# front and sequentially, so parallel sim workers never race to build the
# same merge (see build_merged_ret_cache.py for why this is safe/cheap: pure
# HDF5 group copies, no ray tracing).
# ---------------------------------------------------------------------------
declare -A seen_pairs=()
for row in "${rows[@]}"; do
  IFS=',' read -r _name _txp5g _txp4g1 _txp4g2 _ret5g ret4g1 ret4g2 _rest <<< "${row}"
  pair_key="${ret4g1}_${ret4g2}"
  if [[ -z "${seen_pairs[${pair_key}]+x}" ]]; then
    seen_pairs["${pair_key}"]=1
  fi
done
printf '[GNN sweep] pre-building %d distinct RET cache pair(s)...\n' "${#seen_pairs[@]}"
for pair_key in "${!seen_pairs[@]}"; do
  IFS='_' read -r ret4g1 ret4g2 <<< "${pair_key}"
  python3 "${merge_tool}" --tilt-5g 9 --tilt-4g1 "${ret4g1}" --tilt-4g2 "${ret4g2}" > /dev/null
done
echo "[GNN sweep] cache pre-build done"

# ---------------------------------------------------------------------------
# Keep `parallel_jobs` worker slots filled, each pinned to its own
# `cores_per_job`-wide CPU range. Reuse a slot as soon as its process exits.
# ---------------------------------------------------------------------------
row_is_done()
{
  local run_dir="$1"
  [[ -f "${run_dir}/cell_kpi.csv" ]] || return 1
  [[ -f "${run_dir}/run.log" ]] || return 1
  ! grep -qE "FATAL|aborted\." "${run_dir}/run.log"
}

active_pids=()
active_names=()
active_cpus=()
sweep_failed=0
terminate_active()
{
  if ((${#active_pids[@]} > 0)); then
    kill -TERM "${active_pids[@]}" 2>/dev/null || true
    wait "${active_pids[@]}" 2>/dev/null || true
  fi
}
trap terminate_active EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

total_rows="${#rows[@]}"
index=0
while ((index < total_rows || ${#active_pids[@]} > 0)); do
  for ((slot = 0; slot < parallel_jobs; ++slot)); do
    if [[ -n "${active_pids[slot]:-}" ]]; then
      pid="${active_pids[slot]}"
      if kill -0 "${pid}" 2>/dev/null; then
        continue
      fi
      if wait "${pid}"; then
        exit_code=0
      else
        exit_code=$?
        sweep_failed=1
      fi
      printf '[complete] %s cpu=%s exit=%s\n' "${active_names[slot]}" "${active_cpus[slot]}" "${exit_code}"
      printf '%s,%s,%s,%s,%s\n' "${active_names[slot]}" "${pid}" "${active_cpus[slot]}" \
        "${exit_code}" "$(date --iso-8601=seconds)" >> "${completion_file}"
      unset 'active_pids[slot]' 'active_names[slot]' 'active_cpus[slot]'
    fi

    # Completed rows must not consume a worker slot.
    while ((index < total_rows)); do
      name="${rows[index]%%,*}"
      if ! row_is_done "${results_root}/${name}"; then
        break
      fi
      printf '[skip] %s already completed\n' "${name}"
      index=$((index + 1))
    done
    if ((index >= total_rows)); then
      continue
    fi

    IFS=',' read -r name txp5g txp4g1 txp4g2 ret5g ret4g1 ret4g2 cio5g cio4g1 cio4g2 ttt hys <<< "${rows[index]}"
    index=$((index + 1))

    run_dir="${results_root}/${name}"
    rm -rf "${run_dir}"

    # Extract the seed index straight out of "gnn_0042" -> 42.
    seed_idx="$((10#${name##*_}))"
    sumo_trace="${ue_position_dir}/ue_positions_seed${seed_idx}.csv"
    if [[ ! -f "${sumo_trace}" ]]; then
      printf 'Missing mobility trace: %s\n' "${sumo_trace}" >&2
      exit 1
    fi

    cache_out="$(python3 "${merge_tool}" --tilt-5g 9 --tilt-4g1 "${ret4g1}" --tilt-4g2 "${ret4g2}")"
    cache18="$(grep '^SIONNA_CACHE_18=' <<< "${cache_out}" | cut -d= -f2-)"
    cache35_from_tilt="scenarios/khu-real/ret_caches/tilt_${ret5g}deg/sionna_rt_cache_3p5ghz_2x2_1x1_lzf.h5"
    cache35="${repo_root}/${cache35_from_tilt}"

    cpu_lo=$((slot * cores_per_job))
    cpu_hi=$((cpu_lo + cores_per_job - 1))
    cpu_range="${cpu_lo}-${cpu_hi}"
    launcher_log="${results_root}/${name}.launcher.log"

    printf '[launch] %s cpu=%s TxP=(%s,%s,%s) RET=(%s,%s,%s) CIO=(%s,%s,%s) TTT=%s HYS=%s\n' \
      "${name}" "${cpu_range}" "${txp5g}" "${txp4g1}" "${txp4g2}" \
      "${ret5g}" "${ret4g1}" "${ret4g2}" "${cio5g}" "${cio4g1}" "${cio4g2}" "${ttt}" "${hys}"

    (
      cd "${repo_root}"
      exec env \
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        taskset -c "${cpu_range}" \
        prlimit --as="${memory_limit_bytes}" \
        "${binary}" \
          --centralFrequency18=1.8e9 --bandwidth18=10e6 \
          --centralFrequency35=3.5e9 --bandwidth35=10e6 \
          --numerology=0 \
          --sionnaCacheFile18="${cache18}" \
          --sionnaCacheFile35="${cache35}" \
          --sionnaUpdatePeriod=1s \
          --udpBaseIntervalMs=200 \
          --trafficMultiplierEnabled=true \
          --simTime="${sim_time}" \
          --kpiReportInterval=1.0 \
          --sumoTrace="${sumo_trace}" \
          --N_Ues=300 \
          --resultsRoot="${results_root}" \
          --runTag="${name}" \
          --handoverTtt="${ttt}" \
          --handoverHysteresis="${hys}" \
          --cellCioDb="gNB_5G:${cio5g},gNB_4G_1:${cio4g1},gNB_4G_2:${cio4g2}" \
          --cellTxPowerDbm="gNB_5G:${txp5g},gNB_4G_1:${txp4g1},gNB_4G_2:${txp4g2}" \
          --cellHysteresisDb="gNB_5G:${hys},gNB_4G_1:${hys},gNB_4G_2:${hys}" \
          --cellTttMs="gNB_5G:${ttt},gNB_4G_1:${ttt},gNB_4G_2:${ttt}" \
          --cellRetBearingDeg=gNB_5G:20,gNB_4G_1:0,gNB_4G_2:10 \
          --cellRetTiltDeg="gNB_5G:${ret5g},gNB_4G_1:${ret4g1},gNB_4G_2:${ret4g2}"
    ) > "${launcher_log}" 2>&1 &

    pid=$!
    active_pids[slot]="${pid}"
    active_names[slot]="${name}"
    active_cpus[slot]="${cpu_range}"
    printf '%s,%s\n' "${name}" "${cpu_range}" >> "${jobs_file}"
  done

  # Bash reaps exited children; kill -0 then lets us wait for each exact PID
  # without blocking on a slower worker or losing an already-finished status.
  if ((${#active_pids[@]} > 0)); then
    sleep 0.2
  fi
done

active_pids=()
trap - EXIT INT TERM
if ((sweep_failed != 0)); then
  printf '[GNN sweep] one or more runs failed -- rerun the same command to retry only the incomplete rows: %s\n' "${results_root}" >&2
  exit 1
fi
printf '[GNN sweep] all rows in range [%d, %d] completed: %s\n' "${start_idx}" "${end_idx}" "${results_root}"

#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
builder="${script_dir}/build_sionna_rt_cache.py"
python_bin="${repo_root}/.venv/bin/python"
cache_root="${RET_CACHE_ROOT:-${repo_root}/scenarios/khu-real/ret_caches}"
ret_values_text="${RET_VALUES:-1 3 5 7 9 12}"
max_parallel="${RET_CACHE_MAX_PARALLEL:-3}"
read -r -a ret_values <<< "${ret_values_text}"

if [[ ! -x "${python_bin}" ]]; then
  printf 'Missing virtual-environment Python: %s\n' "${python_bin}" >&2
  exit 1
fi
if [[ ! -f "${builder}" ]]; then
  printf 'Missing cache builder: %s\n' "${builder}" >&2
  exit 1
fi
if ((max_parallel < 1 || max_parallel > 6)); then
  printf 'RET_CACHE_MAX_PARALLEL must be between 1 and 6, got %s\n' "${max_parallel}" >&2
  exit 1
fi

safe_label()
{
  local label="${1//-/m}"
  printf '%s' "${label//./p}"
}

build_one()
{
  local tilt="$1"
  local frequency="$2"
  local prefix="$3"
  local output="$4"
  local log_file="$5"

  if [[ -f "${output}" && "${RET_FORCE_REBUILD:-0}" != "1" ]]; then
    printf '[skip] existing cache: %s\n' "${output}"
    return
  fi

  local partial="${output}.partial.$$"
  trap 'rm -f -- "${partial}"' RETURN
  printf '[build] tilt=%s frequency=%s prefix=%s output=%s\n' \
    "${tilt}" "${frequency}" "${prefix}" "${output}"
  "${python_bin}" -u "${builder}" \
    --out "${partial}" \
    --frequency "${frequency}" \
    --gnb-rows 2 \
    --gnb-cols 2 \
    --ue-rows 1 \
    --ue-cols 1 \
    --gnb-prefix "${prefix}" \
    --tilt-deg "${tilt}" \
    --max-paths 256 \
    --batch-size 50 \
    --zero-path-policy nearest 2>&1 | tee "${log_file}"
  mv -- "${partial}" "${output}"
  trap - RETURN
}

if ((${#ret_values[@]} == 0)); then
  printf 'RET_VALUES must contain at least one tilt\n' >&2
  exit 1
fi

active_pids=()
active_tilts=()
build_failed=0

wait_for_wave()
{
  local index pid exit_code
  for index in "${!active_pids[@]}"; do
    pid="${active_pids[index]}"
    if wait "${pid}"; then
      exit_code=0
    else
      exit_code=$?
      build_failed=1
    fi
    printf '[complete] tilt=%s pid=%s exit=%s\n' \
      "${active_tilts[index]}" "${pid}" "${exit_code}"
  done
  active_pids=()
  active_tilts=()
}

terminate_active()
{
  if ((${#active_pids[@]} > 0)); then
    kill -TERM "${active_pids[@]}" 2>/dev/null || true
    wait "${active_pids[@]}" 2>/dev/null || true
  fi
}
trap terminate_active INT TERM

printf '[RET cache build] root=%s tilts=%s parallel=%s\n' \
  "${cache_root}" "${ret_values_text}" "${max_parallel}"
for tilt in "${ret_values[@]}"; do
  if [[ ! "${tilt}" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
    printf 'RET value is not numeric: %s\n' "${tilt}" >&2
    exit 1
  fi
  label="$(safe_label "${tilt}")"
  tilt_dir="${cache_root}/tilt_${label}deg"
  mkdir -p "${tilt_dir}"

  (
    set -e
    build_one "${tilt}" 1.8e9 gNB_4G \
      "${tilt_dir}/sionna_rt_cache_1p8ghz_2x2_1x1_lzf.h5" \
      "${tilt_dir}/build_1p8ghz.log"
    build_one "${tilt}" 3.5e9 gNB_5G \
      "${tilt_dir}/sionna_rt_cache_3p5ghz_2x2_1x1_lzf.h5" \
      "${tilt_dir}/build_3p5ghz.log"
  ) &
  active_pids+=("$!")
  active_tilts+=("${tilt}")

  if ((${#active_pids[@]} >= max_parallel)); then
    wait_for_wave
  fi
done

if ((${#active_pids[@]} > 0)); then
  wait_for_wave
fi
trap - INT TERM
if ((build_failed != 0)); then
  printf '[RET cache build] one or more tilt builds failed\n' >&2
  exit 1
fi
printf '[RET cache build] all requested caches completed\n'

#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/../../.." && pwd)"
cache="$(realpath "${RQ3_CACHE_SOURCE:-$repo/experiment/khu/caches/ret_caches}")"
output="${AGORA_OUTPUT:-$repo/experiment/khu/results/agora_evaluation_v2}"
mkdir -p "$output"
output="$(realpath "$output")"
args=(run --rm --init --network none --read-only --cap-drop ALL
 --security-opt no-new-privileges --user "$(id -u):$(id -g)"
 --tmpfs /tmp:rw,nosuid,nodev,size=256m
 --mount "type=bind,src=$repo,dst=/bundle,readonly"
 --mount "type=bind,src=$cache,dst=/cache_source,readonly"
 --mount "type=bind,src=$output,dst=/results"
 --entrypoint python3)
if [[ -n "${AGORA_MEMORY:-}" ]]; then args+=(--memory "$AGORA_MEMORY" --memory-swap "$AGORA_MEMORY"); fi
if [[ -n "${AGORA_CPUSET:-}" ]]; then args+=(--cpuset-cpus "$AGORA_CPUSET"); fi
if [[ -n "${AGORA_CPUS:-}" ]]; then args+=(--cpus "$AGORA_CPUS"); fi
if [[ "${AGORA_DETACH:-0}" == 1 ]]; then args+=(-d --name "${AGORA_NAME:-agora-evaluation-v2}"); fi
exec docker "${args[@]}" "${RQ3_IMAGE:-agora-rq3:cpu-v1}" /bundle/experiment/khu/tools/agora_evaluation_entrypoint.py "$@"

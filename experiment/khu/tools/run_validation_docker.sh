#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/../../.." && pwd)"
image="${RQ3_IMAGE:-agora-rq3:cpu-v1}"
cache="$(realpath "${RQ3_CACHE_SOURCE:-$repo/experiment/khu/caches/ret_caches}")"
output="${VALIDATION_OUTPUT:-$repo/experiment/khu/results/validation_20261111}"
mkdir -p "$output"
output="$(realpath "$output")"
args=(run --rm --init --network none --read-only --cap-drop ALL
 --security-opt no-new-privileges --user "$(id -u):$(id -g)"
 --tmpfs /tmp:rw,nosuid,nodev,size=256m
 --mount "type=bind,src=$repo,dst=/bundle,readonly"
 --mount "type=bind,src=$cache,dst=/cache_source,readonly"
 --mount "type=bind,src=$output,dst=/results"
 --entrypoint python3)
if [[ -n "${RQ3_MEMORY:-}" ]];then args+=(--memory "$RQ3_MEMORY" --memory-swap "$RQ3_MEMORY");fi
if [[ -n "${RQ3_CPUSET:-}" ]];then args+=(--cpuset-cpus "$RQ3_CPUSET");fi
if [[ -n "${RQ3_CPUS:-}" ]];then args+=(--cpus "$RQ3_CPUS");fi
if [[ -n "${RQ3_NAME:-}" ]];then args+=(--name "$RQ3_NAME");fi
if [[ "${RQ3_DETACH:-0}" == 1 ]];then args+=(-d);fi
exec docker "${args[@]}" "$image" /bundle/experiment/khu/tools/validation_entrypoint.py "$@"

#!/usr/bin/env bash
set -euo pipefail
# Run from the server export directory; reserve outputs never share the original results.
reserve_manifest="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/execution_manifest.json"
test -f "$reserve_manifest"
image="${AGORA_IMAGE:-agora-conflict-sweep:cpu-v1}"
input="$(realpath "${AGORA_INPUTS:-./inputs}")"
mkdir -p "${AGORA_RESULTS:-./results-reserve50}"
output="$(realpath "${AGORA_RESULTS:-./results-reserve50}")"
test -d "$input/scenarios/khu-real/ret_caches"
test -d "$input/scenarios/khu-real/ue_position"
args=(run --rm --init --network none --read-only --cap-drop ALL --security-opt no-new-privileges
  --env NUMPY_MADVISE_HUGEPAGE=0
  --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,nosuid,nodev,size=256m
  --mount "type=bind,src=$input/scenarios/khu-real,dst=/opt/agora/scenarios/khu-real,readonly"
  --mount "type=bind,src=$reserve_manifest,dst=/opt/agora/experiments/conflict_scenarios_500/sweep/data/execution_manifest.json,readonly"
  --mount "type=bind,src=$output,dst=/opt/agora/results")
if [[ -n "${AGORA_MEMORY:-}" ]]; then args+=(--memory "$AGORA_MEMORY" --memory-swap "$AGORA_MEMORY"); fi
if [[ -n "${AGORA_CPUS:-}" ]]; then args+=(--cpus "$AGORA_CPUS"); fi
if [[ -n "${AGORA_NAME:-}" ]]; then args+=(--name "$AGORA_NAME"); fi
if [[ "${AGORA_DETACH:-0}" == 1 ]]; then args+=(-d); fi
exec docker "${args[@]}" "$image" "$@"

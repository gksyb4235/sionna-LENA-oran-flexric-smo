#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/../../../.." && pwd)"
export RQ3_CACHE_SOURCE="${RQ3_CACHE_SOURCE:-$HOME/xapp_conflict_experiments/inputs/scenarios/khu-real/ret_caches}"
export VALIDATION_OUTPUT="${VALIDATION_OUTPUT:-$HOME/xapp_conflict_experiments/validation_20261111}"
export RQ3_NAME="${RQ3_NAME:-agora-validation-server}"
export RQ3_DETACH=1
exec bash "$repo/experiment/khu/tools/run_validation_docker.sh" --partition server_container --workers 28 "$@"

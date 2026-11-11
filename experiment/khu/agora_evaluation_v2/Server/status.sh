#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/../../../.." && pwd)"
export RQ3_CACHE_SOURCE="${RQ3_CACHE_SOURCE:-$HOME/xapp_conflict_experiments/inputs/scenarios/khu-real/ret_caches}"
export AGORA_OUTPUT="${AGORA_OUTPUT:-$HOME/xapp_conflict_experiments/agora_evaluation_v2}"
exec bash "$repo/experiment/khu/tools/run_agora_evaluation_docker.sh" --partition server --status "$@"

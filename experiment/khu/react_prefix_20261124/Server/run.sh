#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/../../../.." && pwd)"
export AGORA_DETACH=1
exec bash "$repo/experiment/khu/tools/run_react_prefix_docker.sh" "$@"

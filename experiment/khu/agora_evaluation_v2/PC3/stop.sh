#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/../../../.." && pwd)"
exec python3 "$repo/experiment/khu/tools/agora_evaluation_stop.py" --partition pc3 "$@"

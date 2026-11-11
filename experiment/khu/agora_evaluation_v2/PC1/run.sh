#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/../../../.." && pwd)"
exec python3 "$repo/experiment/khu/tools/agora_evaluation_start.py" --partition pc1 "$@"

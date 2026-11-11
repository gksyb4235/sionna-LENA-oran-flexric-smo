#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/../../../.." && pwd)"
exec python3 "$repo/experiment/khu/tools/start_validation_native.py" --partition pc4 "$@"

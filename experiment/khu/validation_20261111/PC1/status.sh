#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/../../../.." && pwd)"
printf "Existing PC1 retries (16):\n"
python3 "$repo/experiment/khu/tools/validation_sweep.py" --partition local_retry --output "$repo/experiment/khu/results/rq3_retry_20261111" --status
printf "New PC1 queue (80):\n"
exec python3 "$repo/experiment/khu/tools/validation_sweep.py" --partition pc1 --status "$@"

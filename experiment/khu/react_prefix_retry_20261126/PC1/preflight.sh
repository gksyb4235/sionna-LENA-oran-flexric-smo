#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/../../../.." && pwd)"
exec python3 "$repo/experiment/khu/tools/react_prefix_host.py" PC1 --batch react_prefix_retry_20261126 --preflight "$@"

#!/usr/bin/env bash
set -euo pipefail
# Inclusive, zero-based job IDs; identical positional interface to the old sweep.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec python3 "${script_dir}/rq3_sweep.py" "$@"

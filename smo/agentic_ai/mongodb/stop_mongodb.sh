#!/usr/bin/env bash
set -euo pipefail
if [ -f "/home/ubuntu/jaechan/mongodb/run/mongod.pid" ]; then
  kill "$(cat "/home/ubuntu/jaechan/mongodb/run/mongod.pid")"
fi

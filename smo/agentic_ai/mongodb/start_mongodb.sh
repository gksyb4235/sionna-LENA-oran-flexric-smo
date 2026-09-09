#!/usr/bin/env bash
set -euo pipefail
"/home/ubuntu/jaechan/mongodb/current/bin/mongod" --dbpath "/home/ubuntu/jaechan/mongodb/standalone-data" --bind_ip_all --port 27017 --auth --fork --logpath "/home/ubuntu/jaechan/mongodb/logs/mongod.log" --pidfilepath "/home/ubuntu/jaechan/mongodb/run/mongod.pid"

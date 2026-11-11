# All completed planner variants: distributed ns-3 validation

This manifest covers Full AGORA ReAct 2/3/4/5, LLM-only, and SHAP Greedy,
120 conditions each. No new LLM calls or model training are required.
It retains all 120 LLM-only submissions, including predicted violations.
No-plan outcomes are explicitly labeled hold-current fallbacks, not successful plans.

## Execution inventory

There are 449 unique trajectories and 912 method-condition mappings.
82 already completed, prefix-consistent trajectories are reused.
The remaining 367 consist of 16 unfinished retries, 6 prefix rechecks,
and 345 newly needed unique trajectories. Existing rq3_0000..0103 IDs
are unchanged; new IDs are rq3_0104..0448.

Five personal PCs (including the owner's current PC) plus one server:

| Machine | Queue | Shard index | Jobs |
|---|---|---:|---:|
| Current PC | local_retry; already launched separately | - | 16 |
| Other PC 1 | peer_native | 0 of 4 | 45 |
| Other PC 2 | peer_native | 1 of 4 | 45 |
| Other PC 3 | peer_native | 2 of 4 | 45 |
| Other PC 4 | peer_native | 3 of 4 | 44 |
| Server | server_container | no sharding | 172 |

Each simulation receives exactly one logical CPU. Workers controls how many
simulations run simultaneously, with remaining jobs queued. Do not assign the
same shard to two machines. Do not use old numeric-range scripts for this queue.

## Other personal PCs: native Linux / WSL

Use the existing repository and RET cache installation. Set PC_INDEX differently
on each PC: 0, 1, 2, 3. Do not run this queue on the current owner's PC while its
16 retries occupy all cores.

```bash
cd ~/LENA-oran-flexric-smo
git pull --ff-only origin main
PC_INDEX=0
WORKERS=$(nproc)
python3 experiment/khu/tools/validation_sweep.py \
  --partition peer_native --shards 4 --shard-index "$PC_INDEX" \
  --workers "$WORKERS" --preflight

# Run only after preflight succeeds.
nohup python3 -u experiment/khu/tools/validation_sweep.py \
  --partition peer_native --shards 4 --shard-index "$PC_INDEX" \
  --workers "$WORKERS" \
  > "validation_pc${PC_INDEX}.log" 2>&1 < /dev/null &
tail -f "validation_pc${PC_INDEX}.log"
```

The default address-space cap is 8 GiB per process, raised from the previous
3 GiB after a std::bad_alloc failure. This is a cap, not preallocated RAM.
Set WORKERS to the cores assigned to this experiment on shared machines.
The runner checks CPU affinity and prevents simultaneous native queues under
the same user from reserving the same cores.

If preflight reports a backend source mismatch, stop and inspect that mismatch.
Do not disable hash checks or rebuild only some runs without recording the new
runtime. The container command below can also run a peer_native shard when the
same frozen image is installed on the personal PC.

## Server: container execution

Use the previously supplied `agora-rq3:cpu-v1` image. The new plans and scripts
are mounted from Git; no Docker image rebuild is needed. The original runtime,
libraries, and RET caches are hash-checked by the frozen image adapter.

```bash
# First obtain/update this same Git repository on the server.
cd ~/LENA-oran-flexric-smo
git pull --ff-only origin main

# Existing server cache and output location from the previous RQ3 deployment.
export RQ3_CACHE_SOURCE="$HOME/xapp_conflict_experiments/inputs/scenarios/khu-real/ret_caches"
export VALIDATION_OUTPUT="$HOME/xapp_conflict_experiments/validation_20261111"
bash experiment/khu/tools/run_validation_docker.sh --resources

# Set to the CPU allocation available for this experiment.
SERVER_WORKERS=16
bash experiment/khu/tools/run_validation_docker.sh \
  --partition server_container --workers "$SERVER_WORKERS" --preflight

# Run only after preflight succeeds. Survives SSH disconnection.
RQ3_NAME=agora-validation-server RQ3_DETACH=1 \
  bash experiment/khu/tools/run_validation_docker.sh \
  --partition server_container --workers "$SERVER_WORKERS"
docker logs -f agora-validation-server
```

If the image has not been loaded, load the previous package's
`agora-rq3-cpu-v1.tar.gz` with `docker load -i ...` first.
For a different server checkout location, run the commands from that checkout;
the wrapper resolves it automatically. The cache path is independently set by
RQ3_CACHE_SOURCE, and the old results directory is not overwritten.

Optional Docker limits: RQ3_CPUSET, RQ3_CPUS, RQ3_MEMORY.
The runner checks the container's CPU quota. Run one production container per
output directory. Each container is network-disabled and has read-only input
mounts. New results alone are writable.

## Status, retries and returning results

```bash
# On a peer PC, read-only status:
python3 experiment/khu/tools/validation_sweep.py \
  --partition peer_native --shards 4 --shard-index "$PC_INDEX" --status

# On the server, read-only status:
bash experiment/khu/tools/run_validation_docker.sh \
  --partition server_container --status
```

After a failed/interrupted attempt, inspect its run.log. To retry, use the same
queue command with `--retry-failed`. Completed jobs with the same provenance and
unchanged output hashes are skipped. Previous incomplete attempts are archived,
never silently overwritten. No automatic unlimited failure loop is used.

The current PC's 16 retries write to:
`experiment/khu/results/rq3_retry_20261111`.
Other native PCs write to:
`experiment/khu/results/validation_20261111`.
The server writes to VALIDATION_OUTPUT above.

Return complete job directories, execution.json, launcher.log,
simulation/run/{cell_kpi.csv,ue_kpi.csv,parameter_changes.csv,run.log}, and _batches.
Keep outputs grouped by source machine; do not overwrite the earlier imported
rq3_final. For the six prefix rechecks, retain both original and repeated
attempts regardless of which performs better. A re-execution does not itself
prove pre-intervention equivalence; audit that again after collecting results.

## Validation

`python3 experiment/khu/tools/test_validation_sweep.py` checks queue coverage,
overlap, bounds, shard composition, every selected schedule against its source
plan, preservation of pre-origin controls, and retention of LLM-only violations.
The local native and container preflights check source, input, cache and runtime
hashes. This validates execution preparation, not 900-second simulation success.

R5 prefix replay: one cycle across five hosts (2026-11-25 KST)

PC1:16 (already running), PC2:22, PC3:14, PC4:12, Server:21.
All 85 jobs are disjoint. Each ns-3 process gets one CPU core.
No old failed jobs, mismatch repairs, or new LLM calls are included.
Do NOT run the entire r5_new queue on another PC.

On PC2/PC3/PC4, from the existing LENA-oran-flexric-smo repository:
  git pull origin main
  ./ns3 build khu-agora-conflict-sweep -j 2
  bash experiment/khu/react_prefix_20261124/PC2/run.sh
Replace PC2 with PC3 or PC4 for the relevant machine.
The command launches a detached background process. No second terminal is needed.
  bash experiment/khu/react_prefix_20261124/PC2/status.sh
  tail -f experiment/khu/results/react_prefix_20261124/PC2/host_launcher.log

PC1: use status.sh only. Its running 16 jobs are preserved.
The old 85-job parent is suspended so it cannot dispatch the other 69 jobs.
A completion-only monitor preserves each child's actual exit status and verifies outputs.
Do not resume or kill this suspended parent manually.

Prerequisites: the previous frozen inputs, RET caches, and built simulator.
The same backend/source hashes are checked before ns-3 starts.
If a binary rebuild is required, use the existing build workflow for
khu-agora-conflict-sweep. Do not bypass a preflight mismatch.
Server: run the following on the HOST, using the existing agora-rq3:cpu-v1 image.
No host simulator build or manual container entry is required.
  git pull origin main
  bash experiment/khu/react_prefix_20261124/Server/preflight.sh
  bash experiment/khu/react_prefix_20261124/Server/run.sh
run.sh creates the detached Docker container agora-r5-prefix-server, with
21 workers, one core each. The image supplies the frozen simulator at /opt/agora.
The repository and existing RET caches are mounted read-only; results persist
on the host. Preflight verifies image runtime, cache and backend/source hashes.
Defaults (same cache location as the previous server experiment):
  RQ3_CACHE_SOURCE=$HOME/xapp_conflict_experiments/inputs/scenarios/khu-real/ret_caches
  AGORA_OUTPUT=$HOME/xapp_conflict_experiments/react_prefix_20261124/Server
  RQ3_IMAGE=agora-rq3:cpu-v1
  AGORA_CPUS=21
RQ3_CACHE_SOURCE, AGORA_OUTPUT, AGORA_CPUSET, AGORA_MEMORY can be overridden.
At least 21 allowed CPUs are needed (the previous 28-CPU allocation is valid).
  docker logs -f agora-r5-prefix-server
  bash experiment/khu/react_prefix_20261124/Server/status.sh
The detached container is retained after exit to preserve its log.
Memory depends on the simulator; the 8 GiB per-process limit is not a reservation.

Outputs: experiment/khu/results/react_prefix_20261124/<HOST>/
PC1 keeps its already-started output directly under react_prefix_20261124/.
Server outputs use the host AGORA_OUTPUT path described above instead.
Return the complete host output directory, including execution.json and
simulation/run, to the Windows package's incoming/<HOST>/ directory.
Never commit generated simulation results to Git.

These are prefixes of one archived five-round search, not independent budgets.
Any new invalid result is excluded uniformly from all five round comparisons.

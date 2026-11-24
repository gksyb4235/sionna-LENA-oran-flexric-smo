R5 prefix replay: one cycle across five hosts (2026-11-25 KST)

PC1:16 (already running), PC2:22, PC3:14, PC4:12, Server:21.
All 85 jobs are disjoint. Each ns-3 process gets one CPU core.
No old failed jobs, mismatch repairs, or new LLM calls are included.
Do NOT run the entire r5_new queue on another PC.

On each host, from the existing LENA-oran-flexric-smo repository:
  git pull origin main
  bash experiment/khu/react_prefix_20261124/PC2/run.sh
Replace PC2 with PC3, PC4, or Server for the relevant machine.
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
Containers require the existing persistent repository and caches, and at least
21 available CPUs for the Server partition (28-CPU allocation is also valid).
Memory depends on the simulator; the 8 GiB per-process limit is not a reservation.

Outputs: experiment/khu/results/react_prefix_20261124/<HOST>/
PC1 keeps its already-started output directly under react_prefix_20261124/.
Return the complete host output directory, including execution.json and
simulation/run, to the Windows package's incoming/<HOST>/ directory.
Never commit generated simulation results to Git.

These are prefixes of one archived five-round search, not independent budgets.
Any new invalid result is excluded uniformly from all five round comparisons.

Interrupted Server replay retry (2026-11-26 KST)

Only the 21 interrupted Server jobs are included, split PC1=12 and PC3=9.
One core per job; each host launches its whole partition in one cycle.
rq3_0705 (Removing incomplete header) is excluded, as are completed jobs.
The original schedules, seed/run=1/1, and frozen inputs are unchanged.
This restarts from simulation time 0 through 900 seconds, not from 700 seconds.
No original outputs are overwritten; the old interrupted outputs remain intact.
No LLM calls or GNN inference are performed.

From the existing LENA-oran-flexric-smo checkout:
  git pull --ff-only origin main
  bash experiment/khu/react_prefix_retry_20261126/PC1/preflight.sh
  bash experiment/khu/react_prefix_retry_20261126/PC1/run.sh
On PC3 replace PC1 with PC3 in both commands.
These use native Linux/WSL; no server or Docker job is launched.
The existing frozen simulator build and RET caches are reused.

Status and log (replace PC1 with PC3 as appropriate):
  bash experiment/khu/react_prefix_retry_20261126/PC1/status.sh
  tail -f experiment/khu/results/react_prefix_retry_20261126/PC1/host_launcher.log

Outputs:
  experiment/khu/results/react_prefix_retry_20261126/PC1/
  experiment/khu/results/react_prefix_retry_20261126/PC3/
Return complete host directories, including execution.json and simulation/run.
Do not copy these over the original interrupted folders.
Do not run the old full r5_new queue or the old Server partition.

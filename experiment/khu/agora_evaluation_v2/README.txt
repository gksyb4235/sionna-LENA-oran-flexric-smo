Canonical AGORA replay v2. Do not run the legacy validation_20261111 queues.
No simulation is launched during packaging. Each job uses one CPU.
Run your machine preflight.sh then run.sh. Final partial waves are allowed.
Current allocation: PC1=95/16 workers, PC2=138/23, PC3=84/14, PC4=72/12, Server=112/28.
PC1..PC4 have six nominal waves; Server has four. PC1 last wave has fifteen jobs.
allocation.json is the current machine assignment; the replay manifest hash remains frozen.
Original outputs and 449 legacy IDs are preserved. Invalid old Greedy jobs are archival only.
The manifest must have ready_for_execution=true before execution.
Import and validate remote PC2 attempts before assuming any are reusable.

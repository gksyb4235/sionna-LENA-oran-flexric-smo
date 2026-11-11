PC1 is the current computer: 96 total jobs = 16 already-started retries + 80 queued jobs.
16 one-core workers, 6 total cycles (1 existing + 5 new).
Do not restart the existing retries. run.sh starts ONLY the 80 new jobs and refuses to run while the existing retry processes are alive.
After the current retry wave stops, run preflight.sh, then run.sh.
status.sh shows the existing retry queue and the new queue separately.
Existing results: experiment/khu/results/rq3_retry_20261111
New results: experiment/khu/results/validation_20261111

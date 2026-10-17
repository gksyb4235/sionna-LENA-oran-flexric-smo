# Experiment directory map

| Directory | Purpose | Entry point |
|---|---|---|
| conflict_scenarios_500/sweep | Current 500-case scheduled transition dataset | `python3 scripts/conflict_sweep.py setup --build` |
| two_switch_900s | Earlier paired TxP/RET two-change experiment | `scripts/two_switch_batch.py` |
| drl_xapp_offline | Frozen policies pretrained on fixed-run data | `evaluation/README.md` in that directory |
| drl_xapp_conflict | Separate online DRL pilot | Directory README |
| agora_runtime | Local runtime, backups and prior RIC validation | Not versioned |
| agora_conflict_sweep | New sweep result root | Not versioned |

For git pull and distributed execution, start with [SWEEP.md](../SWEEP.md).
The 500-case experiment is the current data collection entry point. Earlier
DRL/two-switch code is retained for reproducibility, not silently merged into it.

Source, tests and frozen input designs are versioned. Local execution receipts,
KPI files, checkpoints, binaries and failed attempts stay at their original paths
and are ignored by Git. No experiment output was moved or deleted by this cleanup.
The offline DRL policies require the separately preserved training artifacts;
git pull does not distribute those historical checkpoints.

Generated scratch targets are installed from each experiment's backend sources.
Do not edit the generated copy as the only source of truth. The tracked common
scenario depends on `scratch/scheduled-parameter-control.h`, which is included.

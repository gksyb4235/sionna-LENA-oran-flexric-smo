# HYS sweep results

The HYS sweep runner stores each invocation under a timestamped `sweep_*`
directory here. Each sweep contains:

- `sweep_config.txt`: shared radio, CPU, memory, and parameter settings
- `jobs.csv`: HYS value, PID, assigned CPU, and launcher log
- `completion.csv`: exit status for every run
- `HYS_*dB_*.launcher.log`: output before the scenario redirects to its run log
- one `HYS_*dB_*` directory per value, containing `command.sh`, `run.log`,
  `ue_kpi.csv`, and `cell_kpi.csv`

Run the complete 0.0-5.0 dB sweep from the repository root with:

```bash
./scenarios/khu-real/tools/run_hys_sweep.sh
```

The default is four concurrent simulations pinned to logical CPUs 0, 2, 4,
and 6, which are four distinct physical cores on this host. Every process has
a 3 GiB virtual-address-space hard limit. The eleven HYS values execute as
three waves of 4, 4, and 3 runs.

Measured with 30-second simulations on this host:

| Concurrent runs | Mean wall time at simTime 29 s | Slowdown vs. one run |
|---:|---:|---:|
| 1 | 20.89 s | baseline |
| 2 | 20.91 s | 0.1% |
| 3 | 21.36 s | 2.2% |
| 4 | 21.87 s | 4.7% |
| 7 | 25.87 s | 23.8% |

Use `HYS_MAX_PARALLEL=2` if strict zero slowdown is more important than total
completion time. Avoid values above 4 for long runs unless the per-run
slowdown is acceptable.

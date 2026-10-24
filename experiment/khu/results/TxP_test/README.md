# TxP sweep results

Run from the repository root:

```bash
./scenarios/khu-real/tools/run_txp_sweep.sh
```

The default sweep applies 31, 33, 35, 37, 39, 41, and 43 dBm to all three
gNBs simultaneously. It runs at most four simulations in parallel and stores
each run's log and KPI CSV files below a timestamped `sweep_*` directory.

Override values or concurrency when needed:

```bash
TXP_VALUES="35 39 43" TXP_MAX_PARALLEL=2 \
  ./scenarios/khu-real/tools/run_txp_sweep.sh
```

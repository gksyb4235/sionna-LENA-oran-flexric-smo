# CIO sweep results

Run from the repository root:

```bash
./scenarios/khu-real/tools/run_cio_sweep.sh
```

The default sweep applies -6, -4, -2, 0, 2, 4, and 6 dB to all three cells
simultaneously. In A3, equal serving- and neighbor-cell CIO values cancel as
`Ocn - Ocp`; this sweep is therefore an invariance/control test, not a useful
cell-selection optimization. Use unequal per-cell CIO values for biasing.

Override values or concurrency when needed:

```bash
CIO_VALUES="-3 0 3" CIO_MAX_PARALLEL=2 \
  ./scenarios/khu-real/tools/run_cio_sweep.sh
```

# KHU A3 handover TTT/HYS sweep

## Fixed conditions

- Simulation time: 200 s
- Sionna update period: 1 s
- UE measurement trace: `scenarios/khu-real/ue-handover-1.csv`
- gNB positions: `scenarios/khu-real/gnbs-ret.csv`
- Numerology: 0
- Algorithm: `NrA3RsrpHandoverAlgorithm`
- One UE, two gNBs, same deterministic streams/path-solver seed for every run
- GUI and E2 disabled for the batch runs; neither participates in the local A3 decision
- InfluxDB KPI interval: 1 s

Before the sweep, `TimeToTrigger` and `Hysteresis` configuration was moved before
`InstallGnbDevice()`. This is required because the gNB handover-algorithm objects
are constructed during device installation.

## Main results

| TTT (ms) | HYS (dB) | START | END_OK | Incomplete | Ping-pong | PDR | Flow throughput (Mbps) | Zero-throughput samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 24 | 23 | 1 | 21 | 89.15% | 0.375155 | 21 |
| 0 | 3 | 20 | 17 | 3 | 15 | 81.09% | 0.341238 | 35 |
| 0 | 6 | 11 | 10 | 1 | 7 | 66.22% | 0.278635 | 65 |
| 256 | 0 | 23 | 21 | 2 | 20 | 81.81% | 0.344276 | 32 |
| 256 | 3 | 16 | 15 | 1 | 13 | 79.18% | 0.333181 | 39 |
| **256** | **6** | **8** | **8** | **0** | **6** | **92.97%** | **0.391228** | **11** |
| 1024 | 0 | 16 | 15 | 1 | 13 | 88.45% | 0.372202 | 22 |
| 1024 | 3 | 10 | 9 | 1 | 7 | 87.26% | 0.367182 | 24 |
| **1024** | **6** | **6** | **5** | **1** | **3** | **93.43%** | **0.393169** | **11** |

`Incomplete` means a `HandoverStart` without a corresponding `HandoverEndOk` in
the trace. It is evidence of an unsuccessful/incomplete procedure, but a dedicated
failure trace is needed to classify its exact RRC/RACH cause.

The baseline `(256 ms, 3 dB)` event/flow values come from the supplied terminal
log. Its KPI aggregates come from `nr_kpi_ttt0256_hys30`; the rerun reproduced the
last handover at 198.256 s and a final `ho_count` of 15.

## Interpretation

- Increasing either TTT or HYS strongly suppresses handovers and ping-pong.
- At HYS 0 dB, increasing TTT from 0 to 1024 ms reduces successful HO from 23 to
  15 and ping-pong from 21 to 13.
- At TTT 1024 ms, increasing HYS from 0 to 6 dB reduces successful HO from 15 to
  5 and ping-pong from 13 to 3.
- HYS 6 dB with TTT 0 ms performs badly: only 66.22% PDR and 65 zero-throughput
  samples. A large amplitude margin alone does not guarantee a stable procedure.
- `(1024, 6)` has the best aggregate PDR/throughput and the fewest handovers, but
  the 185.424 s `1 -> 2` attempt never reaches `END_OK`.
- `(256, 6)` is the robust operating point in this grid: nearly the same throughput
  as `(1024, 6)`, no incomplete HO, and far fewer handovers than the baseline.

## KPI caveat

The current `rsrp_serving_dbm` state is refreshed from RRC measurement-report
events, and SINR-to-IMSI mapping is also established through those reports. Hence
RSRP/SINR sample availability changes with TTT/HYS and those averages are not an
unbiased, continuously sampled RF comparison. Handover events, UDP packet delivery,
flow throughput, and zero-throughput counts are the stronger comparison metrics in
this sweep. A future revision should log periodic per-cell PHY RSRP independently
of Event A3 and add explicit HO-failure/RLF traces.

Each non-baseline run has a matching `ttt*.log` file and an InfluxDB database named
`nr_kpi_tttTTTT_hysHH`. Full numeric aggregates are in `summary.csv`.

## Ping-pong sensitivity to the time window

Ping-pong is counted only when a successful handover reverses the immediately
preceding successful handover and its completion-time interval is strictly less
than the selected window. Incomplete `HandoverStart` events are excluded. A small
comparison tolerance is used because log timestamps are printed to four decimal
places; the resulting 30 s counts match the simulator's original KPI counts.

| TTT (ms) | HYS (dB) | <1 s | <3 s | <5 s | <10 s | <30 s |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 3 | 10 | 14 | 19 | 21 |
| 0 | 3 | 3 | 7 | 9 | 12 | 15 |
| 0 | 6 | 2 | 5 | 6 | 6 | 7 |
| 256 | 0 | 1 | 9 | 11 | 16 | 20 |
| 256 | 3 | 1 | 4 | 4 | 9 | 13 |
| 256 | 6 | 1 | 3 | 4 | 4 | 6 |
| 1024 | 0 | 0 | 5 | 5 | 11 | 13 |
| 1024 | 3 | 0 | 1 | 1 | 3 | 7 |
| 1024 | 6 | 0 | 0 | 0 | 0 | 3 |

The machine-readable version is `pingpong-windows.csv`.

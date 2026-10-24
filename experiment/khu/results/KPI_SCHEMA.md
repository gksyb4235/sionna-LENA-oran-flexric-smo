# Dual-NR final KPI schema

`khu-real-dual-nr-sionna-final.cc` creates one timestamped directory under
`scenarios/results` for every run. Each directory contains:

- `run.log`: continuously flushed stdout/stderr
- `command.sh`: exact command and working directory
- `run.pid`: launch process ID
- `ue_kpi.csv`: one row per active logical UE and KPI interval
- `cell_kpi.csv`: one row per cell and KPI interval

The physical UE/IMSI is reused by the slot pool. Long-window analysis must group
subscriber data by `(ue, session_uid)`, not by IMSI alone.

## Aggregation rules

For a 60- or 180-second window, sum these raw quantities first:

```text
tx/rx packets and bytes
confirmed lost packets
delay samples, delay sum, and delay histogram bins
TB total/corrupt/retransmission/bytes
PRB used/capacity
HO attempts/successes/failures/ping-pongs/RLFs
interval energy and delivered megabits
```

Then recompute ratios from the summed numerator and denominator:

```text
goodput_mbps = sum(rx_bytes) * 8 / (window_s * 1e6)
generated_load_mbps = sum(tx_bytes) * 8 / (window_s * 1e6)
load_satisfaction = sum(rx_bytes) / sum(tx_bytes)
confirmed_loss_ratio = sum(confirmed_lost) /
                       (sum(confirmed_lost) + sum(rx_packets))
DL_BLER = sum(tb_corrupt) / sum(tb_total)
HARQ_retx_ratio = sum(tb_retx) / sum(tb_total)
PRB_utilization = sum(prb_used) / sum(prb_capacity)
HO_success_rate = sum(HO_successes) / sum(HO_attempts)
ping_pong_rate = sum(ping_pongs) / sum(HO_successes)
Mbit_per_J = sum(delivered_megabits) / sum(interval_energy_j)
```

Do not average per-second ratios or P5/P95/P99 values. Recompute signal and
goodput percentiles from the corresponding UE rows. Delay percentiles can be
approximated after summing the interval histogram bins.

## UE quantities

```text
goodput_mbps = rx_bytes_interval * 8 / (interval_s * 1e6)
delivery_ratio_interval = rx_packets_interval / tx_packets_interval
delivery_ratio_cumulative = rx_packets_cumulative / tx_packets_cumulative
confirmed_loss_ratio_cumulative =
    confirmed_lost_packets_cumulative /
    (confirmed_lost_packets_cumulative + rx_packets_cumulative)
dl_bler_interval = tb_corrupt_interval / tb_total_interval
harq_retx_ratio_interval = tb_retx_interval / tb_total_interval
tbler_mean_interval = sum(predicted TB error probability) / tb_total_interval
```

`confirmed_lost_packets_interval` is the delta of the UDP sequence-number loss
counter since the preceding report. The counter uses a 128-packet reordering
window, so a loss may be confirmed later than the interval in which the packet
was transmitted.

Application delay is measured from the `SeqTsHeader` transmit timestamp to UDP
server reception. The scenario configures zero S1 and remote-host propagation
delay, so it is primarily RAN/application delay rather than realistic public
network end-to-end delay.

`delay_sum_ms_interval` and `delay_samples_interval` are directly additive. The
eleven delay histogram columns are mutually exclusive interval bins:

```text
delay_bin_le_1ms_count
delay_bin_1_2ms_count
delay_bin_2_5ms_count
delay_bin_5_10ms_count
delay_bin_10_20ms_count
delay_bin_20_50ms_count
delay_bin_50_100ms_count
delay_bin_100_200ms_count
delay_bin_200_500ms_count
delay_bin_500_1000ms_count
delay_bin_gt_1000ms_count
```

The bins sum exactly to `delay_samples_interval`; they are not cumulative CDF
columns.

Handover interruption is recorded once, on the first KPI row after the first
post-HO packet arrives:

```text
ho_measurement_event = 1
ho_event_packet_gap_ms = first post-HO RX - last pre-HO RX
ho_event_excess_interruption_ms =
    max(0, ho_event_packet_gap_ms - configured packet interval)
```

Rows with `ho_measurement_event=0` contain `nan` in the three `ho_event_*`
columns. `ho_failure_cause_event` is populated only when
`ho_failures_interval > 0`.

## Cell quantities

```text
cell_goodput_mbps = sum(active UE goodput_mbps)
avg_ue_goodput_mbps = cell_goodput_mbps / num_ues
generated_load_mbps = tx_bytes_interval * 8 / (interval_s * 1e6)
load_satisfaction_ratio = rx_bytes_interval / tx_bytes_interval
confirmed_loss_ratio_interval =
    confirmed_lost_packets_interval /
    (confirmed_lost_packets_interval + rx_packets_interval)
prb_utilization_pct = prb_used_reg / prb_capacity_reg * 100
```

An interval satisfaction ratio can exceed one when packets transmitted just
before the interval boundary arrive just after it. Aggregate byte counters over
the desired window before interpreting satisfaction.

`rsrp_avg_linear_dbm` and `sinr_avg_linear_db` average power in the linear
domain and convert the result back to dB. Per-tick P5 values with only a few UEs
behave nearly like a minimum; recompute long-window cell-edge percentiles from
UE rows instead of averaging cell P5 columns.

Per-cell interval traffic/TB/delay counters are assigned to the UE's confirmed
serving cell at the report tick. An interval containing a handover can therefore
contain a small amount of pre-HO traffic attributed to the target cell. The
error is bounded by one KPI interval; use UE/session totals for exact
subscriber-level results.

## Energy model

```text
rf_tx_power_w = 10^(tx_power_dbm / 10) / 1000

ON:    bs_total_power_w = 130 + 4.7 * rf_tx_power_w
SLEEP: bs_total_power_w = 30
OFF:   bs_total_power_w = 5

interval_energy_j = bs_total_power_w * interval_s
delivered_megabits_interval = cell_goodput_mbps * interval_s
```

The model is TxP/state dependent, not PRB-load or bandwidth dependent. Compute
long-window Mbit/J from summed delivered megabits and energy. With equal TxP and
all cells ON, Mbit/J mainly ranks traffic carried per cell rather than a detailed
hardware-efficiency model.

## Mobility counting rules

- `ho_attempt_count` increments on a valid A3 HO start.
- `ho_success_count` increments on `HandoverEndOk`.
- `ho_failure_count` is split into no-preamble, max-RACH, leaving-timeout and
  joining-timeout causes where the trace provides one.
- `pingpong_count` increments when a successful HO reverses the preceding
  successful HO within `pingPongWindow`.
- `rlf_count` increments from the UE RRC `RadioLinkFailure` trace.
- The radio may execute an immediate correction HO after a logical session
  appears. HO events during `slotSettlingHoWindow` after every session start are
  excluded from mobility counters, for both fresh attach and reused slots.
- Events belonging to an inactive/replaced logical UE are excluded.

## Per-cell controls

Maps use comma-separated `cell:value` entries:

```text
--cellCioDb=gNB_5G:3,gNB_4G_1:0,gNB_4G_2:0
--cellTxPowerDbm=gNB_5G:43,gNB_4G_1:43,gNB_4G_2:43
--cellHysteresisDb=gNB_5G:5,gNB_4G_1:5,gNB_4G_2:5
--cellTttMs=gNB_5G:512,gNB_4G_1:512,gNB_4G_2:512
--cellRetTiltDeg=gNB_5G:15,gNB_4G_1:15,gNB_4G_2:15
--cellRetBearingDeg=gNB_5G:10,gNB_4G_1:0,gNB_4G_2:0
```

These are experimental configuration/state columns, not performance KPI. They
remain in `cell_kpi.csv` so runtime controller changes can be aligned with KPI
windows. Unspecified cells use the common TxP/HYS/TTT fallback or RET values
from `gnbs-ret.csv`; CIO defaults to zero.

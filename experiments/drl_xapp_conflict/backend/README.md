# Isolated ns-3 backend

This adapter runs two research controllers against actual live ns-3 objects.
It uses local file lockstep, **not E2/RIC transport**. Existing scenarios and NR
libraries are not edited. E2, AGORA control, and scheduled parameter actions
must be disabled. All runs need distinct, fresh control directories.

Install using the existing WSL Python environment:

```bash
python /path/to/backend/install_backend.py --repo /home/user/LENA-oran-flexric-smo
cd /home/user/LENA-oran-flexric-smo
./ns3 build khu-drl-xapp-conflict
```

The generated source is `scratch/drl-xapp-conflict/khu-drl-xapp-conflict.cc`.
The executable is under `build/scratch/drl-xapp-conflict/`.
Installer hashes record the exact source and adapter used.

Environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DRL_CONTROL_DIR` | absent | Enables the bridge for this fresh run directory |
| `DRL_START_S` | 180 | First decision in simulation seconds |
| `DRL_INTERVAL_S` | 30 | Decision interval in simulation seconds |
| `DRL_TIMEOUT_S` | 300 | Action wait limit in wall-clock seconds |

At each decision the scenario flushes its KPI CSVs and atomically publishes
`observation.json` with `seq`, `time_s`, `terminal`, `cell_kpi_path`, and
`parameters`. Parameters are `cio_5g`, `cio_4g1`, `cio_4g2`, `ttt_ms`, `hys_db`.
The Python controller must aggregate rows strictly earlier than `time_s`, then
atomically replace `action.json` with `{"seq": 0, "updates": {"ttt_ms": 256}}`.
An empty updates object means hold. Terminal observation needs no action.
The terminal event runs one nanosecond before stop, while reporting the intended
stop time, so it includes the final `.5`-second KPI record.

Actions are validated as one transaction before mutation: CIO in [-6,6] dB,
TTT in [40,64,128,256,512,1024] ms, and HYS in [0,1,2,3,4,6] dB.
TTT and HYS are global; all gNBs must initially have matching values.
No-op values do not trigger unnecessary RRC reconfiguration.

CIO is written and read back on **every pooled UE**, including idle slots.
The existing `SetCellIndividualOffset` cancels stale pending A3 triggers.
TTT/HYS use existing `NrGnbRrc::UpdateA3HandoverConfig`, which updates A3 config
and schedules RRC connection reconfiguration for current UEs. `receipts.jsonl`
verifies all gNB configuration values immediately. This verifies the source
configuration and scheduling; it does **not** claim every UE received its RRC
message synchronously. Per-second CSV parameter columns also read live state.

Timeout, invalid action, a conflicting controller, or inconsistent readback
fails the run instead of silently collecting invalid RL transitions. Old action
sequence files are ignored; future sequence files are rejected.

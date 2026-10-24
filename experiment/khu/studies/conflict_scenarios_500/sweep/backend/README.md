# Isolated scheduled-control backend

This target replays predetermined settings. It does not execute an xApp/DRL policy,
LLM, Near-RT RIC or Non-RT RIC, and does not measure RIC latency.

The input base is the reviewed `scratch/khu-real-dual-nr-sionna-final.cc`, SHA256
`ee9815a541cb744bab0ebf55628dc2601285f2de2e078d098e86c7b689256034`.
The installer creates `scratch/agora-conflict-sweep` and preserves the original.
It refuses a changed base, missing APIs, symlinks, unowned target files or target
edits. Identical reinstalls perform no writes. No shared NR/channel sources are
changed. Each computer must use the same reviewed ns-3 fork and dependency hashes;
compilation locally is required rather than copying a machine-specific binary.

```bash
python3 backend/install_backend.py --repo /home/user/LENA-oran-flexric-smo
python3 backend/install_backend.py --repo /home/user/LENA-oran-flexric-smo --install
cd /home/user/LENA-oran-flexric-smo
./ns3 build khu-agora-conflict-sweep -j 2
```

Binary: `build/scratch/agora-conflict-sweep/ns3.48-khu-agora-conflict-sweep-default`.
Always pass `--ns3::NrHelper::UseIdealRrc=true` and
`--ns3::NrHelper::E2ModeNr=false`. TTT 340/720 ms are research-domain values not
supported by the Real RRC encoding; this backend refuses Real RRC.

Initial eleven parameters use the existing CLI:
`--cellTxPowerDbm`, `--cellRetTiltDeg`, `--cellCioDb`, `--cellTttMs`, and
`--cellHysteresisDb`, plus common `--handoverTtt`/`--handoverHysteresis` defaults.
Cell maps use `gNB_5G:value,gNB_4G_1:value,gNB_4G_2:value`.
Names `4g1` and `4g2` refer to 1.8 GHz NR cells, not an LTE stack.

The schedule uses `--parameterSchedule=absolute/file.csv` and
`--parameterScheduleRetCacheRoot=absolute/ret_caches`:

```csv
time_s,parameter,value
180,txp_5g,30
180,txp_4g1,33
360,cio_4g2,2
540,ttt_ms,720
540,hys_db,5
720,ret_5g,12
```

Equal timestamps are one grouped event; no other simulator callback runs between
its updates. This is simulator-callback atomicity, not rollback or synchronous UE
RRC acknowledgement. The whole schedule is validated at time zero after device initialization;
duplicate keys, no-op updates, decreasing or out-of-duration times are rejected.
TTT/HYS are shared across three gNBs; changed CIO is applied to all pooled UE RRC
instances, including idle slots. A3 updates schedule connected-UE reconfiguration.
The source already exports current physical settings in per-second cell KPI rows.

Frozen domains:

- TxP per cell: 30, 33, 37, 40, 43 dBm.
- RET per cell: 3, 5, 7, 9, 12 degrees; exact caches required.
- CIO: 5G 5..9 dB; 4G1/4G2 -2..2 dB, integer steps.
- TTT: 128, 256, 340, 512, 720 ms.
- HYS: 0.5..5 dB in 0.5 dB steps.

RET caches are preloaded and validated against live cell position, band, bearing
and tilt. A change activates the exact per-cell cache and antenna angle. Never
substitute a nearby angle. Initial low-band cells with different RET angles need
the existing `ret_caches/merged_percell` cache. Keep both band cache sets, merged
caches, gNB positions and mobility traces when copying the testbed to other PCs.

Receipt `parameter_changes.csv` lives in the scenario output directory and has
`scheduled_time_s,applied_time_s,parameter,before,after,status`. Each changed field
has one row; every row in a group reports `applied` only after all eleven physical
settings match their expected values at the exact scheduled simulator time.
The scope is gNB TxP/RET, every pooled UE CIO, common gNB A3 configuration, and
scheduled RRC reconfiguration. It is not a UE-acknowledged E2 application result.

The collector uses 1-second KPI bins ending on .5-second timestamps. After a
change at 180 s, a KPI row ending at 180.5 s straddles the old/new setting; use
explicit interval overlap rules or exclude boundary bins for before/after plots.
Do not interpret the first post-change row as a fully settled 1-second outcome.

# Isolated backend for offline-trained DRL xApps

This target applies frozen xApp decisions to actual ns-3 UE/gNB RRC objects.
It uses a local file lockstep bridge, not E2/RIC transport. The original scenario,
the previous online-DRL target, AGORA, and the NR libraries remain unchanged.

Install and build inside the WSL repository:

```bash
.venv/bin/python experiments/drl_xapp_offline/backend/install_backend.py
./ns3 build khu-drl-xapp-offline
```

The generated target is `scratch/drl-xapp-offline/khu-drl-xapp-offline.cc`.
The default build executable is
`build/scratch/drl-xapp-offline/ns3.48-khu-drl-xapp-offline-default`.
The campaign uses a separately copied executable at
`experiments/drl_xapp_offline/bin/ns3-drl-offline`.
`drl_backend_manifest.json` records the source and generated-file hashes.

## Data-supported control contract

| Parameter | Accepted values |
| --- | --- |
| `cio_5g` | Integer 5, 6, 7, 8, 9 dB |
| `cio_4g1`, `cio_4g2` | Integer -2, -1, 0, 1, 2 dB |
| `ttt_ms` | 128, 256, 340, 512, 720 ms |
| `hys_db` | 0.5 through 5 dB in 0.5 dB increments |

Values are targets, not rounded/clipped requests. A whole action is validated
before mutation. TTT/HYS are common to all three gNBs. Equal values are no-ops.

**Ideal RRC is required.** The runner must explicitly pass
`--ns3::NrHelper::UseIdealRrc=true`; the generated scenario also checks the actual
helper attribute and aborts if false. Ideal RRC passes the integer TTT directly
to UE timers, including 340 and 720 ms. The Real RRC enum serializer does not
support those two values and falls through to 5120 ms; it does not round them
to 320/640 ms. Do not remove this guard or transfer these actions to Real RRC
without redefining the data/action contract.

## Lockstep transport

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `DRL_CONTROL_DIR` | absent | Fresh, unique per-run control directory; enables bridge |
| `DRL_START_S` | 180 | First decision in simulation seconds |
| `DRL_INTERVAL_S` | 30 | Decision interval in simulation seconds |
| `DRL_TIMEOUT_S` | 300 | Wall-clock wait limit for a controller response |

The bridge flushes the KPI CSV files before atomically writing
`observation.json` (`seq`, `time_s`, `terminal`, `cell_kpi_path`, `parameters`).
The five live parameters are CIO for each cell, TTT, and HYS. Fixed TxP/RET
background values can be read from completed CSV rows by the Python controller.
Observations must use only rows strictly before the decision time.

The controller atomically publishes, for example,
`{"seq":0,"updates":{"ttt_ms":340,"hys_db":1.5}}` in `action.json`.
An empty `updates` means hold. Terminal observations need no action.

CIO is set and checked on every pooled UE RRC, including idle slots. TTT/HYS
update the live gNB A3 configuration and schedule connected-UE reconfiguration.
`receipts.jsonl` reports gNB readback and scheduling, not synchronous receipt
of the RRC message by every UE. E2, AGORA control, and predetermined parameter
schedules must be disabled. Inconsistent readback, invalid actions, timeouts,
or reused control directories fail explicitly.

## Verification scope

Before the full campaign, verify actual short ns-3 runs apply CIO targets and
TTT 340/720 ms plus half-step HYS, with matching receipts and later CSV rows.
Verify a Real RRC invocation is rejected and an unchanged target does not trigger
another reconfiguration. Installer and action validation checks do not establish
learned-policy quality; the 900-second paired evaluations do that separately.

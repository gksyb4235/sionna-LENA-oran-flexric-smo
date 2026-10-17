# Frozen simulator evaluation

This folder runs **16 new ns3 simulations**, with policies already trained and
selected offline. It does not train or select checkpoints. Nothing imports or
changes `experiments/drl_xapp_conflict`.

| Held-out mobility traces | Arms per trace |
| --- | --- |
| 1419, 1420, 1421, 1422 | no xApp, LB only, Mobility only, both |

All simulations last 900 seconds with 300 UEs. The physical baseline is TxP
43 dBm and RET 9 degrees in all cells; CIO `[6, 0, -1]` dB, TTT 512 ms, HYS 1 dB.
LB activates at 180 seconds and Mobility at 540 seconds, with decisions every
30 seconds. Mobility-only therefore shares the no-control prefix until 540;
both shares the LB-only prefix until 540. The no-control arm still performs the
same observation/acknowledgement handshake.

The controller uses `selected/lb.pt` and `selected/mo.pt` under the **new**
experiment root. The copied/new simulator binary is `bin/ns3-drl-offline`. These
files and Python code, physical inputs, traces, and linked libraries are hashed
before launch and checked again. Source files must not change during evaluation.

## WSL commands

First copy only the new Python/evaluation/selected files from the Windows
workspace. The helper preserves WSL `bin/`, `backend/`, all raw outputs, and
`previous_online_stop.json`. It performs no deletion or process launch and
refuses to edit an initiated main evaluation or a running smoke test.

```bash
python3 '/mnt/c/Users/gksyb/Desktop/GNN 학습 데이터셋/experiments/drl_xapp_offline/evaluation/deploy.py' --dry-run
python3 '/mnt/c/Users/gksyb/Desktop/GNN 학습 데이터셋/experiments/drl_xapp_offline/evaluation/deploy.py'
```

Before selected checkpoints exist, `--allow-missing-selected` can prepare code
only; run the normal deployment again after offline training finishes. Every
copied artifact receives a checksum in `deployment_receipt.json`.

After preparing the WSL binary, controller, and selected offline checkpoints,
run the separate contract test:

```bash
cd /home/user/LENA-oran-flexric-smo/experiments/drl_xapp_offline
/home/user/LENA-oran-flexric-smo/.venv/bin/python evaluation/campaign.py smoke --smoke-tag contract_120s
```

This launches two 120-second simulations, `none` and `both`, using three UEs
by default (`--smoke-n-ues 300` is available). First observation and both-app
activation occur at 90 seconds, after the required complete history exists;
decisions occur at 90 and 105 seconds, followed by the terminal observation at
120 seconds. This tests handshake, frozen inference, real application/readback,
and receipt timing. Its shortened control interval and UE count are **not**
performance measurements. Output is exclusively under `smoke/<tag>/`, including
its own `plan.json`, `preflight.json`, `status.json`, and `results/`. It never
creates or modifies the main `plan.json`, `results/`, or campaign status. A
repeated tag is rejected; use a new tag when preserving evidence from a retry.

The strict simulator loader requires the trace to contain no more than the
requested number of UE IDs. Each smoke session therefore creates its own
`inputs/` CSV containing all original rows for the earliest-active three UE IDs
(or the requested smoke count). The original mobility trace stays unchanged.
The smoke plan records selected IDs, selection times, row counts, and hashes of
both original and filtered files. Only smoke plans may override a trace path;
the main 16-run plan rejects trace overrides and always uses full original inputs.

Initialize the main plan using WSL Python (Windows initialization is rejected):

```bash
cd /home/user/LENA-oran-flexric-smo/experiments/drl_xapp_offline
/home/user/LENA-oran-flexric-smo/.venv/bin/python evaluation/campaign.py init
/home/user/LENA-oran-flexric-smo/.venv/bin/python evaluation/campaign.py preflight
/home/user/LENA-oran-flexric-smo/.venv/bin/python evaluation/campaign.py launch --workers 16
```

`launch` runs in the foreground; a separately managed background session may
redirect it to a new log. It refuses duplicate starts or existing run folders.
The runner staggers worker launches by at least 8 seconds and requires 3 GiB of
reserve plus the estimated 1.5 GiB for each new simulator/controller pair. This
is a concurrency limit, not a guarantee that memory permits all 16 together.
An interruption terminates only child process groups created by this runner.

```bash
/home/user/LENA-oran-flexric-smo/.venv/bin/python evaluation/campaign.py status
/home/user/LENA-oran-flexric-smo/.venv/bin/python evaluation/campaign.py verify
/home/user/LENA-oran-flexric-smo/.venv/bin/python evaluation/report.py --root .
```

`results/<job>/` contains exact commands, ns3/controller logs, cell/UE KPI CSVs,
and control observations/actions/receipts. `execution.json` records verification,
the planned and actual simulation time of each control, and measured decision
latency. The report writes derived `analysis/metrics.json`, `windows.csv`,
`factorial.csv`, and `report.txt`; raw results stay unchanged.

## Controller contract

```text
python -m drl_offline.controller
  --control-dir PATH --mode none|lb|mo|both --load-dir ROOT/selected
  --lb-start-s 180 --mo-start-s 540 --policy eval --seed 909
```

Use the existing atomic observation/action handshake: input `observation.json`,
response `action.json={seq,updates}`, and simulator-written `receipts.jsonl`.
Action logs have `seq,time_s,terminal,updates,controller_ms,apps`, where each
active app provides `decision_ms`. No action is sent after a terminal input.

The terminal `summary.json` must include:

```json
{
  "status": "complete", "last_time_s": 900, "policy": "eval",
  "online_learning": false,
  "model_hashes_before": {"lb": "...", "mo": "..."},
  "model_hashes_after": {"lb": "...", "mo": "..."},
  "online_gradient_updates": 0, "online_replay_inserts": 0,
  "exploratory_actions": 0
}
```

Verification checks all 2,700 cell-second records, terminal observations,
receipt sequence and timing, parameter readback, cell-reported CIO/TTT/HYS,
allowed xApp activation times and parameter ownership, and frozen weights.
Changing checkpoint hashes, gradient updates, replay insertion, or exploration
fails the run. Receipt acceptance alone does not count as application success.

## Comparison

The primary intervals are 540–720 and 720–900 seconds; the report also includes
all five 180-second intervals. It compares Jain, ping-pong/HO attempts, RLF,
RLF per 1,000 UE-seconds, HO failures, Goodput, and delay at the same trace/time.
Rates use ratios of summed event counts, and PP ratios are undefined below
10 denominator events. Jain is computed from the three interval PRB means;
the mean of per-second Jain values is reported separately.

The factorial interaction is `both - LB - Mobility + none`. Prefix equality
must hold before controls diverge. This is a small, paired, descriptive test:
the interaction sign alone does not prove harmful conflict, statistical
significance, or policy convergence. Assess whether each xApp improves its
own metric alone and whether combining them worsens either app's metric or
violates a declared guard constraint. Post-control random-number consumption
can diverge despite matched starting seeds and mobility traces.

## Preserve a failed main attempt before retrying

Do not deploy new files over a failed attempt before recording its original
code. Run the standalone archive helper **from the mounted Windows source**
while the WSL experiment still contains the deployed version that failed:

```bash
python3 '/mnt/c/Users/gksyb/Desktop/GNN 학습 데이터셋/experiments/drl_xapp_offline/evaluation/archive_failed.py' --attempt attempt001 --dry-run
python3 '/mnt/c/Users/gksyb/Desktop/GNN 학습 데이터셋/experiments/drl_xapp_offline/evaluation/archive_failed.py' --attempt attempt001
```

The helper requires a terminal `failed`/`interrupted` campaign, checks recorded
PIDs against `/proc` identity/start times, and scans for unrecorded owned
simulator/controller/runner processes. It refuses live or ambiguous ownership;
PID reuse is recorded instead of signalling the unrelated process. No process
is stopped by this helper. Wait for all debugging reproductions to finish too.

It first copies the **existing WSL** Python/evaluation source, binary,
checkpoints, backend source, README, and available training/data-audit reports
to `failed_attempts/attempt001/snapshot/`. It then moves only these explicit
runtime entries into the attempt directory: `results/`, `job_specs/`,
`plan.json`, `preflight.json`, `campaign_status.json`, `campaign_console.log`,
`background_launch.json`, optional `completion_verification.json`, and a dead
`.evaluation-launch.lock`. The active source/binary/checkpoints remain in place;
previous smoke runs and previous-online-stop records are preserved.

`archive_manifest.json` records input/output hashes, source consistency against
the old preflight, original status, inventories of every moved artifact, and
two process checks. It preserves the old status byte-for-byte, including any
stale displayed running IDs. Large mobility/cache data and shared libraries
are not copied; the preserved preflight identifies their hashes. Nothing is
deleted. An incomplete archive retains its staging folder and lock for manual
inspection; deployment and new runner operations refuse that lock.

Only after a complete archive, deploy fixes, initialize a new main plan in
WSL, and start the new attempt. The archive frees the exact runtime paths that
otherwise correctly reject duplicate starts. New terminal campaign states
clear `running_ids` and recompute completed, failed, and unstarted counts from
final worker records after all owned children exit.

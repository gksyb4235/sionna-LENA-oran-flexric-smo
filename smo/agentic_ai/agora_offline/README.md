# AGORA offline control workbench

This module runs the frozen final Temporal GNN and a five-role agent workflow against recorded observations. After an explicit operator selection, its live bridge can send the reviewed plan through the RIC gateway and FlexRIC/E2 to ns3. Offline queue registration remains available separately. See the [repository execution guide](../../../README.md) for launching services, binding a simulator run and checking application/restoration receipts. The Windows GNN overview at port8773 remains available. The retired Agent Ops stack is preserved in the local legacy archive described in smo/README.md.

## Start

From the repository in WSL:

```bash
cd /home/user/LENA-oran-flexric-smo/smo/agentic_ai
../../.venv/bin/python -m agora_offline.launch
```

Open **http://127.0.0.1:18081**. The launcher uses an existing process if its health endpoint identifies this application. It writes `.state/server.log` and `.state/server.json`. For foreground operation use `../../.venv/bin/python -m agora_offline.server`.

The server binds to127.0.0.1 and uses the repository virtual environment. Dependencies are listed in requirements.txt. Credentials are read server-side from the independent agentic_ai `.env` and environment variables, with environment variables taking precedence. `OPENAI_MODEL=openai:gpt-...` is normalized to the API model name; `AGORA_MODEL` can override it. Credentials are never returned to the UI or exported. Model name and five role prompts are editable in settings. A session takes an immutable settings snapshot.

**LLM mode sends the entered prompt, selected observation summaries, model predictions and retrieved DAG evidence to the configured OpenAI API.** Deterministic mode makes no LLM calls and is labelled as a rule-based demonstration. Its results verify the pipeline, not LLM planning ability. After explicit user approval, a live gpt-5.4-mini run completed all five roles, the original forecast, Analyze, DAG lookup and batched alternatives. The provider uses Responses API with stateless encrypted continuation kept only in server memory; the UI shows public text and actual tool evidence.

## Reproduce a useful case

Choose test run `gnn_0130`, issue time180s, horizon180s. Enter:

> 앞으로 3분 동안 Cell5G의 TxP를43dBm에서33dBm으로 줄여 에너지를 절감하려고 합니다. 원안부터 확인하고, 문제가 있으면 시간·강도·대상 셀 대안을 비교해 주세요.

The demonstration first evaluates exactly `txp_5g=33`, then searches the declared time/intensity/cell grid when needed. Under the current threshold-v3-app5 defaults this example has no feasible candidate in the searched grid. Earlier archived tests selected `txp_5g=40` under a different, more permissive policy; those results must not be treated as passing the new thresholds. Both are model forecasts, not measured control benefits. Historical records such as `tests/demo_cases.json` retain their original policy.

Five declared applications are available: Energy Saving lowers TxP; Throughput Maximization adjusts TxP/RET; Coverage Optimization adjusts RET; Load Balancing adjusts CIO; Mobility Optimization adjusts shared TTT/HYS. RET changes tilt, not beam width. Directional benefit must be predicted, not assumed. Both parsers preserve compound explicit transitions, including TTT and HYS together. Energy Saving and Coverage are rApps; the other three are xApps. Their common contract and GNN inputs are unchanged. Application purpose determines live routing.

## Model and replay contract

- Frozen v16 checkpoint SHA256: `76513c944f4de8e2d94fb8fbe3f6a50b72b51432b38c4b0e9beb76ceede1dbf8`.
- 153,537 parameters; 46 graph nodes: 5 application types, 5 parameter types, 3 cells and 33 cell/KPI nodes. Cell and KPI GRUs encode past seconds before two heterogeneous graph blocks. Per-cell numeric RCP values are edge features. App edges encode static declared capabilities; a requesting app's identity and numeric goals are not GNN inputs.
- One checkpoint directly predicts both 90s and 180s intervals. The 180s forecast is not an average of two 90s forecasts. Three seeds and two ablations were trained; the checkpoint was selected using validation data only.
-Up to180s of1Hz past observations, masks, eleven current controls, eleven candidate controls, absolute issue time and application delay are supplied to the GNN. Inputs end strictly before the issue time. At`t=180`, input intervals are`[0,180)`.
-Candidate output covers`[t+delay,t+delay+horizon)`. Supported delays are0/30/60/90s; supported origins are30..720s and the action start must be at most720s. Observations are sampled every second, supervised training origins every30s and evaluation origins every10s. Off-evaluation-grid origins are marked unvalidated temporal interpolation.
- 1,000 selected runs are available locally: 720 train / 180 validation / 100 test. Original run IDs and renumbered Final_Dataset IDs are kept distinct in the manifest. Replay NPZ files are excluded from Git; tracked manifests retain their SHA-256 hashes. Use scripts/agora_replay.py to verify, export or import both required replay variants when moving this checkout. Checkpoints and model contracts remain tracked.
- Eleven learned outputs per cell: goodput, PRB, BLER, mean RSRP, mean SINR, HO attempts, ping-pongs, RLF, P10 RSRP, successful HO and received-packet delay above 100ms. Counters are interval sums. Mean RSRP and P10 pool valid serving-cell UE-time samples over the full interval; this mean definition differs from v15. Missing RSRP is masked and does not quantify all-UE outage. Mean minus P10 is a derived spread. Power, network energy efficiency, PRB balance and mobility ratios are calculated from predictions. Mean delay, loss and UE share remain diagnostic observations only; the history chart has 17 measured fields plus UE count.
-No future recorded KPI, alternative-run future, SHAP value or DAG edge is a GNN input. Offline source files contain future observations, but the runtime slices only the past for inference. Future values are used only in separate validation scripts.

TxP is accepted within30..43dBm.36/41dBm and other non-grid values are explicitly marked unvalidated interpolation. Other controls default to the trained grids: RET3/5/7/9/12; CIO5g5..9, CIO4g1/4g2−2..2; TTT128/256/340/512/720ms; HYS0.5..5dB in0.5dB steps. The live gateway independently checks simulator control domains and the bound run. The canonical launcher uses per-cell RET caches and Ideal RRC, preserving the learned 340/720ms TTT values; Real RRC must satisfy its encodable TTT enumeration. Model acceptance alone does not certify simulator applicability. Historical labels4g1/4g2 identify1.8GHz NR cells in these data, not LTE cells.

## Agent and estimator interaction

1. **Intent Agent** translates the user request into unchanged requested controls, duration, delay, goals and constraints. If a stated current value differs from the replay configuration, the run fails visibly instead of changing the baseline silently.
2. **Conflict Agent** evaluates exactly one requested plan. The estimator also computes an internal keep forecast for the same future window. The UI counts this as one request candidate and separately records the keep reference.
3. **KPI Analyst** calls `Analyze` and may retrieve `lookup_context`. Numeric constraint diagnostics, signed coordinate-reset sensitivities and scoped DAG evidence are shared with the mitigation planner. Public decision summaries are streamed; private reasoning is not requested or exposed.
4. **Mitigation Agent** uses a bounded ReAct loop. First it evaluates the full single-control magnitude/cell/delay grid, including every weaker integer-dBm TxP value when interpolation is enabled. Results return best candidates per changed-control group plus numerical Analyze evidence. For a single TxP request, a secondary batch evaluates two/three-cell allocations around the agent-selected evaluated pivot, even if a single-cell candidate is already feasible. Joint controls preserve direction and RIC domain; the sum of absolute dB changes cannot exceed the original request magnitude (a control-distance bound, not equivalent energy savings). Unrelated control families remain prohibited. The API permits128 candidates per batch, four alternative rounds and513 total candidates including the original. Unsearched candidates and budget exhaustion remain explicit. All tool results return to the agent before the next call; a deterministic coverage guard can finish required coverage if an agent returns no valid tools.
5. **Deployment Agent** summarizes the server-selected result and prepares a JSON validation plan. “Recommended” means predicted feasible and goal-met; a feasible plan with an unmet goal is marked partial. When none is feasible, only diagnostic candidates are returned and no executable recommendation is selected.

The deterministic estimator implements the draft Eq.9 order: **hard feasibility first, goal achievement U descending, requested-intent deviation D ascending**. U is the weighted sum of saturated directional progress ratios, upper-clipped at1 but not lower-clipped at0. D includes attenuation, same-RIC relocation and normalized delay. An LLM cannot override the predictions or numeric ordering. Only the best1–2feasible candidates, or up to2infeasible diagnostics, are returned to the planner; every evaluated row stays in the UI audit. The tool distinguishes a violated guard from a mere optimization-goal shortfall.

The threshold-v3-app5 editor separates purpose, multiple weighted improvement goals, editable basic constraints and additional constraints. Active constraints are absolute thresholds, including derived ratios. Defaults are goodput retention >=95%, total RLF <=2/90s or <=4/180s, pooled ping-pongs/successful-HO <=train Q75 (13.1387%/90s; 12.9826%/180s), and worst-cell P10 RSRP >=-100dBm. `context/threshold_reference_v16.json` records training-only recommendations: 720 runs, issue times 30..720s every 10s and overlapping 90/180s windows. Changed window sampling explains why the 180s RLF recommendation changed from 3 to 4. These are research defaults, not SLAs or independent samples. Old threshold-v2 policies retain their explicit values when reopened.

Presets use network energy efficiency +5%, goodput +5%, PRB imbalance (1-Jain) -20%, ping-pong/successful-HO -20%, or coverage P10 +3dB together with mean RSRP +1dB. Coverage also limits mean-minus-P10 spread. Throughput adds a BLER ceiling; balancing adds maximum cell-average PRB <=80%. Users can add, edit, disable or remove constraints, set multiple goals/weights and choose cell subsets. Horizon changes preserve manually edited values. The immutable editor specification governs both deterministic and LLM modes and cannot be overwritten by an agent. Historical unsupported prediction metrics produce insufficient evidence rather than passing silently.

The estimator pools numerators/denominators for derived ratios. Goodput retention uses the same future window's keep forecast, so moving traffic to another cell is not itself a network-goodput violation. Power follows the simulator's ON-state formula `130 + 4.7 * 10**((TxP_dBm-30)/10)`W/cell; energy is power times horizon, and network efficiency is total goodput divided by total power. Raw runtime outputs remain recorded separately. Undefined ratios, fewer than10predicted successful handovers and percentage goals with zero reference are insufficient evidence, never automatic passes. Mean RSRP/SINR do not establish UE outage coverage. Loss cohort alignment and mean-delay limitations remain explicit; no invented universal absolute limits are supplied. Derived metrics do not retrain or alter the frozen GNN.

The default margin multiplier is 0: these are uncalibrated point checks. v16 does not reuse v15 calibration. Auxiliary RLF probabilities refer specifically to network counts >2/90s and >3/180s, are uncalibrated diagnostics, and do not replace the editable constraint thresholds. Nonzero margins require suitable validation calibration. Test errors are never reused as calibration.

`lookup_context` preserves the original DAG evidence scope: fixed configuration, baseline controls, one tested value and900s observations. A switched90/180s request normally receives an out-of-scope flag. Such evidence may inform search hypotheses; it cannot certify the new intervention. Coordinate-reset model sensitivities are not SHAP values or identified causal effects.

Before the Mitigation Agent's first LLM call, `prepare_mitigation_context` ensures that broad DAG context for the original and permitted relocated controls is available, even if the Analyst retrieved only one KPI. The exact delivered context is audited with source event numbers, relation counts and a hash. The UI exposes that context and the three axes' evaluated/planned counts. Shared TTT/HYS controls mark cell relocation as not applicable; cells already at the supported bound report why relocation is blocked. DAG context is never fed into the GNN.

The October13 search regression reproduced an earlier incomplete33/37dBm search. `tests/test_search.py` ensures40/41dBm and cell alternatives are evaluated even when a mocked LLM repeats narrow proposals, and continues combinations under the new thresholds without relaxing them. Archived live search reports refer to the earlier policy. Tests also cover blocked-cell explanations, all control families and explicit budget exhaustion.

## APIs and audit

The final operator panel presents up to three distinct, actually evaluated candidates. Feasible candidates retain goal-achievement/deviation ordering. If none is feasible, fully evaluable candidates are ordered by `sum(max(violation,0)/max(abs(effective_threshold),1))`, then violated-constraint count, goal achievement and deviation. Unlike the per-batch diagnostic ranking, this shortlist prioritizes total violation severity and fixes the normalization to thresholds across delayed candidates. This is an unweighted dimensionless comparison score, not a conflict probability or a calibrated risk. Unknown evidence ranks last and cannot be queued. Every prediction, original threshold, remaining violation, unmet goal and interpolation warning remains available. No threshold is relaxed; `selected` remains reserved for feasible model recommendations, while `operator_review.candidates` are explicitly operator choices. The Deployment Agent receives all three with their numerical evidence and must explain fallback alternatives rather than ending at "none feasible". Old completed sessions reconstruct this panel from saved candidate events without rerunning inference.

An operator selects one card. Its intent identifies Energy Saving/Coverage rApps or Throughput/Load Balancing/Mobility xApps. Live routes are fixed by app role, and control ownership is checked before queue registration. Custom intents remain available for offline evaluation. TxP/RET alone do not imply a Non-RT route.

Non-RT requests now go to non-rt-ric on port9000: persistent A1 policy, shared rApp handler, execution xApp gateway, FlexRIC/E2, ns3. The gateway checks the stored policy before forwarding. xApp requests go directly to the Near-RT execution gateway. The old local-A1 submission endpoint on the gateway is disabled; historical readbacks remain readable. Offline queue registration and its historical split option do not send controls.

Both paths retain model provenance, review hash, operator choice, changed controls and restoration schedule. Duplicate selections are idempotent; incompatible selections for an existing receipt are rejected. Simulation time is never treated as wall-clock time. Live status distinguishes policy storage, xApp delivery and independent ns3 scheduling/application/restoration readbacks. A successful HTTP response is not evidence of parameter application or KPI improvement. Only changed values are restored, with conflict detection before restoration.

| Endpoint | Purpose |
|---|---|
|`GET /api/status`, `/api/runs`, `/api/snapshot`|Model contract and past-only replay state|
|`POST /api/predict/batch`|Raw final-GNN batch forecasts: run_id, issue_time_s, plans with full parameters/delay_s/horizon_s|
|`POST /api/estimate/batch`|Forecasts, guard checks and U/D ordering: run_id, issue_time_s, intent, requested_plan, candidates|
|`GET/PUT /api/settings`|Five prompts, research policy and model name; no key disclosure|
|`GET /api/intent-policy?horizon_s=90\|180`|Metric options, purpose presets and train-derived threshold recommendations|
|`POST /api/sessions`|Start a prompt with run_id, issue_time_s, mode, horizon_s and optional structured intent_spec (or legacy intent_override)|
|`GET /api/sessions/{id}/events`|SSE public summaries, tool requests/results and decisions; Last-Event-ID reconnection|
|`GET /api/sessions/{id}`|Current session status, result and event record|
|`POST /api/sessions/{id}/cancel`|Stop further orchestration|
|`GET /api/sessions/{id}/export`|Download result, schedule, prompts, model provenance and evidence|
|`POST /api/sessions/{id}/deploy`|Register the operator's candidate_id, target_ric, review_sha256 and accept_constraint_violations in the offline outbox|
|`GET /api/sessions/{id}/deployment`|Download the saved operator selection and exact per-RIC apply/restore schedules|
|`GET /api/deployments?ric=near_rt\|non_rt`|List the latest200 saved queue items, optionally filtered by RIC; no network delivery|
|`GET /api/live/runs`|List simulator runs and their live control readiness|
|`POST /api/sessions/{id}/deploy-live`|Bind the reviewed candidate to a simulator run and submit the exact schedule through the RIC gateway|
|`GET /api/sessions/{id}/deployment-live`|Read independent live scheduling, application and restoration status|

Each session records `.state/sessions/{id}/events.jsonl` incrementally and `session.json` on completion. Exports include the start/end schedule and a planned return to pre-request settings at the duration end; the effect of restoration is not forecast. Offline records do not execute a schedule; an explicitly selected live deployment schedules it in the ns3 controller through E2. Completed sessions can be reopened after restart using `/?session=<id>` without executing the agents or GNN again.

## Verification and remaining experiments

```bash
cd /home/user/LENA-oran-flexric-smo/smo/agentic_ai
../../.venv/bin/python -m unittest discover -s agora_offline/tests -v
../../.venv/bin/python -m unittest agora_offline.tests.test_v16
```

Portable inference is checked against 136 frozen GPU forecasts across five origins, both horizons and supported delays, with tolerance 0.001 training standard deviations for CPU/CUDA kernel differences. The measured Windows CPU maximum difference is 0.000221 training standard deviations. Tests also mutate future replay, verify pooled P10, ratio denominators, all five apps, compound mobility, immutable thresholds, finite search coverage, operator fallback and queue-only deployment. LLM protocol tests use explicit test doubles; historical live LLM records describe v15 and do not validate v16 planning quality.

These checks establish implementation correctness, model portability and one live LLM workflow's operability. A single successful LLM run does not establish planning quality across intents, online switched-action accuracy or actual energy savings. Separate RIC integration tests check actual E2 application and restoration; these are control-path tests, not performance evaluations. Later ns3 evaluation should replay the same seed/mobility/traffic until the issue time, branch the unchanged/requested/selected plans, measure post-action KPI guards and objectives, and compare LLM+GNN, GNN-only search and context ablations under the same candidate/query budget. The current fixed-run replay must not be presented as that branched experiment.

OpenAI function calling and streaming follow the [official function-calling guide](https://developers.openai.com/api/docs/guides/function-calling), [Responses streaming guide](https://developers.openai.com/api/docs/guides/streaming-responses) and [stateless reasoning continuation guide](https://developers.openai.com/api/docs/guides/reasoning).
# LangSmith observation

Set these values in the existing server-only `smo/agentic_ai/.env` file, which is
excluded from Git. The application accepts `LANGSMITH_TRACING_V2` as an alias;
`LANGSMITH_TRACING` takes precedence when both are present.

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=<server-side-key>
LANGSMITH_PROJECT=AGORA
```

Install `agora_offline/requirements.txt` into the repository virtual environment
and restart the UI service. `/api/status` reports public tracing configuration.
Each session records a root trace ID in its `tracing` field. The LangSmith tree
contains role-agent calls, public LLM inputs/outputs and usage, GNN batch
evaluation, Analyze, DAG lookup, and the final recommendation. Opaque provider
continuation/reasoning items and credentials are omitted. Tracing failures do
not change the numerical decision or trigger deployment. Only an explicit
operator action can initiate live RIC control.

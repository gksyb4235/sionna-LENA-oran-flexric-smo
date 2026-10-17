# Offline-trained DRL xApps and frozen ns3 evaluation

This isolated experiment uses the original GNN dataset's **training partition
only** to pretrain two small, independent Double DQN policies. It then evaluates
their frozen decisions in real ns3 simulations. Existing GNN/AGORA models and
`experiments/drl_xapp_conflict` remain separate.

## What is learned

The 720 training runs hold all 11 parameters fixed for 900 seconds. They contain
no observed action-change transitions. Consequently this is **model-based offline
DRL**, not direct offline Q-learning on real `(s, a, r, s_next)` samples.

1. Map original run IDs to the renumbered raw folders and verify parameters.
2. Build nonoverlapping 30-second KPI bins, preserving raw sums.
3. Split these 720 runs by complete run into 576 fitting and 144 internal
   validation/selection runs. The original 180 validation and 100 test runs are
   not used. Scalers are fitted on the 576 fitting runs only.
4. Fit three bootstrapped response MLPs: 11 parameters plus time, aggregate
   generated traffic and connected-UE profile -> per-cell PRB, UE and Goodput;
   network HO attempts/successes/failures, ping-pongs, RLF and delay.
5. Generate 48,000 synthetic transitions per xApp: 16,000 observed starting
   contexts, three model-generated steps each. No synthetic outcome is labelled
   an actual ns3 observation. Penalize ensemble reward disagreement and distance
   from fitting configurations.
6. Train three Double DQN seeds per xApp, 6,000 optimizer updates each; choose
   the best of the three using only the 144-run internal synthetic evaluation.
7. Freeze the chosen checkpoints before new ns3 evaluation. Runtime has no
   optimizer, replay insertion or exploratory action.

The conditional response assumes an immediate response after a parameter change;
the fixed-parameter data cannot establish transient behavior. Synthetic rollouts
replay the future aggregate load/UE profile as an environment driver. Connected
UE and generated-load statistics are not strictly independent of control, so this
is an explicit approximation, not a causal identification claim. The actor sees
only past/current observations, current parameters and current simulation time.

## Actor and actions

Each policy is a 60 -> 64 -> 64 -> actions MLP with ReLU. The state contains
metrics over the latest 30 and 90 seconds, the five controlled parameters,
simulation time, and six background TxP/RET values. This policy is separate from
the Temporal GNN used by AGORA.

| xApp | Objective | Allowed actions |
| --- | --- | --- |
| Load Balancing | Improve PRB Jain index | Hold, or change one cell's CIO: 16 actions |
| Mobility Optimization | Reduce ping-pongs per HO attempt | Hold, or choose TTT/HYS jointly: 51 actions |

CIO levels: 5g `[5,6,7,8,9]`; other cells `[-2,-1,0,1,2]` dB.
TTT: `[128,256,340,512,720]` ms. HYS: 0.5 through 5 dB, step 0.5.
These match the collected data, rather than inventing six levels for every
parameter. Equivalent no-op actions are masked; hold remains available.

Both objectives include soft RLF, HO-failure, delivery and change penalties.
They are **not hard threshold guarantees**. In particular fewer ping-pongs can
come with fewer HOs and more RLF; report attempts, successes, RLF and Goodput
alongside the ratio. The response model's sparse RLF and delay predictions are
weak, and ensemble agreement does not fix this limitation.

## Real evaluation: 16 runs

Four unseen mobility traces (1419--1422) each run four matched arms:
no xApp, LB alone, Mobility alone, both. Each run lasts 900 simulation seconds
with 300 UEs. TxP is 43 dBm and RET is 9 degrees in every cell. Initial
CIO is `[6,0,-1]`, TTT is 512 ms and HYS is 1 dB.

- LB activates at 180 seconds; Mobility activates at 540 seconds.
- Controllers observe and decide every 30 seconds.
- With both active, both independent actors use the same pre-action observation.
- ns3 blocks at the decision point until the matching action arrives, then
  records actual application and parameter readback. This experiment uses the
  isolated local controller bridge, not an E2/RIC deployment.
- `UseIdealRrc=true` is explicit: the collected nonstandard TTT values 340/720 ms
  require this path. Real-RRC ASN.1 encoding does not preserve these values.
- Compare identical time windows and traces. Before controls diverge, prefixes
  must match. Compare standalone improvement first, then concurrent degradation
  and factorial interaction. Neither an interaction alone nor one favorable
  trajectory proves conflict or general policy quality.

This is a first paired evaluation with four traces, not a statistical convergence
claim. Do not select new checkpoints using these test outcomes. If results lead
to redesign, freeze a new version and use fresh test traces.

## Files and execution

- `data_audit/`: train-only manifest, raw mapping audit and prepared arrays.
- `python/drl_offline/`: common contracts, response model, trainer and controller.
- `training/`: fitting metrics, candidate seeds, selection and provenance.
- `selected/lb.pt`, `selected/mo.pt`: frozen runtime checkpoints.
- `backend/`: isolated ns3 bridge installer and build verification.
- `evaluation/`: deployment, smoke test, 16-run campaign and paired reporting.
- `development_archive/`: preserved initial development pass; never deployed.

Train on Windows from this workspace with `PYTHONPATH` set to this experiment's
`python` directory, using `.venv/Scripts/python.exe -m drl_offline.train`.
The trainer refuses to overwrite selected checkpoints.

Deployment and Linux execution commands are in [evaluation/README.md](evaluation/README.md).
Live results reside at
`/home/user/LENA-oran-flexric-smo/experiments/drl_xapp_offline/results/`.
The runner automatically verifies completed runs and produces paired metrics.

The earlier eight online-learning simulations were deliberately stopped with
their partial data/checkpoints retained; they are not this frozen evaluation.

The first 16-run launch was halted after one no-control seed1421 process received
SIGILL at about 19.5 seconds, before any policy action. Its outputs and executable
snapshot are retained in WSL `failed_attempts/attempt001/`. An isolated rerun with
the same 900-second configuration and bridge passed 30.5 seconds; all 60 earlier
cell rows and 79 columns matched exactly. No cause or simulator-code fix has been
established. See `backend/debug_reproduction_summary.json` and the native-crash
audit for evidence. A subsequent attempt must use the same frozen policies and
be reported separately; failed partial runs are not performance results.

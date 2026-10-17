# Isolated DRL xApp controller

Dependencies: Python 3.10+, NumPy, PyTorch. CPU, one PyTorch thread per controller.
No pandas, OpenAI, GNN surrogate, or existing AGORA controller is used.

Run from this directory, or add this directory to `PYTHONPATH`:

```bash
python -m drl_xapps.controller --control-dir /absolute/run/control \
  --mode both --policy train --reward local --seed 42 \
  --lb-start-s 180 --mo-start-s 180 --checkpoint-dir /absolute/chain/checkpoints
python -m unittest discover -s tests -v
```

Each episode must have its own control directory. Resume training via
`--load-dir /previous/checkpoints`; full replay, optimizer, exploration counter,
and RNG state are retained. Frozen evaluation uses `--policy eval`, with either
one `--load-dir` or independently selected `--lb-load /file/lb.pt` and
`--mo-load /file/mo.pt`. Eval requires a checkpoint with actual gradient updates;
it cannot silently treat a randomly initialized network as a learned policy.

For staged coexistence use `--lb-start-s 180 --mo-start-s 540`. Inactive apps
neither act nor collect transitions. `--mode none|lb|mo|both` supports paired
factorial evaluation. `--policy heuristic` is an explicitly non-learning baseline.

## File protocol

The simulator atomically writes `observation.json`:

```json
{"seq":0,"time_s":180,"terminal":false,"cell_kpi_path":"/run/cell_kpi.csv",
 "parameters":{"cio_5g":6,"cio_4g1":0,"cio_4g2":-1,"ttt_ms":512,"hys_db":1}}
```

The controller atomically replies with `action.json`:

```json
{"seq":0,"updates":{"cio_4g1":2,"ttt_ms":256,"hys_db":2}}
```

The simulator blocks simulation time while waiting, applies updates, and reports
actual parameter readback in the next observation. The controller checks that
previously requested values were applied. Terminal observation consumes the
last measured reward and saves a checkpoint; no terminal action is written.
No simulation steps are taken by this Python module itself.

## Action levels and ownership

- Load Balancing: hold plus one selected cell's CIO delta ±1, ±2, ±3, ±4,
  ±5, or ±6 dB: 37 catalog entries. Executed CIO is bounded to [-6,6] dB.
  Clipped no-ops and duplicate resulting vectors are masked before selection
  and Double-DQN bootstrapping. CIO affects target-cell selection preference,
  not transmit power.
- Mobility Optimization: hold plus Cartesian product of six TTT values
  [40,64,128,256,512,1024] ms and six HYS values [0,1,2,3,4,6] dB: 37 entries.
  Larger TTT/HYS suppress or delay handovers; they are not guaranteed to improve
  all mobility metrics. TTT/HYS are applied consistently to all cells.
- Intensities are physical control levels, independent of reward weights.
- Both actors use the identical pre-action observation. They update disjoint
  parameter keys; coexistence is independent learning, not centralized MARL.

## State, reward, learning, and honest interpretation

The 54-element state combines completed 30s/90s cell KPI histories, actual CIO,
TTT/HYS, and current simulation time. Per-window features include three cells'
PRB/UE/goodput, network Jain, HO and ping-pong counts, smoothed ping-pong ratio,
HO sample support and failure rate, RLF counts and UE-exposure rate, delivery-to-generated ratio,
packet-weighted mean delay, traffic and UE load. Future/current-unfinished rows
are excluded; missing cells and discontinuous timestamps are errors. Delivery
ratio is an interval traffic accounting diagnostic, not a cohort packet-loss rate.

The LB reward values Jain level and improvement. MO values reduced ping-pong/HO,
with a 0.1-rate, 10-attempt weak smoothing prior and sample-support scaling;
zero HO gets no positive mobility reward. Successful-HO support gates mobility
rewards, and HO failures receive a separate penalty (local .5, guarded 2 times
the sample-support-weighted failure ratio), preventing failed-attempt denominator
inflation from receiving a perfect mobility score. Both variants penalize RLF, low
delivery, and control movement. `guarded` increases these penalties and adds the
other app's objective as a soft constraint. These are research reward choices,
not proven operator SLAs. See `rewards.py` for the complete coefficients.
Rewards are computed from actual observations in the interval following the
selected action, including the terminal interval. A 90s history helps state
estimation but is not used to leak pre-action outcomes into a 30s action reward.

Each actor is a 54→64→64→37 ReLU MLP (10,085 parameters), using Double DQN,
target network, Adam, Huber loss, gradient clipping, and bounded replay. Default
gamma=.97, batch=16, warmup=16 actual transitions, four gradient steps/transition,
hard target sync every 100 gradient steps. Epsilon decreases linearly from .9
to .1 over 1,000 transitions. A 900s episode with first action at180s yields only
24 transitions; a few episodes are a pilot, not evidence of convergence.

`actions.jsonl` records every observation, actual parameters, chosen action,
exploration flag, reward attribution, reward terms, gradient losses, and latency.
`summary.json` records local/total training steps and explicitly declines to
claim convergence. `controller_error.json` records fail-closed errors. Checkpoints
are trusted local PyTorch objects; do not load untrusted checkpoint files.

Demonstrate policy improvement on held-out mobility traces with exploration and
learning disabled, paired against hold and heuristic baselines. Demonstrate
conflict using the same trace and initial settings in none/LB/MO/both arms, with
MO activation at540s. Training rewards and a favorable single run do not prove
generalization or a causal conflict.

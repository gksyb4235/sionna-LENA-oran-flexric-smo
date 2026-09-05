# Local SMO AI/ML stack

Start all local processes from the repository root:

```bash
pip install -e 'smo/aimlfw[agentic]'
PYTHONPATH=. .venv/bin/python -m smo.aimlfw.local_stack start
```

In another terminal, inspect component readiness:

```bash
PYTHONPATH=. .venv/bin/python -m smo.aimlfw.local_stack check
```

Before the first GNN is trained, the expected overall state is `degraded`:
Inference Service and KPI Advisor report `not_ready`; this does not prevent
feature extraction or model training.

Bootstrap the first model from an ns-3 result directory:

```bash
PYTHONPATH=. .venv/bin/python -m smo.aimlfw.local_workflow bootstrap-model \
  scenarios/results/<result-directory>
```

After the model is loaded, an advisor request can be evaluated, persisted as
an Evidence Record, and its first recommendation explicitly approved for A1
publication:

```bash
PYTHONPATH=. .venv/bin/python -m smo.aimlfw.local_workflow advise-and-publish \
  advisor-request.json --approved-by <operator-id>
```

The request JSON is the body of KPI Advisor's `POST /invoke`. A1 publication
requires a recommendation containing all three fixed cells. The command
returns the persisted Evidence ID and the A1 policy instance/status. Stop the
stack with `python -m smo.aimlfw.local_stack stop` or `Ctrl-C` in the start
terminal.

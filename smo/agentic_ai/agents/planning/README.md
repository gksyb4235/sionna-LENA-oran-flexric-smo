# Planning Agent

The Planning Agent receives Decomposition Agent output JSON and creates one LangGraph-style sub-graph plan for each subtask.

The planner now uses a MongoDB Knowledge Registry plus the local registry seed files to decide which registered agents and tools can be composed into a sub-graph. It must not invent unregistered agents/tools.

All files for this agent live under:

```text
/home/ubuntu/jaechan/agents/planning
```

## Input Contract

The input is the Decomposition Agent JSON:

```json
{
  "intent": "original user intent",
  "subtasks": ["first subtask", "second subtask"],
  "golden_goal_context_used": false
}
```

## Output Contract

The HTTP API and `--json` CLI mode return:

```json
{
  "intent": "original user intent",
  "subtask_plans": [],
  "overall_status": "completed | pending_approval | partial | blocked | rejected"
}
```

Each subtask plan includes:

- `subtask_id`
- `subtask`
- `langgraph_spec`
- `representative_agent`
- `attempts`
- `status`
- `result`
- `knowledge_context_used`

## Knowledge Registry

Seed documents live in:

```text
/home/ubuntu/jaechan/agents/planning/registry
```

- `registry/agents.json`: registered agents, descriptions, capabilities, and limitations.
- `registry/tools.json`: registered tools, operations, descriptions, and limitations.

MongoDB collections:

```text
KNOWLEDGE_MONGODB_URI=mongodb://...
KNOWLEDGE_DATABASE=knowledge
KNOWLEDGE_AGENTS_COLLECTION=agents
KNOWLEDGE_TOOLS_COLLECTION=tools
```

Seed or update MongoDB after editing registry JSON:

```bash
cd /home/ubuntu/jaechan/agents/planning
.venv/bin/python tools/knowledge_registry_admin.py --json
```

Dry-run without MongoDB:

```bash
.venv/bin/python tools/knowledge_registry_admin.py --dry-run --json
```

Only `status: "active"` registry entries are selected by default. Use `status: "planned"` for future agents/tools that should be documented but not planned into executable subgraphs yet.

## Planning Behavior

For each subtask:

1. Query Knowledge DB for semantically similar registered agents/tools.
2. Pass the Knowledge DB candidates and registered local runtime context into the Planning LLM.
3. Require the LLM subgraph to use only registered agents/tools.
4. Route registered local runtime subgraphs to the adapter:
   - `probe-agent` + `grafana-prometheus-tool`
   - `representative-core-agent` + `core-kubectl-command-tool`
5. Return `blocked` instead of inventing placeholder agents/tools when no registered combination is available.

## CLI

By default, the CLI wraps the Planning Agent result with `------planning-agent------` and `----end----` markers. Use an input file:

```bash
cd /home/ubuntu/jaechan/agents/planning
.venv/bin/python agent.py --input decomposition.json
```

Or stdin:

```bash
cat decomposition.json | .venv/bin/python agent.py
```

Use machine-readable JSON without section markers:

```bash
.venv/bin/python agent.py --json --input decomposition.json
```

## HTTP

Run the server:

```bash
cd /home/ubuntu/jaechan/agents/planning
.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8081
```

Endpoints:

- `GET /health`
- `POST /invoke`
- `POST /feedback` with a Monitoring Agent `replan_required` event and
  `Authorization: Bearer $MONITORING_AGENT_TOKEN`
- `POST /approve`

## Instruction Files

- `agent.md` defines planning responsibilities and orchestration behavior.
- `skills/planning-tools/SKILL.md` defines how to use Knowledge DB and runtime tools.

## Tests

```bash
cd /home/ubuntu/jaechan/agents/planning
.venv/bin/python -m unittest discover -s tests
```

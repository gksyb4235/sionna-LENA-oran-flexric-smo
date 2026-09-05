# Decomposition Agent

This is the first agent in the multi-agent system. It receives a natural language intent and decomposes it into a simple list of subtasks that a future planning agent can consume.

All files for this agent live under:

```text
/home/ubuntu/jaechan/agents/decomposition
```

The DeepAgents source checkout used by this project lives under:

```text
/home/ubuntu/jaechan/deepagents
```

## Output Contract

The CLI prints one JSON object with exactly these fields:

```json
{
  "intent": "original user intent",
  "subtasks": ["first subtask", "second subtask"],
  "golden_goal_context_used": false
}
```

- `intent`: the original natural language intent.
- `subtasks`: a list of subtask strings.
- `golden_goal_context_used`: legacy field; `true` only when the Decomposition Knowledge DB returned an active matched pattern.

## Instruction Files

- `agent.md` defines the agent role, behavior, and output rules.
- `skills/decomposition-tools/SKILL.md` defines tool-use instructions in the DeepAgents skill format.

`agent.py` reads both through DeepAgents at runtime:

- `agent.md` is loaded into the system prompt.
- `/skills` is passed to `create_deep_agent(..., skills=["/skills"])`.

## Environment

Create `.env` from `.env.example` and set:

```text
OPENAI_API_KEY=...
OPENAI_MODEL=openai:gpt-5.4-mini
DECOMPOSITION_KNOWLEDGE_MONGODB_URI=mongodb://decomposition_reader:...@127.0.0.1:27017/decomposition?authSource=decomposition
DECOMPOSITION_KNOWLEDGE_DATABASE=decomposition
DECOMPOSITION_KNOWLEDGE_COLLECTION=decomposition_patterns
DECOMPOSITION_KNOWLEDGE_TIMEOUT_MS=1500
```

The runtime account is read-only. Pattern registration uses the separate MongoDB provisioning script and admin credentials. If the URI is empty, unreachable, or no active text-matched pattern exists, the agent still decomposes the intent and returns `golden_goal_context_used=false`.

## Runtime Notes

DeepAgents requires Python 3.11 or newer. Keep runtime artifacts inside `/home/ubuntu/jaechan`, for example:

```bash
export UV_CACHE_DIR=/home/ubuntu/jaechan/.cache/uv
export UV_PYTHON_INSTALL_DIR=/home/ubuntu/jaechan/.local/share/uv/python
export UV_TOOL_DIR=/home/ubuntu/jaechan/.local/share/uv/tools
```

## Usage

The default CLI starts from the Decomposition Agent, then automatically hands the decomposition JSON to the Planning Agent and prints each agent result as a marked section.

```bash
cd /home/ubuntu/jaechan/agents/decomposition
.venv/bin/python agent.py "Monitor each network NF"
```

Use `--decompose-only` only when you want to inspect the three-field decomposition contract by itself. Use `--json` when another script needs only the final JSON payload without CLI section markers.

```bash
cd /home/ubuntu/jaechan/agents/decomposition
.venv/bin/python agent.py --decompose-only "Monitor each network NF"
```

## Tests

The contract tests use only the standard library and can run before DeepAgents dependencies are installed:

```bash
cd /home/ubuntu/jaechan/agents/decomposition
python3 -m unittest discover -s tests
```

## Decomposition To Planning Handoff

Run the default decomposition-to-planning flow:

```bash
cd /home/ubuntu/jaechan/agents/decomposition
.venv/bin/python agent.py "Monitor each network NF"
```

Run decomposition only:

```bash
cd /home/ubuntu/jaechan/agents/decomposition
.venv/bin/python agent.py --decompose-only "Monitor each network NF"
```

`--plan` is still accepted, but it is now equivalent to the default behavior. `--json` suppresses the section markers and prints only the final JSON payload.

When `PLANNING_AGENT_URL` is empty, the handoff adapter calls the local Planning Agent CLI at `/home/ubuntu/jaechan/agents/planning`. When `PLANNING_AGENT_URL` is set, it posts the decomposition JSON to `POST /invoke` on that URL.

HTTP endpoints:

- `POST /invoke`: returns decomposition JSON only.
- `POST /invoke-and-plan`: runs decomposition, hands off to Planning Agent, and returns the planning report.

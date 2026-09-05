---
name: planning-tools
description: Operational guidance for Knowledge DB lookup, LangGraph-style sub-graph design, and mock execution.
---

# Planning Tools

## Tool Boundary

This skill is about planning-time tool operation. Planning Agent identity, output contract, and orchestration policy are defined in `agent.md`.

## Knowledge DB Lookup

Use Knowledge DB to discover available lower-level agents and tools before designing a sub-graph.

Default registry collections:

- agents: `knowledge.agents`
- tools: `knowledge.tools`

Lookup behavior:

1. Query candidates using the current subtask text.
2. Treat candidates as approximate semantic matches, not exact matches.
3. Prefer candidates with similar capabilities, tags, descriptions, names, or strong `_match_score` values.
4. Use at most five agents and five tools in v1.
5. If Knowledge DB is unavailable, empty, or below the similarity threshold, use only registered local runtime agents/tools when they match; otherwise return a blocked plan. Never infer placeholder planning nodes.

## LangGraph-Style SubGraph Design

For each subtask, use the Planning Agent LLM to create a declarative graph spec with:

- `nodes`: agent and tool nodes.
- `edges`: handoff or tool-use transitions.
- `entrypoint`: the first representative agent node.
- `terminal_nodes`: completion-check nodes.
- `representative_agent`: the node the Planning Agent communicates with.

Do not compile the graph in v1. The Execution Agent will own compile/runtime behavior later.

## Mock Execution

The Execution Agent is not available yet.

In v1:

- Use registered runtime adapters for supported subtasks.
- Return `blocked` when no registered agent/tool combination exists for a subtask.
- Preserve the planned graph in the report so the future Execution Agent can consume it.

## Retry Policy

The production retry limit is controlled by `PLANNING_MAX_ATTEMPTS`, default `3`.

In v1, perform exactly one mock attempt per subtask.

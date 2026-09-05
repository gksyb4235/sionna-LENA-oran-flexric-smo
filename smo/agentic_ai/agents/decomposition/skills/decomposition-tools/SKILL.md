---
name: decomposition-tools
description: Guidance for using preloaded Decomposition Knowledge DB patterns.
---

# Decomposition Tools

## Knowledge Boundary

This skill is only about tool operation. Agent identity, decomposition quality rules, and the final JSON output contract are defined in `agent.md`, not here.

The runtime retrieves active text-matched decomposition patterns before invoking the model. The model does not query MongoDB directly.

## Use Returned Patterns

- Apply every required semantic and decomposition rule from each provided active pattern.
- Preserve pattern-required targets, numbers, durations, conditions, and agent notification roles.
- Do not reveal database details or internal pattern metadata in the final response.
- Do not invent pattern facts when no matching pattern was provided.
- Follow the `monitoring-decomposition` skill for monitoring and maintained-state requests.

## Failure Handling

MongoDB configuration, connection, empty collection, and missing text-index failures remain non-blocking for unrelated intents.

- Continue with decomposition using only the original intent.
- Let `agent.md` govern the final output field values.

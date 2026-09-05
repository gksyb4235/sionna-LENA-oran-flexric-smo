---
name: core-kubectl-tools
description: Use this skill for free5gc Core NF kubectl action requests, including get, describe, label, annotate, patch, delete, restart, scale, exec, attach, cp, apply, replace, edit, node, and namespace operations.
---

# Core kubectl Tool Guidance

## Tool Boundary

This skill governs how to create safe kubectl command proposals for the Representative Core Agent. The LLM only emits structured JSON. It never emits arbitrary shell commands and never executes commands.

## Proposal Rules

- Always use the JSON command proposal schema from `agent.md`.
- Use `namespace=free5gc-v4` for free5gc Core NF work unless the user explicitly asks for a namespace operation.
- Use Core NF names in `nfs`; do not invent pod suffixes.
- Put operation-specific values in `parameters`.

## Risk and Approval

All proposals require approval. The safety layer will classify risk.

- Read-only operations such as get, describe, and logs are low risk.
- Metadata or spec-changing operations such as label, annotate, patch, and non-zero scale are medium risk.
- Delete, pod reset, rollout restart, scale to zero, exec, attach, cp, apply, replace, edit, node operations, and namespace deletion or modification are high risk.

High-risk operations should still be proposed when they match the user intent. They become `pending_approval`; do not self-censor them only because they are risky.

## Failure Handling

If the target NF, operation, or required parameters are unclear, still produce the closest structured proposal only when it is safe and explicit enough. Otherwise allow validation to block the request.

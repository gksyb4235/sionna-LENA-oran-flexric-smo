# AGORA SMO

The active entry point is `python scripts/agora.py up` from the repository root.

| Path | Responsibility |
|---|---|
| agentic_ai/agora_offline/ | Final frozen GNN, five reasoning agents, goal/constraint editor and operator UI |
| agentic_ai/agora_offline/applications.py | One shared contract for two rApps and three xApps |
| agentic_ai/.env | Private provider settings; ignored by Git |
| non_rt_ric/ | Durable A1 policies and rApp dispatch, hosted by non-rt-ric |
| ric_control/ | Execution xApp dispatch, FlexRIC/E2 binding and ns3 readbacks |

Energy Saving and Coverage Optimization are rApps. Throughput Maximization, Load Balancing and Mobility Optimization are xApps. Roles select the deployment route and do not alter GNN weights, graph nodes or forecasting inputs. All five use the same application contract and reviewed-plan shape.

```text
Operator-reviewed Energy / Coverage plan
  -> Non-RT RIC (non-rt-ric, port 9000)
  -> Energy Saving / Coverage rApp handler
  -> execution xApp gateway (port 18082)
  -> FlexRIC Near-RT RIC -> E2 -> ns3

Operator-reviewed Throughput / Load Balancing / Mobility plan
  -> xApp handler -> execution xApp gateway
  -> FlexRIC Near-RT RIC -> E2 -> ns3
```

The application handlers share a process and implementation. They are logical app instances, not five independently deployed daemons. The common C execution xApp is the actuator. Roles are chosen from intent purpose; TxP/RET alone do not determine a route. Custom/mixed intents remain evaluable offline but require a declared app with matching control ownership for live deployment.

The standalone mediator retains its generic policy API and adds fixed AGORA policy type 401 with a nested Pydantic schema, persistent SQLite policy instances, rApp validation, delivery and status lookup. The execution gateway verifies the exact stored policy before forwarding a rApp request. Updating an already accepted policy id to a different intervention is rejected; evaluate and select a new plan instead. Deletion does not cancel an ns3 schedule.

This is a local research Non-RT implementation using the existing mediator, not a claim of full O-RAN A1 conformance. E2 style 401/action 1/parameter 33000 remain the local control extension. Policy storage, execution acceptance, actual application, restoration and KPI benefit are separate facts.

## Provider configuration

Copy `agentic_ai/.env.example` to `agentic_ai/.env` when setting up a new machine, and supply the provider key there or through environment variables. The current machine's existing settings have already been migrated without changing their effective values. No legacy agents folder is read. Secrets are not committed or served to the browser.

## Archived components

The old agents, dashboard, agent_ops, MongoDB initialization files, deepagents source checkout, aimlfw services and their tests were removed from the active tree. Their complete local trees, environments and saved states are preserved under `experiments/agora_runtime/legacy/`. The exact paths and original Git commit are in `legacy/latest.json`. Historical design notes under .kiro describe the retired framework.

No new service starts the retired stack. To recover it, use the archive manifest to copy the preserved trees back to their original paths, checking for conflicts first. The currently used GNN replay data, UI sessions, checkpoints, RIC configuration, RET caches and monitoring data stay in place.

See [AGORA.md](../AGORA.md) for simulation, export/import and validation commands.

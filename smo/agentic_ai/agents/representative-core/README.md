# Representative Core Agent

Creates kubectl command proposals for free5gc Core NF actions and returns `pending_approval` reports for the Planning Agent.

Default behavior does not execute kubectl:

```bash
.venv/bin/python agent.py --json "restart AMF"
```

Explicit approval executes the validated kubectl argv rebuilt from the proposal:

```bash
.venv/bin/python agent.py --approve --input pending-report.json
```

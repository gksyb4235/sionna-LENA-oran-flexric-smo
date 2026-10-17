# CPU container verification

Local Docker Desktop/WSL, Ubuntu 24.04 amd64 runtime, no GPU device, no network,
read-only root, non-root UID 1000; 2 CPU quota and 12 GiB memory limit.

- Actual ns3 smoke: 20 simulated seconds, 8 scheduled groups, 20 applied field changes,
  60 cell KPI rows; all passed. Binary SHA256:
  `3d6b4b351f4e300c0fb5e159a775ec98d86ea1c3bfe4b72cce0ebc756a24e83c`.
- All 500 scenarios passed input/cache metadata/library ABI preflight.
- Input archive: 531 regular files independently matched their SHA256 manifest.
- CPU quota 2 was detected; request for 3 concurrent single-CPU jobs was rejected.
- Sweep/resource tests: 16 passed; repository entry-point tests: 3 passed.
- Native vs container output comparison: {"cell_kpi.csv": true, "ue_kpi.csv": true, "parameter_changes.csv": true}.

This does not validate 900-second stability, peak memory, server permissions,
or scientific conflict effectiveness. Run an initial small batch on the server
before increasing concurrency. Detailed local report: `/home/user/agora-cpu-export-v1/verification.json`.

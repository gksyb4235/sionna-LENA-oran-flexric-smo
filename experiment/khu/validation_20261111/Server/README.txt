Server: 84 jobs, 28 simultaneous one-core workers, 3 full cycles.
Requires the previously supplied agora-rq3:cpu-v1 Docker image. No rebuild is needed.
Run preflight.sh, then run.sh. The container runs detached.
Logs: docker logs -f agora-validation-server
Override RQ3_CACHE_SOURCE and VALIDATION_OUTPUT if your server paths differ.

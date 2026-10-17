FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3 python3-venv libpython3.12 libgsl27 libgtk-3-0t64 libxml2 libsqlite3-0 util-linux ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/agora
COPY runtime/ /opt/agora/
RUN python3 -m venv .venv && mkdir -p results scenarios/khu-real/ret_caches scenarios/khu-real/ue_position && chmod -R a+rX /opt/agora
ENV LD_LIBRARY_PATH=/opt/agora/build/lib PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
ENTRYPOINT ["python3", "/opt/agora/deploy/conflict-sweep/entrypoint.py"]
CMD ["--help"]

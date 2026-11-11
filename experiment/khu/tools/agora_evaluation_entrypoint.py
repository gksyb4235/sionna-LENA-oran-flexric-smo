"""Use the previously frozen Docker runtime with the canonical read-only bundle."""
import importlib.util
import os
from pathlib import Path
import runpy
import sys

spec = importlib.util.spec_from_file_location('frozen_container', Path('/opt/agora/deploy/rq3/entrypoint.py'))
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
if '--status' not in sys.argv and '--dry-run' not in sys.argv:
    runtime.verify_runtime_and_caches()
os.environ.update(AGORA_BUNDLE='/bundle', AGORA_REPO='/opt/agora', AGORA_OUTPUT='/results')
runpy.run_path('/bundle/experiment/khu/tools/agora_evaluation_sweep.py', run_name='__main__')

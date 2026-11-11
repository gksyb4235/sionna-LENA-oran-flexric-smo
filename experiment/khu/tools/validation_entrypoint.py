"""Reuse the immutable RQ3 image with a read-only Git checkout of new plans."""
import importlib.util, os, runpy, sys
from pathlib import Path

if __name__=='__main__':
    spec=importlib.util.spec_from_file_location('frozen_container',Path('/opt/agora/deploy/rq3/entrypoint.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    if sys.argv[1:]==['--resources']:
        import json
        result=m.resources();result['image_scope']='Frozen runtime with all-method Git manifest';print(json.dumps(result,indent=2))
    else:
        if '--dry-run' not in sys.argv and '--status' not in sys.argv:m.verify_runtime_and_caches()
        os.environ.update(VALIDATION_BUNDLE='/bundle',VALIDATION_REPO='/opt/agora',VALIDATION_OUTPUT='/results')
        runpy.run_path('/bundle/experiment/khu/tools/validation_sweep.py',run_name='__main__')

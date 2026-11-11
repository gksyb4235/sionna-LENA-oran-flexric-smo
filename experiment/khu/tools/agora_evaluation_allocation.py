"""Read a machine allocation without changing the frozen replay manifest."""
import hashlib
import json
from pathlib import Path

HOSTS = ('pc1', 'pc2', 'pc3', 'pc4', 'server')
DATA = Path('experiment/khu/agora_evaluation_v2')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_allocation(root, manifest=None):
    root = Path(root)
    manifest_path = root / DATA / 'manifest.json'
    if manifest is None:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    path = root / DATA / 'allocation.json'
    if not path.exists():
        return dict(partitions=manifest['partitions'],
                    workers=manifest.get('allocation_model', {}).get('workers', {}),
                    allocation_source='manifest', allocation_sha256=None)
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if data.get('schema') != 'agora-machine-allocation-v1':
        raise ValueError('Unknown allocation schema')
    if data.get('manifest_sha256') != sha(manifest_path):
        raise ValueError('Allocation belongs to a different replay manifest')
    parts, workers = data['partitions'], data['workers']
    if set(parts) != set(HOSTS) or set(workers) != set(HOSTS):
        raise ValueError('Allocation must specify all five machines')
    if any(type(n) is not int or n < 1 for n in workers.values()):
        raise ValueError('Allocation workers must be positive integers')
    if any(not isinstance(ids, list) or any(not isinstance(j, str) for j in ids)
           for ids in parts.values()):
        raise ValueError('Invalid allocation job list')
    actual = [j for host in HOSTS for j in parts[host]]
    expected = [j for host in HOSTS for j in manifest['partitions'][host]]
    if len(actual) != len(set(actual)) or len(expected) != len(set(expected)):
        raise ValueError('Duplicate job in machine allocation')
    if set(actual) != set(expected):
        raise ValueError('Allocation omitted, added or reserved a replay job')
    return dict(data, allocation_source=str(path), allocation_sha256=sha(path))

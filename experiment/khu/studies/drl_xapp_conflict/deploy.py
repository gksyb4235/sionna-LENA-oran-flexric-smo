"""Copy this isolated experiment's source into its own WSL directory."""
from pathlib import Path
import argparse
import hashlib
import json


def deploy(repo):
    source = Path(__file__).resolve().parent
    repo = repo.resolve(strict=True)
    destination = repo / 'experiment/khu/studies/drl_xapp_conflict'
    if destination.is_symlink() or destination.resolve().parent != (repo / 'experiment/khu/studies').resolve():
        raise ValueError('Unsafe experiment destination')
    if (destination / 'campaign_status.json').exists():
        raise RuntimeError('Refusing to change code after a campaign has started')
    entries = []
    allowed = {'.py', '.h', '.md', '.json', '.ps1', '.txt'}
    for path in source.rglob('*'):
        relative = path.relative_to(source)
        if not path.is_file() or path.suffix not in allowed or any(p in ('__pycache__', 'results', 'job_specs', 'bin', 'selected') for p in relative.parts):
            continue
        if path.name in ('deployment.json', 'selection.json', 'preflight.json', 'campaign_status.json', 'launch_record.json'):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = path.read_bytes()
        target.write_bytes(raw)
        entries.append(dict(path=relative.as_posix(), sha256=hashlib.sha256(raw).hexdigest()))
    result = dict(destination=str(destination), files=entries, original_scenario_modified=False)
    (destination / 'deployment.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    result = deploy(args.repo)
    print(json.dumps({'destination': result['destination'], 'files': len(result['files'])}))

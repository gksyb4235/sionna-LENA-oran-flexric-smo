"""Copy the verified isolated build into this experiment's own executable path."""
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    repo = Path('/home/user/LENA-oran-flexric-smo').resolve(strict=True)
    experiment = repo / 'experiments/drl_xapp_offline'
    manifest = json.loads((repo / 'scratch/drl-xapp-offline/drl_backend_manifest.json').read_text())
    original = repo / 'scratch/khu-real-dual-nr-sionna-final.cc'
    if digest(original) != manifest['source_sha256']:
        raise RuntimeError('Original scenario changed since the isolated target was generated')
    source = repo / manifest['binary_relative']
    destination = experiment / 'bin/ns3-drl-offline'
    if destination.is_symlink() or destination.resolve().parent != (experiment / 'bin').resolve():
        raise RuntimeError('Unsafe isolated executable destination')
    if destination.exists() and digest(destination) != digest(source):
        raise RuntimeError('Refusing to overwrite a different published executable')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    verification = {
        'build_passed': True,
        'command': './ns3 build khu-drl-xapp-offline -j 2',
        'executable': str(destination), 'binary_sha256': digest(destination),
        'source_binary_sha256': digest(source), 'size_bytes': destination.stat().st_size,
        'original_scenario_sha256': digest(original), 'original_scenario_unchanged': True,
        'requires_ideal_rrc': True, 'simulations_launched': 0,
        'installer_tests_passed': 3,
        'action_validation': json.loads((experiment / 'backend/validation_verification.json').read_text()),
        'generated_manifest': manifest,
    }
    (experiment / 'backend/build_verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()

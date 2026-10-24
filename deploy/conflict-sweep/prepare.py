"""Create an allowlisted runtime Docker context and separate input tar (Linux)."""
import argparse, hashlib, json, platform, re, shutil, subprocess, tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BUNDLE=Path('experiment/khu/studies/conflict_scenarios_500/sweep')
CONTAINER_BUNDLE=Path('experiments/conflict_scenarios_500/sweep')
DEPLOY=Path('deploy/conflict-sweep')
INPUT_PREFIX=Path('scenarios/khu-real')
DATA_FILES={'candidate_review.json','conflict_scenario_params_reviewed_500.csv',
            'event_screening_1500.csv','execution_manifest.json','pilot_ids.json'}

def container_path(path):
    """Map canonical host paths to the stable CPU-v1 image layout."""
    path=Path(path)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Package paths must be repository-relative: {path}')
    for host,wire in ((BUNDLE,CONTAINER_BUNDLE),
                      (Path('experiment/khu/datasets'),INPUT_PREFIX),
                      (Path('experiment/khu/caches'),INPUT_PREFIX)):
        if path.is_relative_to(host): return wire/path.relative_to(host)
    return path

def allowed_bundle_file(relative):
    """Runtime code and frozen data only; no local configs or results."""
    if len(relative.parts)==1:
        return relative.suffix=='.py' or relative.name in {
            'config.example.json','README_ko.txt','REVIEW_ko.txt','.gitignore'}
    if len(relative.parts)==2 and relative.parts[0]=='data':
        return relative.name in DATA_FILES
    return (len(relative.parts)==2 and relative.parts[0] in ('backend','tests')
            and relative.suffix in ('.py','.h'))

def runtime_paths(root):
    # Query both layouts: before commit the index still contains old paths.
    tracked=subprocess.check_output(['git','ls-files','-z',str(BUNDLE),
        str(CONTAINER_BUNDLE),str(DEPLOY)],cwd=root).decode().split('\0')
    candidates={Path(path) for path in tracked if path}
    for folder in (root/BUNDLE,root/CONTAINER_BUNDLE,root/DEPLOY):
        candidates.update(path.relative_to(root) for path in folder.glob('*.py'))
    selected={}
    for relative in sorted(candidates):
        source=root/relative
        if relative.is_relative_to(CONTAINER_BUNDLE) and (root/BUNDLE).is_dir():
            source=root/BUNDLE/relative.relative_to(CONTAINER_BUNDLE)
        wire=container_path(relative)
        allowed=(wire.is_relative_to(CONTAINER_BUNDLE)
                 and allowed_bundle_file(wire.relative_to(CONTAINER_BUNDLE)))
        if wire.is_relative_to(DEPLOY):
            allowed=(len(wire.relative_to(DEPLOY).parts)==1
                     and (wire.suffix in ('.py','.sh','.md') or wire.name=='Dockerfile'))
        if allowed and source.is_file():
            if not source.resolve().is_relative_to(root.resolve()):
                raise ValueError(f'Runtime input escapes repository: {source}')
            selected[wire]=source
    return selected

def input_paths(root,inputs):
    """Keep immutable manifest paths and tar members in the legacy wire layout."""
    mapped={}
    for relative in sorted(inputs):
        wire=container_path(relative)
        if not wire.is_relative_to(INPUT_PREFIX):
            raise ValueError(f'Unexpected input outside KHU datasets/caches: {relative}')
        source=root/relative
        if not source.is_file():
            raise FileNotFoundError(f'Run scripts/conflict_sweep.py setup --build first: {relative}')
        source=source.resolve()
        if not source.is_relative_to(root.resolve()):
            raise ValueError(f'Dataset/cache escapes repository: {relative}')
        if wire in mapped and source!=mapped[wire]:
            raise ValueError(f'Inputs collide in the container: {wire}')
        mapped[wire]=source
    return mapped

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if platform.machine()!='x86_64': p.error('This binary bundle requires Linux x86_64')
    out=args.output.resolve()
    if out.exists(): p.error('Use a new output directory; existing exports are never overwritten')
    host_bundle=ROOT/BUNDLE if (ROOT/BUNDLE).is_dir() else ROOT/CONTAINER_BUNDLE
    config=json.loads((host_bundle/'config.example.json').read_text())
    manifest=json.loads((host_bundle/'data/execution_manifest.json').read_text())
    binary=ROOT/config['binary']
    linked=subprocess.check_output(['ldd',str(binary)],text=True)
    if 'not found' in linked: raise RuntimeError(linked)
    native=[Path(m) for m in re.findall(r'=> (/\S+)',linked) if Path(m).is_relative_to(ROOT)]
    inputs={Path(j['trace_path']) for j in manifest['jobs']}
    inputs.add(Path(config['gnb_positions']))
    cache=Path(config['ret_cache_root'])
    for a in (3,5,7,9,12):
        for band in ('1p8ghz','3p5ghz'):
            inputs.add(cache/f'tilt_{a}deg/sionna_rt_cache_{band}_2x2_1x1_lzf.h5')
        for b in (3,5,7,9,12):
            if a!=b: inputs.add(cache/f'merged_percell/merged_1p8ghz_4g1_{a}deg_4g2_{b}deg.h5')
    inputs=input_paths(ROOT,inputs)
    runtime=out/'context/runtime';runtime.mkdir(parents=True)
    copied={}
    def copy(source,rel):
        rel=container_path(rel)
        dest=runtime/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest);copied[str(rel)]=digest(dest)
    for rel in [Path(config['binary']),*[Path(s) for s in config['source_files']],Path('scratch/khu-real-dual-nr-sionna-final.cc')]:
        copy(ROOT/rel,rel)
    for src in native: copy(src,src.relative_to(ROOT))
    # Only tracked allowlisted sweep code/data, never local configs/results/secrets.
    for rel,source in runtime_paths(ROOT).items(): copy(source,rel)
    site=ROOT/'.venv/lib/python3.12/site-packages'
    for pattern in ('numpy','numpy.libs','numpy-*.dist-info','h5py','h5py.libs','h5py-*.dist-info'):
        matches=list(site.glob(pattern))
        if not matches: raise FileNotFoundError(f'Missing Python dependency: {pattern}')
        for src in matches:
            for file in src.rglob('*'):
                if file.is_file() and '__pycache__' not in file.parts and file.suffix!='.pyc':
                    copy(file,Path('.venv/lib/python3.12/site-packages')/file.relative_to(site))
    config.update(ns3_root='/opt/agora',output_root='results')
    for key in ('binary','ret_cache_root','gnb_positions'):
        config[key]=str(container_path(config[key]))
    config['source_files']=[str(container_path(path)) for path in config['source_files']]
    (runtime/CONTAINER_BUNDLE/'config.container.json').write_text(json.dumps(config,indent=2)+'\n')
    input_hashes={str(rel):digest(source) for rel,source in inputs.items()}
    print('Packing separate cache and mobility inputs...',flush=True)
    with tarfile.open(out/'inputs.tar','w',dereference=True) as tar:
        for rel,source in inputs.items(): tar.add(source,arcname=str(rel),recursive=False)
    report=dict(format='agora-cpu-runtime-v1',architecture='linux/amd64',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        binary_sha256=digest(binary),runtime_files=copied,input_files=input_hashes,
        scope='Existing native build packaged; not a clean source rebuild; cached CPU sweep only')
    for dest in (out/'bundle_manifest.json',runtime/'bundle_manifest.json'):
        dest.write_text(json.dumps(report,indent=2)+'\n')
    shutil.copy2(ROOT/'deploy/conflict-sweep/Dockerfile',out/'context/Dockerfile')
    print(json.dumps(dict(output=str(out),input_files=len(inputs),runtime_files=len(copied)),indent=2))
if __name__=='__main__':main()

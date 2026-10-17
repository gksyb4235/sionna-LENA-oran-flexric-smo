"""Create an allowlisted runtime Docker context and separate input tar (Linux)."""
import argparse, hashlib, json, platform, re, shutil, subprocess, tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BUNDLE=Path('experiments/conflict_scenarios_500/sweep')
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
    config=json.loads((ROOT/BUNDLE/'config.example.json').read_text())
    manifest=json.loads((ROOT/BUNDLE/'data/execution_manifest.json').read_text())
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
    for rel in inputs:
        if not (ROOT/rel).is_file(): raise FileNotFoundError(f'Run scripts/conflict_sweep.py setup --build first: {rel}')
    runtime=out/'context/runtime';runtime.mkdir(parents=True)
    copied={}
    def copy(source,rel):
        dest=runtime/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest);copied[str(rel)]=digest(dest)
    for rel in [Path(config['binary']),*[Path(s) for s in config['source_files']],Path('scratch/khu-real-dual-nr-sionna-final.cc')]:
        copy(ROOT/rel,rel)
    for src in native: copy(src,src.relative_to(ROOT))
    # Only tracked allowlisted sweep code/data, never local configs/results/secrets.
    tracked=subprocess.check_output(['git','ls-files','-z',str(BUNDLE),'deploy/conflict-sweep'],cwd=ROOT).decode().split('\0')
    paths={Path(x) for x in tracked if x}
    paths.update(x.relative_to(ROOT) for x in (ROOT/'deploy/conflict-sweep').glob('*.py'))
    paths.update(x.relative_to(ROOT) for x in (ROOT/BUNDLE).glob('*.py'))
    for rel in sorted(paths):
        if (ROOT/rel).is_file(): copy(ROOT/rel,rel)
    site=ROOT/'.venv/lib/python3.12/site-packages'
    for pattern in ('numpy','numpy.libs','numpy-*.dist-info','h5py','h5py.libs','h5py-*.dist-info'):
        matches=list(site.glob(pattern))
        if not matches: raise FileNotFoundError(f'Missing Python dependency: {pattern}')
        for src in matches:
            for file in src.rglob('*'):
                if file.is_file() and '__pycache__' not in file.parts and file.suffix!='.pyc':
                    copy(file,Path('.venv/lib/python3.12/site-packages')/file.relative_to(site))
    config.update(ns3_root='/opt/agora',output_root='results')
    (runtime/BUNDLE/'config.container.json').write_text(json.dumps(config,indent=2)+'\n')
    input_hashes={str(rel):digest(ROOT/rel) for rel in sorted(inputs)}
    print('Packing separate cache and mobility inputs...',flush=True)
    with tarfile.open(out/'inputs.tar','w') as tar:
        for rel in sorted(inputs): tar.add(ROOT/rel,arcname=str(rel),recursive=False)
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

#!/usr/bin/env python3
"""Copy/link verified completed replays; index incomplete attempts separately.

No existing output is moved, removed or overwritten. Imported completion is not
a claim of prefix parity or observed intent satisfaction; those require analysis.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def fingerprint(job):
    keys = ('job_id', 'candidate_id', 'initial_parameters', 'events', 't0_s',
            'evaluation_end_exclusive_s', 'trace_sha256', 'historical_csv_sha256')
    return hashlib.sha256(json.dumps({k: job[k] for k in keys}, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--label', required=True)
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--mode', choices=['copy', 'link', 'index'], default='copy')
    args = parser.parse_args()
    if not args.label or any(x not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for x in args.label):
        raise ValueError('Use a simple source label')
    source, destination, bundle = args.source.resolve(), args.destination.resolve(), args.bundle.resolve()
    if destination == source or destination.is_relative_to(source):
        raise ValueError('Import destination must be outside the source tree')
    manifest_bytes = (bundle / 'experiment/khu/agora_evaluation_v2/manifest.json').read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    jobs = {j['job_id']: j for j in manifest['jobs']}
    spec = importlib.util.spec_from_file_location('frozen', bundle / 'experiment/khu/tools/agora_frozen_sweep.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    target = destination / args.label / stamp
    target.mkdir(parents=True, exist_ok=False)
    (target / 'manifest_snapshot.json').write_bytes(manifest_bytes)
    rows = []
    for receipt in sorted(source.glob('rq3_*/execution.json')):
        folder = receipt.parent
        record = read(receipt)
        receipt_hash = sha(receipt)
        row = dict(job_id=folder.name, source=str(folder), source_receipt_sha256=receipt_hash,
                   recorded_status=record.get('status'), imported_complete=False,
                   prefix_parity_verified=False, intent_metrics_computed=False)
        job = jobs.get(folder.name)
        if not job:
            row['reason'] = 'Unknown ID; not imported'
        elif record.get('status') != 'complete':
            row['reason'] = 'Incomplete/failed attempt indexed only; raw outputs remain at source'
            write(target / 'incomplete_metadata' / (folder.name + '.json'), record)
        else:
            checks = runner.verify_output(bundle, job, folder, record.get('returncode'))
            output_hashes = checks.get('output_sha256', {})
            recorded = record.get('verification', {}).get('output_sha256', {})
            hashes_match = bool(output_hashes) and output_hashes == recorded
            provenance_match = (record.get('job_id') == job['job_id'] and
                                record.get('provenance', {}).get('backend_source_sha256') == manifest['backend_source_sha256'])
            source_stable = sha(receipt) == receipt_hash
            row.update(verification=checks, recorded_output_hashes_match=hashes_match,
                       frozen_backend_source_hashes_match=provenance_match,
                       source_stable=source_stable, job_fingerprint=fingerprint(job))
            if checks.get('execution_verified') and hashes_match and source_stable and provenance_match:
                if args.mode == 'copy':
                    copied = target / folder.name
                    shutil.copytree(folder, copied)
                    for relative, expected in output_hashes.items():
                        if sha(copied / relative) != expected:
                            raise ValueError('Copied output hash mismatch: ' + str(copied / relative))
                        if sha(folder / relative) != expected:
                            raise ValueError('Source output changed while importing: ' + str(folder / relative))
                    if sha(copied / 'execution.json') != receipt_hash or sha(receipt) != receipt_hash:
                        raise ValueError('Source receipt changed while importing')
                    row['organized_path'] = str(copied)
                elif args.mode == 'link':
                    linked = target / folder.name
                    linked.symlink_to(folder, target_is_directory=True)
                    row['organized_path'] = str(linked)
                else:
                    row['organized_path'] = str(folder)
                row['imported_complete'] = True
                row['reason'] = 'Execution and recorded output hashes verified; paired attribution still requires prefix analysis'
            else:
                row['reason'] = 'Verification failed; source preserved and not promoted to completed archive'
        rows.append(row)
    report = dict(schema='agora-observation-import-v1', source=str(source), destination=str(target),
                  source_label=args.label, mode=args.mode, manifest_sha256=manifest_sha256,
                  imported_at_utc=datetime.now(timezone.utc).isoformat(), attempts=len(rows),
                  completed_verified=sum(row['imported_complete'] for row in rows), rows=rows,
                  original_outputs_unchanged=True, no_best_attempt_selection=True)
    write(target / 'import_index.json', report)
    print(json.dumps({k: report[k] for k in ('destination', 'attempts', 'completed_verified', 'mode')}, indent=2))


if __name__ == '__main__':
    main()

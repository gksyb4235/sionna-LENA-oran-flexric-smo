"""Merge verified shard reports without copying or overwriting raw simulation output."""
import argparse
import json
from pathlib import Path
from sweep import canonical_hash, read_json, validate_manifest


def combine(manifest: dict, report_paths: list[Path]) -> dict:
    expected = {job["job_id"] for job in manifest["jobs"]}
    rows, seen, shared_inputs, data_review = [], set(), {}, []
    fingerprint = canonical_hash(manifest)
    for path in report_paths:
        report = read_json(path)
        if report.get("manifest_hash") != fingerprint:
            raise ValueError(f"Manifest mismatch: {path}")
        shared = report.get("shared_input_sha256", {})
        if report.get("jobs") and not report.get("input_provenance_available"):
            data_review.append(f"Missing frozen source/cache/trace provenance: {path}")
        if report.get("input_provenance_available") and not shared:
            raise ValueError(f"Provenance flag set but no shared input hashes: {path}")
        for name, digest in shared.items():
            if name in shared_inputs and shared_inputs[name] != digest:
                raise ValueError(f"Cross-PC source/cache/trace input mismatch: {name} ({path})")
            if report.get("frozen_inputs", {}).get(name) != digest:
                raise ValueError(f"Shared input differs from frozen record: {name} ({path})")
            shared_inputs[name] = digest
        for row in report["jobs"]:
            ident = row["job_id"]
            if ident not in expected or ident in seen:
                raise ValueError(f"Unknown or duplicate scenario {ident}: {path}")
            seen.add(ident)
            rows.append({**row, "source_report": str(path)})
    return {"experiment_id": manifest["experiment_id"], "manifest_hash": fingerprint,
            "reported_jobs": len(rows), "verified_complete": sum(bool(r.get("verified_complete")) for r in rows),
            "missing_job_ids": sorted(expected - seen),
            "all_verified_complete": seen == expected and not data_review and all(r.get("verified_complete", False) for r in rows),
            "cross_pc_shared_input_count": len(shared_inputs),
            "cross_pc_shared_input_sha256": shared_inputs, "data_review_required": data_review,
            "cross_pc_scope": "Overlap equality for source/cache/trace/gnb inputs. Each PC may build different binary/library bytes; these hashes remain in each shard report.",
            "jobs": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = combine(validate_manifest(read_json(args.manifest)), args.reports)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "jobs"}, ensure_ascii=False))
    raise SystemExit(0 if report["all_verified_complete"] else 1)

#!/usr/bin/env python3
"""Verify or transfer ignored replay observations using the tracked manifests."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "smo/agentic_ai/agora_offline/runtime"
VARIANTS = ("assets", "assets_v16")


def inventory(runtime=RUNTIME):
    """Return the exact allowed archive members from the committed manifests."""
    result = {}
    for variant in VARIANTS:
        manifest = json.loads((runtime / variant / "manifest.json").read_text())
        for row in manifest["files"]:
            name = PurePosixPath(row["file"])
            if not str(name).startswith("runs/"):
                continue
            if len(name.parts) != 2 or name.suffix != ".npz" or name.name in (".", ".."):
                raise ValueError(f"Invalid replay manifest path: {name}")
            key = f"{variant}/{name}"
            if key in result:
                raise ValueError(f"Duplicate manifest path: {key}")
            target = (runtime / key).resolve()
            if not target.is_relative_to(runtime.resolve()):
                raise ValueError(f"Replay path escapes runtime: {key}")
            result[key] = {"path": target, **row}
    return result


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(runtime=RUNTIME):
    rows = inventory(runtime)
    failures = []
    for name, row in rows.items():
        if not row["path"].is_file():
            failures.append(f"missing: {name}")
        elif digest(row["path"]) != row["sha256"]:
            failures.append(f"checksum mismatch: {name}")
    if failures:
        raise ValueError(
            f"{len(failures)} replay files need restoration; "
            "use python scripts/agora_replay.py import <archive.zip>.\n" + "\n".join(failures[:10])
        )
    return {"files": len(rows), "bytes": sum(row["path"].stat().st_size for row in rows.values())}


def export_archive(archive, runtime=RUNTIME):
    """Keep NPZ members compressed as-is; never modify local observations."""
    report = verify(runtime)
    archive = Path(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as bundle:
        for name, row in inventory(runtime).items():
            bundle.write(row["path"], name)
    return {**report, "archive": str(archive), "sha256": digest(archive)}


def import_archive(archive, runtime=RUNTIME):
    """Validate the complete archive before installing missing or corrupt members."""
    rows = inventory(runtime)
    restored = 0
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(set(names)) != len(names) or set(names) != set(rows):
            raise ValueError("Archive entries must exactly match the replay manifests")
        for name, row in rows.items():
            info = bundle.getinfo(name)
            if info.file_size != row["bytes"] or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Invalid archive member: {name}")
            if hashlib.sha256(bundle.read(name)).hexdigest() != row["sha256"]:
                raise ValueError(f"Archive checksum mismatch: {name}")
        for name, row in rows.items():
            target = row["path"]
            if target.is_file() and digest(target) == row["sha256"]:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=target.parent, prefix=".replay-", delete=False
                ) as stream:
                    temporary = Path(stream.name)
                    stream.write(bundle.read(name))
                temporary.replace(target)
                restored += 1
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
    return {**verify(runtime), "restored": restored}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    for name in ("export", "import"):
        sub.add_parser(name).add_argument("archive", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "verify":
            report = verify()
        elif args.command == "export":
            report = export_archive(args.archive)
        else:
            report = import_archive(args.archive)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

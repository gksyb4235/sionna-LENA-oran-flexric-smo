"""Package the selected frozen model and replay data without training dependencies.

Run once in the original workspace:
  python services/agora_offline_stage/agora_offline/runtime/build_assets.py --workspace .

The resulting runtime directory can be copied to Linux unchanged. Run IDs retain
their ORIGINAL training IDs, not the later Final_Dataset renumbering.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

HERE = Path(__file__).resolve().parent
CHECKPOINT_SHA256 = "b9fc9cbdb2b1245948428386d52969cc93537410e54d6c161b0a471ea0d9c324"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=HERE / "assets")
    args = parser.parse_args()
    workspace, destination = args.workspace.resolve(), args.destination.resolve()
    data_workspace = workspace
    if (workspace / "experiments/legacy").is_dir():
        workspace = workspace / "experiments/legacy"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "runs").mkdir(exist_ok=True)
    checkpoint = workspace / "agora_final_90_180/model/parameter5_backbone.pt"
    if digest(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Selected checkpoint hash does not match the frozen final model")
    shutil.copy2(checkpoint, destination / "parameter5_backbone.pt")
    # Keep model computations verbatim, changing only local imports. The parity
    # test compares this vendored chain with the original final Forecast.
    files = {
        "base_model.py": ("agora_paper_v13/models.py", {
            "from agora_compact_v6.models import dense":
            "def dense(n, h, out, dropout=0.):\n    return nn.Sequential(nn.Linear(n,h), nn.SiLU(), nn.Dropout(dropout), nn.Linear(h,out))"
        }),
        "horizon_model.py": ("agora_horizon_v14/models.py", {
            "from agora_paper_v13.models import PaperGNN as Base": "from .base_model import PaperGNN as Base"
        }),
        "edge_model.py": ("agora_edge_v15/models.py", {
            "from agora_horizon_v14.models import PaperGNN as Baseline": "from .horizon_model import PaperGNN as Baseline",
            "from agora_paper_v13.models import FourTypeBlock": "from .base_model import FourTypeBlock"
        }),
    }
    source_hashes = {}
    for local_name, (source_name, replacements) in files.items():
        source = workspace / source_name
        contents = source.read_text(encoding="utf-8")
        for old, new in replacements.items():
            if old not in contents:
                raise ValueError(f"Expected source import absent: {source_name}: {old}")
            contents = contents.replace(old, new)
        (HERE / local_name).write_text(contents, encoding="utf-8", newline="\n")
        source_hashes[source_name] = digest(source)
    shutil.copy2(workspace / "agora_paper_v13/app_registry.json", HERE / "app_registry.json")

    source = workspace / "agora_second_v7/data"
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    for name in ("y.npy", "valid.npy", "aux.npy", "num.npy", "den.npy", "runs.npz"):
        if digest(source / name) != metadata["files_sha256"][name]:
            raise ValueError(f"Original prepared data hash mismatch: {name}")
    arrays = {name: np.load(source / f"{name}.npy", mmap_mode="r")
              for name in ("y", "valid", "aux", "num", "den")}
    with np.load(source / "runs.npz", allow_pickle=False) as index:
        run_ids, params, splits = (index[k].copy() for k in ("run_ids", "params", "split"))
    renumber = {}
    mapping = data_workspace / "Final_Dataset/_renumber_map.csv"
    if mapping.exists():
        with mapping.open(encoding="utf-8-sig", newline="") as handle:
            renumber = {r["original_name"]: r["name"] for r in csv.DictReader(handle)}
    parameter_values = {name: np.unique(params[:, i]).astype(float).tolist()
                        for i, name in enumerate(metadata["parameter_names"])}
    index = []
    for i, run_id in enumerate(run_ids):
        index.append({"run_id": str(run_id), "split": str(splits[i]),
                      "final_dataset_run_id": renumber.get(str(run_id)),
                      "current_rcp": dict(zip(metadata["parameter_names"], params[i].astype(float).tolist())),
                      "original_cell_csv_sha256": metadata["original_cell_sha256"][str(run_id)]})

    def one(i):
        path = destination / "runs" / (str(run_ids[i]) + ".npz")
        np.savez_compressed(path, **{name: a[i] for name, a in arrays.items()})
        return {"file": str(path.relative_to(destination)).replace("\\", "/"),
                "bytes": path.stat().st_size, "sha256": digest(path)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(one, range(len(run_ids))))
    result = {
        "format": "agora-offline-runtime-v1", "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_file_hashes_verified": True,
        "source_model_sha256": source_hashes,
        "source_data_sha256": {name: metadata["files_sha256"][name]
                               for name in ("y.npy", "valid.npy", "aux.npy", "num.npy", "den.npy", "runs.npz")},
        "parameter_names": metadata["parameter_names"], "parameter_values": parameter_values,
        "target_names": metadata["target_names"], "input_names": metadata["input_names"],
        "cell_labels": metadata["cell_labels"], "primary": metadata["primary"],
        "run_id_namespace": "Original training IDs; final_dataset_run_id is explicitly separate",
        "split_counts": metadata["split_counts"], "runs": index, "files": records,
    }
    (destination / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"runs": len(records), "compressed_replay_bytes": sum(r["bytes"] for r in records),
                      "destination": str(destination), "checkpoint_sha256": CHECKPOINT_SHA256}, indent=2))


if __name__ == "__main__":
    main()

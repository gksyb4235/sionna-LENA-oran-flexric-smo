#!/usr/bin/env python3
"""Generate the 1000-row GNN training parameter grid: one row per UE mobility
seed (scenarios/khu-real/ue_position/ue_positions_seed{0..999}.csv), with
per-cell TxP/RET/CIO plus scenario-wide TTT/HYS drawn uniformly at random
(independently, per column) from each xApp's candidate value set. IID-ness
of the resulting sample over the full joint space is explicitly not a design
goal here -- see conversation notes.

Columns (12): name, txp_5g, txp_4g1, txp_4g2, ret_5g, ret_4g1, ret_4g2,
              cio_5g, cio_4g1, cio_4g2, ttt_ms, hys_db

- TxP (Energy Saving, per-cell): {43,40,37,33,30} dBm
- RET (Coverage Optimization, per-cell): {3,5,7,9,12} deg -- fully
  independent per cell; see build_merged_ret_cache.py for how the ns-3 run
  gets a matching Sionna cache for any of the 5^3=125 combinations.
- CIO (Load Balancing, per-cell): offset in {-2,-1,0,1,2} dB applied to a
  per-cell default (gNB_5G default=7, gNB_4G_1/gNB_4G_2 default=0); the
  stored column is the resulting *absolute* CIO fed to --cellCioDb.
- TTT (Mobility Optimization, scenario-wide): {128,256,340,512,720} ms
- HYS (Mobility Optimization, scenario-wide): {0.5,1.0,...,5.0} dB (10 values, 0.5 step)

Usage:
    generate_gnn_dataset_params.py [--n 1000] [--seed 2026] [--out PATH]
"""
import argparse
import csv
import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KHU_REAL_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_OUT = os.path.join(KHU_REAL_DIR, "gnn_dataset_params.csv")

TXP_VALUES = [43, 40, 37, 33, 30]
RET_VALUES = [3, 5, 7, 9, 12]
CIO_OFFSETS = [-2, -1, 0, 1, 2]
CIO_DEFAULT_5G = 7
CIO_DEFAULT_4G = 0
TTT_VALUES = [128, 256, 340, 512, 720]
HYS_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

COLUMNS = [
    "name", "txp_5g", "txp_4g1", "txp_4g2",
    "ret_5g", "ret_4g1", "ret_4g2",
    "cio_5g", "cio_4g1", "cio_4g2",
    "ttt_ms", "hys_db",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026, help="RNG seed for reproducibility")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    rows = []
    for i in range(args.n):
        txp_5g, txp_4g1, txp_4g2 = rng.choice(TXP_VALUES, size=3)
        ret_5g, ret_4g1, ret_4g2 = rng.choice(RET_VALUES, size=3)
        off_5g, off_4g1, off_4g2 = rng.choice(CIO_OFFSETS, size=3)
        ttt_ms = rng.choice(TTT_VALUES)
        hys_db = rng.choice(HYS_VALUES)

        rows.append({
            "name": f"gnn_{i:04d}",
            "txp_5g": int(txp_5g), "txp_4g1": int(txp_4g1), "txp_4g2": int(txp_4g2),
            "ret_5g": int(ret_5g), "ret_4g1": int(ret_4g1), "ret_4g2": int(ret_4g2),
            "cio_5g": CIO_DEFAULT_5G + int(off_5g),
            "cio_4g1": CIO_DEFAULT_4G + int(off_4g1),
            "cio_4g2": CIO_DEFAULT_4G + int(off_4g2),
            "ttt_ms": int(ttt_ms),
            "hys_db": float(hys_db),
        })

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")

    # Quick coverage summary, useful sanity check on the random draw.
    import collections
    for col in ["ret_5g", "ret_4g1", "ret_4g2", "ttt_ms", "hys_db"]:
        counts = collections.Counter(r[col] for r in rows)
        print(f"  {col}: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()

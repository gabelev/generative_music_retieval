"""Random Semantic ID baseline.

Assign uniform random 3-token SemIDs per track. Tests whether the
generative model can memorize arbitrary mappings without any structure.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cliques-csv", required=True,
                        help="CSV with a track_id column.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-levels", type=int, default=3)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.cliques_csv) as f:
        rows = [r["track_id"] for r in csv.DictReader(f)]
    rng = np.random.default_rng(args.seed)
    codes = rng.integers(0, args.codebook_size, size=(len(rows), args.n_levels))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["track_id", "c1", "c2", "c3"])
        w.writeheader()
        for tid, row in zip(rows, codes):
            w.writerow({"track_id": tid, "c1": int(row[0]), "c2": int(row[1]), "c3": int(row[2])})

    unique = len(set(map(tuple, codes.tolist())))
    print(f"Random IDs: {len(rows)} tracks, codebook {args.codebook_size}^{args.n_levels}")
    print(f"  unique SemIDs: {unique}/{len(rows)} (clash rate {1 - unique/len(rows):.4f})")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

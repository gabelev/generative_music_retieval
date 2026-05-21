"""Read predictions JSON + cliques CSV and emit MR1/MRR/MAP/Recall@k.

Predictions JSON shape (from inference.py or biencoder.py):
  { "<query_track_id>": ["<track_id_1>", "<track_id_2>", ...], ... }
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.evaluation.metrics import compute_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True,
                        help="JSON: {query_id: [ranked track_ids]}")
    parser.add_argument("--cliques-csv", required=True,
                        help="CSV with at least track_id, clique_id columns.")
    parser.add_argument("--out", required=True,
                        help="JSON metrics report.")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10])
    args = parser.parse_args()

    with open(args.predictions) as f:
        predictions: dict[str, list[str]] = json.load(f)

    clique_of: dict[str, str] = {}
    with open(args.cliques_csv) as f:
        for r in csv.DictReader(f):
            clique_of[r["track_id"]] = r["clique_id"]

    metrics = compute_all(predictions, clique_of, ks=args.ks)
    print("Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:14s}: {v:.4f}")
        else:
            print(f"  {k:14s}: {v}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

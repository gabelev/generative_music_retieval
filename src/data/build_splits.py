"""Build clique-level train/val/test splits and (query, target) training pairs.

  - Splits BY CLIQUE (never by track) to prevent leakage.
  - Drops cliques with <2 tracks present in the SemID file (no positive pair).
  - For each clique in a split, emits all ordered (query_track, target_track)
    pairs with their respective SemIDs.

Inputs:
  --cliques-csv : track_id, clique_id, [filepath, ...]
  --semids-csv  : track_id, c1, c2, c3

Outputs:
  <out-dir>/<name>_train.csv
  <out-dir>/<name>_val.csv
  <out-dir>/<name>_test.csv

Each row: clique_id, query_track_id, target_track_id,
          query_c1..c3, target_c1..c3
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cliques-csv", required=True)
    parser.add_argument("--semids-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--name", required=True,
                        help="Prefix for output files (e.g. discogs_vi_mert_ids).")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-by", choices=["pair", "clique"], default="pair",
                        help="pair (default): transductive/DSI-style — every clique is in the "
                             "corpus, individual (query->cover) pairs are held out. "
                             "clique: inductive/cold-start — entire cliques held out (much harder).")
    args = parser.parse_args()

    # Load SemIDs
    semids: dict[str, tuple[int, int, int]] = {}
    with open(args.semids_csv) as f:
        for r in csv.DictReader(f):
            semids[r["track_id"]] = (int(r["c1"]), int(r["c2"]), int(r["c3"]))
    print(f"Loaded {len(semids)} SemIDs from {args.semids_csv}")

    # Load cliques
    by_clique: dict[str, list[str]] = defaultdict(list)
    with open(args.cliques_csv) as f:
        for r in csv.DictReader(f):
            tid = r["track_id"]
            if tid in semids:  # only keep tracks with a SemID
                by_clique[r["clique_id"]].append(tid)

    # Keep only cliques with >=2 tracks (need at least 1 pair)
    usable = {c: ts for c, ts in by_clique.items() if len(ts) >= 2}
    dropped = len(by_clique) - len(usable)
    print(f"Cliques: {len(by_clique)} total, {len(usable)} usable (>=2 tracks), {dropped} dropped")

    rng = random.Random(args.seed)

    def all_pairs(cid: str) -> list[dict]:
        rows = []
        tracks = usable[cid]
        for q in tracks:
            for t in tracks:
                if q == t:
                    continue
                qs, ts = semids[q], semids[t]
                rows.append({
                    "clique_id": cid,
                    "query_track_id": q, "target_track_id": t,
                    "query_c1": qs[0], "query_c2": qs[1], "query_c3": qs[2],
                    "target_c1": ts[0], "target_c2": ts[1], "target_c3": ts[2],
                })
        return rows

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    if args.split_by == "clique":
        # Inductive / cold-start: whole cliques go to one split.
        clique_ids = sorted(usable.keys())
        rng.shuffle(clique_ids)
        n = len(clique_ids)
        n_train = int(args.train_frac * n)
        n_val = int(args.val_frac * n)
        assign = ({c: "train" for c in clique_ids[:n_train]}
                  | {c: "val" for c in clique_ids[n_train:n_train + n_val]}
                  | {c: "test" for c in clique_ids[n_train + n_val:]})
        for cid in clique_ids:
            splits[assign[cid]].extend(all_pairs(cid))
        print(f"Clique split: train={n_train}, val={n_val}, test={n - n_train - n_val} cliques")
    else:
        # Transductive (DSI/TIGER-style): every clique contributes pairs to every
        # split, with >=1 pair guaranteed in train so each track is indexed.
        for cid in sorted(usable.keys()):
            pairs = all_pairs(cid)
            rng.shuffle(pairs)
            n = len(pairs)
            n_train = max(1, int(args.train_frac * n))
            n_val = int(args.val_frac * n)
            # never let val/test eat the only pairs
            n_val = min(n_val, max(0, n - n_train))
            splits["train"].extend(pairs[:n_train])
            splits["val"].extend(pairs[n_train:n_train + n_val])
            splits["test"].extend(pairs[n_train + n_val:])
        print(f"Pair split (transductive): every clique in corpus; pairs held out per-clique")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clique_id",
        "query_track_id", "target_track_id",
        "query_c1", "query_c2", "query_c3",
        "target_c1", "target_c2", "target_c3",
    ]

    total_pairs = 0
    for split_name, rows in splits.items():
        out_path = out_dir / f"{args.name}_{split_name}.csv"
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        total_pairs += len(rows)
        n_q = len({r["query_track_id"] for r in rows})
        print(f"  {split_name}: {len(rows)} pairs, {n_q} unique queries -> {out_path.name}")

    print(f"Total ordered pairs: {total_pairs}")


if __name__ == "__main__":
    main()

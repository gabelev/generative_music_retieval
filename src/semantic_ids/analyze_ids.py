"""Semantic ID quality diagnostics.

For a SemID CSV (track_id, c1, c2, c3) joined with a cliques CSV (track_id, clique_id):
  - clash rate        : 1 - n_unique_semids / n_tracks
  - codebook utilization per level
  - within-clique vs cross-clique prefix-overlap rates (do covers share prefixes?)

The "structure" hypothesis predicts within > cross for MERT/EnCodec SemIDs and
within == cross for random IDs.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_semids(path: Path) -> dict[str, tuple[int, int, int]]:
    with path.open() as f:
        return {r["track_id"]: (int(r["c1"]), int(r["c2"]), int(r["c3"]))
                for r in csv.DictReader(f)}


def load_cliques(path: Path) -> dict[str, str]:
    """track_id -> clique_id"""
    with path.open() as f:
        return {r["track_id"]: r["clique_id"] for r in csv.DictReader(f)}


def prefix_overlap_stats(
    semids: dict[str, tuple[int, int, int]],
    cliques: dict[str, str],
    seed: int = 42,
    n_cross_pairs_target: int = 100_000,
) -> dict:
    """Compare within-clique vs cross-clique prefix-share rates.

    Returns rates for sharing c1, (c1,c2), (c1,c2,c3). The cross-clique sample
    is a random subset to keep the comparison computationally tractable.
    """
    # Group by clique
    by_clique: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for tid, sem in semids.items():
        if tid in cliques:
            by_clique[cliques[tid]].append(sem)

    # Within-clique pairs
    within_pairs: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    for c, sems in by_clique.items():
        for i in range(len(sems)):
            for j in range(i + 1, len(sems)):
                within_pairs.append((sems[i], sems[j]))

    # Cross-clique: random sample of ordered pairs from different cliques
    all_tracks = [(c, sem) for c, sems in by_clique.items() for sem in sems]
    rng = np.random.default_rng(seed)
    cross_pairs: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    n_attempts = 0
    while len(cross_pairs) < n_cross_pairs_target and n_attempts < n_cross_pairs_target * 4:
        a, b = rng.integers(0, len(all_tracks), size=2)
        if all_tracks[a][0] != all_tracks[b][0]:
            cross_pairs.append((all_tracks[a][1], all_tracks[b][1]))
        n_attempts += 1

    def rates(pairs):
        if not pairs:
            return {"n": 0, "p1": 0.0, "p12": 0.0, "p123": 0.0}
        n = len(pairs)
        p1 = sum(1 for a, b in pairs if a[0] == b[0]) / n
        p12 = sum(1 for a, b in pairs if a[:2] == b[:2]) / n
        p123 = sum(1 for a, b in pairs if a == b) / n
        return {"n": n, "p1": p1, "p12": p12, "p123": p123}

    return {"within": rates(within_pairs), "cross": rates(cross_pairs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semids-csv", required=True)
    parser.add_argument("--cliques-csv", required=True)
    parser.add_argument("--out", required=True,
                        help="JSON report path.")
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--show-examples", type=int, default=5,
                        help="Print SemIDs of N cliques for visual sanity check.")
    args = parser.parse_args()

    semids = load_semids(Path(args.semids_csv))
    cliques = load_cliques(Path(args.cliques_csv))
    print(f"Loaded {len(semids)} SemIDs, {len(cliques)} clique-mapped tracks.")

    # Restrict to tracks present in both
    common = {tid: semids[tid] for tid in semids if tid in cliques}
    print(f"  Intersection: {len(common)} tracks")

    sem_arr = np.array(list(common.values()))
    unique_semids = len(set(map(tuple, sem_arr.tolist())))
    util = [len(np.unique(sem_arr[:, lvl])) for lvl in range(sem_arr.shape[1])]

    print(f"Clash rate                       : {1 - unique_semids / len(common):.4f}")
    print(f"Unique SemIDs                    : {unique_semids} / {len(common)}")
    print(f"Codebook utilization per level   : {util} (out of {args.codebook_size})")

    overlap = prefix_overlap_stats(common, cliques)
    print()
    print("Prefix-overlap rates (do covers share SemID prefixes?):")
    print(f"  WITHIN clique  (n={overlap['within']['n']}):")
    print(f"    share c1     : {overlap['within']['p1']:.4f}")
    print(f"    share (c1,c2): {overlap['within']['p12']:.4f}")
    print(f"    share all    : {overlap['within']['p123']:.4f}")
    print(f"  CROSS clique   (n={overlap['cross']['n']}):")
    print(f"    share c1     : {overlap['cross']['p1']:.4f}")
    print(f"    share (c1,c2): {overlap['cross']['p12']:.4f}")
    print(f"    share all    : {overlap['cross']['p123']:.4f}")

    # Example cliques
    if args.show_examples:
        from collections import defaultdict
        by_clique: dict[str, list[tuple[str, tuple[int, int, int]]]] = defaultdict(list)
        for tid, sem in common.items():
            by_clique[cliques[tid]].append((tid, sem))
        examples = [(c, tids) for c, tids in by_clique.items() if len(tids) >= 2][:args.show_examples]
        print()
        print(f"Example cliques (first {args.show_examples}):")
        for c, tids in examples:
            print(f"  clique {c}:")
            for tid, sem in tids[:4]:
                print(f"    {tid[:20]:20s}  ({sem[0]:>3}, {sem[1]:>3}, {sem[2]:>3})")

    report = {
        "n_tracks": len(common),
        "n_unique_semids": unique_semids,
        "clash_rate": 1 - unique_semids / len(common),
        "codebook_utilization": util,
        "codebook_size": args.codebook_size,
        "prefix_overlap": overlap,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

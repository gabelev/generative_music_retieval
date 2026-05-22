"""Error analysis for generative retrieval: when T5 retrieves a WRONG track,
is that track still acoustically close to the query?

Graceful-degradation hypothesis: even on a miss, the generative model emits a
Semantic ID near the right region of the space, so the resolved (wrong) track
is more MERT-similar to the query than a random track.

For the T5 predictions file, we split test queries into hits (a correct cover
appears in the predicted list) and misses, then compare three MERT cosine
similarities:
  - query -> its true clique-mates        (ceiling)
  - query -> T5's top-1 prediction on a MISS  (the quantity of interest)
  - query -> random corpus tracks         (floor)

Outputs a JSON report and a histogram figure.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_embeddings(path: Path) -> tuple[dict[str, int], np.ndarray]:
    with np.load(path) as npz:
        ids = list(npz.files)
        mat = np.stack([npz[t] for t in ids], axis=0).astype(np.float32)
    mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    return {t: i for i, t in enumerate(ids)}, mat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True,
                        help="T5 predictions JSON: {query_id: [ranked track_ids]}")
    parser.add_argument("--embeddings", required=True,
                        help="MERT embeddings .npz (for acoustic similarity).")
    parser.add_argument("--cliques-csv", required=True)
    parser.add_argument("--out", required=True, help="JSON report path.")
    parser.add_argument("--fig", default="paper/figures/error_analysis.pdf")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    idx, emb = load_embeddings(Path(args.embeddings))
    clique_of: dict[str, str] = {}
    with open(args.cliques_csv) as f:
        for r in csv.DictReader(f):
            clique_of[r["track_id"]] = r["clique_id"]
    by_clique: dict[str, list[str]] = defaultdict(list)
    for t, c in clique_of.items():
        by_clique[c].append(t)

    with open(args.predictions) as f:
        predictions: dict[str, list[str]] = json.load(f)

    rng = np.random.default_rng(args.seed)

    def cos(a: str, b: str) -> float | None:
        if a not in idx or b not in idx:
            return None
        return float(emb[idx[a]] @ emb[idx[b]])

    ceiling: list[float] = []      # query -> true clique-mates
    miss_pred: list[float] = []    # query -> top-1 prediction, on a miss
    floor: list[float] = []        # query -> random tracks
    n_hit = n_miss = n_miss_empty = 0
    all_track_ids = list(idx.keys())

    for qid, preds in predictions.items():
        qc = clique_of.get(qid)
        if qc is None:
            continue
        mates = [t for t in by_clique.get(qc, []) if t != qid]
        if not mates:
            continue

        # ceiling: similarity to true clique-mates
        for m in mates:
            s = cos(qid, m)
            if s is not None:
                ceiling.append(s)

        # floor: similarity to random tracks
        for _ in range(3):
            r = all_track_ids[rng.integers(0, len(all_track_ids))]
            if r != qid:
                s = cos(qid, r)
                if s is not None:
                    floor.append(s)

        # hit/miss
        hit = any(clique_of.get(p) == qc and p != qid for p in preds)
        if hit:
            n_hit += 1
            continue
        n_miss += 1
        if not preds:
            n_miss_empty += 1
            continue
        s = cos(qid, preds[0])
        if s is not None:
            miss_pred.append(s)

    def stats(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0, "mean": None, "std": None, "median": None}
        a = np.array(xs)
        return {"n": len(a), "mean": float(a.mean()), "std": float(a.std()),
                "median": float(np.median(a))}

    report = {
        "n_queries": n_hit + n_miss,
        "n_hit": n_hit,
        "n_miss": n_miss,
        "n_miss_empty_prediction": n_miss_empty,
        "cos_query_to_true_cover": stats(ceiling),
        "cos_query_to_t5_miss_prediction": stats(miss_pred),
        "cos_query_to_random": stats(floor),
    }
    print(json.dumps(report, indent=2))

    cm = report["cos_query_to_t5_miss_prediction"]["mean"]
    cf = report["cos_query_to_random"]["mean"]
    cc = report["cos_query_to_true_cover"]["mean"]
    if cm is not None:
        print()
        print(f"Graceful degradation check:")
        print(f"  true cover    : {cc:.4f}")
        print(f"  T5 miss top-1 : {cm:.4f}")
        print(f"  random        : {cf:.4f}")
        verdict = "graceful (misses land between random and true)" if cm > cf else "not graceful"
        print(f"  --> {verdict}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out}")

    # Histogram figure
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        bins = np.linspace(-0.2, 1.0, 40)
        ax.hist(floor, bins=bins, alpha=0.55, label="query vs random", color="#bdbdbd", density=True)
        ax.hist(miss_pred, bins=bins, alpha=0.65, label="query vs T5 miss (top-1)",
                color="#e8710a", density=True)
        ax.hist(ceiling, bins=bins, alpha=0.55, label="query vs true cover", color="#1a73e8",
                density=True)
        ax.set_xlabel("MERT cosine similarity", fontsize=9)
        ax.set_ylabel("density", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        figp = Path(args.fig)
        figp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figp, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Wrote {figp}")
    except Exception as e:
        print(f"figure skipped: {e!r}")


if __name__ == "__main__":
    main()

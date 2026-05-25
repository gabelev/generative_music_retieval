"""MERT bi-encoder + FAISS retrieval baseline.

For each query track, find the top-K nearest tracks by cosine similarity of
their MERT embeddings. This is the "standard practice" baseline that the
generative-retrieval pipeline claims to replace.

Eval restricts queries to those in the test split (so the comparison with T5
is apples-to-apples), but the FAISS index is built over ALL tracks in the
dataset (the model has access to the full candidate pool, as in deployment).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True,
                        help=".npz from extract_mert.py, keyed by track_id (768-d).")
    parser.add_argument("--test-csv", required=True,
                        help="The test split CSV; deduplicates queries from it.")
    parser.add_argument("--cliques-csv", required=True,
                        help="track_id -> clique_id mapping. Used to gate the candidate pool.")
    parser.add_argument("--out", required=True,
                        help="JSON: {query_track_id: [track_id, ...]}.")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidate-pool", default="all",
                        choices=["all", "test_cliques_only"],
                        help="'all'=index every embedded track; 'test_cliques_only'="
                             "restrict candidates to tracks whose clique appears in the test set "
                             "(closed-world eval, lower bar; reported for completeness).")
    args = parser.parse_args()

    print(f"Loading embeddings from {args.embeddings}")
    with np.load(args.embeddings) as npz:
        track_ids = list(npz.files)
        emb = np.stack([npz[t] for t in track_ids], axis=0).astype(np.float32)
    print(f"  {emb.shape}")

    clique_of: dict[str, str] = {}
    with open(args.cliques_csv) as f:
        for r in csv.DictReader(f):
            clique_of[r["track_id"]] = r["clique_id"]

    # Dedup queries from test split
    queries: list[str] = []
    test_cliques: set[str] = set()
    seen: set[str] = set()
    with open(args.test_csv) as f:
        for r in csv.DictReader(f):
            qid = r["query_track_id"]
            test_cliques.add(r["clique_id"])
            if qid not in seen:
                seen.add(qid)
                queries.append(qid)
    print(f"  {len(queries)} unique queries from {args.test_csv}")

    # Candidate pool
    if args.candidate_pool == "test_cliques_only":
        keep_idxs = [i for i, t in enumerate(track_ids) if clique_of.get(t) in test_cliques]
    else:
        keep_idxs = list(range(len(track_ids)))
    pool_ids = [track_ids[i] for i in keep_idxs]
    pool_emb = emb[keep_idxs]
    print(f"  candidate pool: {len(pool_ids)} tracks ({args.candidate_pool})")

    # L2-normalize -> inner product = cosine
    pool_emb = pool_emb.copy()
    faiss.normalize_L2(pool_emb)
    index = faiss.IndexFlatIP(pool_emb.shape[1])
    index.add(pool_emb)
    id_of_pool_pos = {i: tid for i, tid in enumerate(pool_ids)}

    # Build query batch
    q_idxs: list[int] = []
    valid_queries: list[str] = []
    for qid in queries:
        if qid not in track_ids:
            continue
        q_idxs.append(track_ids.index(qid))
        valid_queries.append(qid)
    if len(valid_queries) < len(queries):
        print(f"  dropped {len(queries) - len(valid_queries)} queries with no embedding")
    q_emb = emb[q_idxs].copy()
    faiss.normalize_L2(q_emb)

    # k+1 in case the top hit is the query itself; clamp to pool size to avoid
    # FAISS returning -1 sentinels when top_k >= pool_size.
    k_search = min(args.top_k + 1, len(pool_ids))
    distances, indices = index.search(q_emb, k_search)

    predictions: dict[str, list[str]] = {}
    for i, qid in enumerate(valid_queries):
        ranked: list[str] = []
        for pos in indices[i]:
            pos = int(pos)
            if pos < 0:    # FAISS pads with -1 when fewer than k neighbors exist
                continue
            tid = id_of_pool_pos[pos]
            if tid == qid:
                continue
            ranked.append(tid)
            if len(ranked) >= args.top_k:
                break
        predictions[qid] = ranked

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

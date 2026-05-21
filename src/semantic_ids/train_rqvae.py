"""Sequential-kmeans RQ-VAE on MERT embeddings -> 3-level Semantic IDs.

For each level: fit MiniBatchKMeans on the current residual, assign each
embedding to its nearest centroid, subtract that centroid, repeat. The
quantized codes form the Semantic ID.

Outputs:
  --out-codebooks: pickle of list[MiniBatchKMeans]  (one per level)
  --out-semids:    CSV  (track_id, c1, c2, c3)
"""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def train_rqvae(embeddings: np.ndarray, n_levels: int, codebook_size: int,
                seed: int = 42, batch_size: int = 4096) -> list[MiniBatchKMeans]:
    codebooks: list[MiniBatchKMeans] = []
    residual = embeddings.astype(np.float32, copy=True)
    for level in range(n_levels):
        km = MiniBatchKMeans(
            n_clusters=codebook_size,
            batch_size=batch_size,
            random_state=seed + level,
            n_init=3,
            max_iter=200,
            reassignment_ratio=0.01,
        )
        km.fit(residual)
        assigned = km.predict(residual)
        centroids = km.cluster_centers_[assigned]
        residual = residual - centroids
        print(f"  level {level}: trained codebook size {codebook_size}, "
              f"residual norm mean={np.linalg.norm(residual, axis=1).mean():.4f}")
        codebooks.append(km)
    return codebooks


def encode_rqvae(embeddings: np.ndarray, codebooks: list[MiniBatchKMeans]) -> np.ndarray:
    """Return [N, n_levels] int code matrix."""
    residual = embeddings.astype(np.float32, copy=True)
    codes = []
    for km in codebooks:
        assigned = km.predict(residual)
        codes.append(assigned)
        residual = residual - km.cluster_centers_[assigned]
    return np.stack(codes, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True,
                        help=".npz keyed by track_id, values are 768-d arrays.")
    parser.add_argument("--out-codebooks", required=True,
                        help="Pickle file of trained codebooks.")
    parser.add_argument("--out-semids", required=True,
                        help="CSV: track_id, c1, c2, c3.")
    parser.add_argument("--n-levels", type=int, default=3)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-embeddings", default=None,
                        help="Optional separate NPZ to TRAIN the codebooks on; "
                             "--embeddings is then only used to ENCODE. Use this to "
                             "train on the larger Discogs-VI set and encode Covers80 with "
                             "the same codebooks for cross-dataset consistency.")
    args = parser.parse_args()

    print(f"Loading embeddings from {args.embeddings}")
    with np.load(args.embeddings) as npz:
        track_ids = list(npz.files)
        emb = np.stack([npz[t] for t in track_ids], axis=0)
    print(f"  shape: {emb.shape}, dtype: {emb.dtype}")

    if args.train_embeddings:
        print(f"Training codebooks on {args.train_embeddings}")
        with np.load(args.train_embeddings) as npz:
            train_emb = np.stack([npz[t] for t in npz.files], axis=0)
        print(f"  train shape: {train_emb.shape}")
    else:
        train_emb = emb

    codebooks = train_rqvae(train_emb, args.n_levels, args.codebook_size, seed=args.seed)

    print(f"Encoding {len(track_ids)} tracks")
    codes = encode_rqvae(emb, codebooks)

    out_cb = Path(args.out_codebooks)
    out_cb.parent.mkdir(parents=True, exist_ok=True)
    with out_cb.open("wb") as f:
        pickle.dump(codebooks, f)
    print(f"Saved codebooks -> {out_cb}")

    out_csv = Path(args.out_semids)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["track_id", "c1", "c2", "c3"])
        w.writeheader()
        for tid, row in zip(track_ids, codes):
            w.writerow({"track_id": tid, "c1": int(row[0]), "c2": int(row[1]), "c3": int(row[2])})

    # Quick stats
    unique_per_level = [len(np.unique(codes[:, lvl])) for lvl in range(args.n_levels)]
    unique_semids = len(set(map(tuple, codes.tolist())))
    print(f"Codebook utilization per level: {unique_per_level}")
    print(f"Unique 3-token SemIDs: {unique_semids}/{len(track_ids)} "
          f"(clash rate {1 - unique_semids/len(track_ids):.3f})")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()

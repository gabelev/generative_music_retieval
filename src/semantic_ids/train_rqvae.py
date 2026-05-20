"""Train sequential-kmeans RQ-VAE on MERT embeddings and emit SemIDs.

n_levels=3, codebook_size=256. Save codebooks (sklearn objects) and
per-track SemID assignments.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# TODO: implement per CLAUDE.md Task 7 (train_rqvae + encode_rqvae).


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--out-codebooks", required=True)
    parser.add_argument("--out-semids", required=True)
    parser.add_argument("--n-levels", type=int, default=3)
    parser.add_argument("--codebook-size", type=int, default=256)
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()

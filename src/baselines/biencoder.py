"""MERT bi-encoder + FAISS retrieval baseline.

L2-normalize embeddings, IndexFlatIP, top-K by inner product. This is the
"standard practice" baseline that generative retrieval claims to replace.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# TODO: implement per CLAUDE.md Task 12.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--cliques-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()

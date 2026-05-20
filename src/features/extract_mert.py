"""MERT-v1-95M embedding extraction.

For each track:
  - load audio, resample to 24kHz mono
  - take 30-sec center clip
  - run through frozen MERT-v1-95M
  - mean-pool hidden states from layers 7..12 across time -> 768-d vector

Output: data/embeddings/mert/{dataset}_embeddings.npz  ({track_id: 768-d array})
"""
from __future__ import annotations

import argparse
from pathlib import Path

# TODO: implement per CLAUDE.md Task 5 snippet, batched, GPU-aware.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--cliques-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()

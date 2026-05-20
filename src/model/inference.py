"""Constrained beam-search inference.

Position 0: restrict to <L0_C*> tokens
Position 1: restrict to <L1_C*> tokens
Position 2: restrict to <L2_C*> tokens

For each query, return top-K candidate SemIDs and resolve to track IDs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# TODO: implement prefix_allowed_tokens_fn for constrained decoding,
# beam_search with width 20, map generated SemIDs -> track IDs.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--beam-width", type=int, default=20)
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()

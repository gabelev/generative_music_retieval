"""Random Semantic ID baseline: assign uniform 3-token tuples per track."""
from __future__ import annotations

import argparse
from pathlib import Path

# TODO: implement reproducible random assignment + clash-rate report.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cliques-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-levels", type=int, default=3)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()

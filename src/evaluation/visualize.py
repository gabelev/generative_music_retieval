"""Figures for the paper.

  - t-SNE of SemID embeddings (codebook-expanded), colored by clique
  - prefix-overlap heatmap (within-clique vs cross-clique, per condition)
  - results table -> LaTeX
"""
from __future__ import annotations

import argparse

# TODO: implement.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semids-csv", required=True)
    parser.add_argument("--cliques-csv", required=True)
    parser.add_argument("--out-dir", default="paper/figures")
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()

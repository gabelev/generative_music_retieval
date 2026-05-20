"""Build clique-level train/val/test splits and (query, target_semid) training pairs.

Split BY CLIQUE (never by track) to prevent leakage.
For each clique with versions [A, B, C, ...], emit all ordered pairs:
  (A_semid -> B_semid), (A_semid -> C_semid), (B_semid -> A_semid), ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

# TODO: implement split-by-clique + ordered pair generation per condition.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", choices=["covers80", "discogs_vi"], required=True)
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()

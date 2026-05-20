"""T5-small fine-tuning loop for generative retrieval.

Adds 256*3 SemID tokens + 3 level indicators to vocab, resizes embeddings,
trains with AdamW (lr 5e-4), early-stops on val MRR.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# TODO: implement per CLAUDE.md Task 10.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()

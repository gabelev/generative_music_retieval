"""Convert raw EnCodec frame-level codes to per-track 3-token Semantic IDs.

Aggregation strategies: majority_vote (default), center_frame, mean_then_requantize.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# TODO: implement aggregation strategies; emit CSV (track_id, c1, c2, c3).


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes-npz", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--strategy", default="majority_vote")
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()

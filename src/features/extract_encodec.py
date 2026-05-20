"""EnCodec 24kHz RVQ code extraction.

For each track, return both (a) frame-level codes [n_codebooks, n_frames]
and (b) a per-track SemID derived by majority vote across frames per level
(first 3 codebook levels).
"""
from __future__ import annotations

import argparse
from pathlib import Path

# TODO: implement per CLAUDE.md Task 6 snippet. Save both raw codes (.npz)
# and the aggregated SemID CSV (track_id, c1, c2, c3).


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--cliques-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bandwidth", type=float, default=1.5)
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()

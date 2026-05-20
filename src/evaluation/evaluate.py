"""Top-level evaluation entrypoint.

Loads a predictions file (query_id -> ranked track_ids) and a cliques map,
emits a JSON report with MR1, MRR, MAP, Recall@1/5/10.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# TODO: implement.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--cliques-csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()

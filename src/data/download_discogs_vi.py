"""Download Discogs-VI-YT metadata from Zenodo and pick a working subset.

Source: https://zenodo.org/records/13983028

Outputs:
  data/raw/discogs_vi/Discogs-VI-YT-light-20240701.json
  data/splits/discogs_vi_subset.csv  cols: version_id, clique_id, youtube_id
"""
from __future__ import annotations

import argparse
from pathlib import Path

# TODO: implement metadata fetch + JSON parse + subset selection (2-3K cliques, >=2 versions).


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/raw/discogs_vi")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--n-cliques", type=int, default=2500)
    parser.add_argument("--min-versions", type=int, default=2)
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()

"""Download Covers80 dataset and build clique mapping CSV.

Source: http://labrosa.ee.columbia.edu/projects/coversongs/covers80/

Outputs:
  data/raw/covers80/covers32k/<Artist+Song>/<file>.mp3
  data/splits/covers80_cliques.csv  cols: track_id, clique_id, filepath, version
"""
from __future__ import annotations

import argparse
from pathlib import Path

# TODO: implement fetch + tar extraction, parse list1.txt/list2.txt, emit cliques CSV.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/raw/covers80")
    parser.add_argument("--splits-dir", default="data/splits")
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()

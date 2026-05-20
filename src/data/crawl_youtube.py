"""yt-dlp wrapper to crawl Discogs-VI-YT audio for a subset CSV.

Downloads each YouTube ID, transcodes to 24kHz mono WAV (MERT requirement),
and records success/failure per row.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# TODO: implement yt-dlp batch download + ffmpeg transcode + success log.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-csv", default="data/splits/discogs_vi_subset.csv")
    parser.add_argument("--out-dir", default="data/raw/discogs_vi")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()

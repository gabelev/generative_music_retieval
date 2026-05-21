"""Download successfully-crawled Discogs-VI WAVs from a private HF dataset repo
and emit a CSV pairing track_id (youtube_id) with the local filepath.

This is the bridge between the crawl output (lives on HF) and the feature
extractors (want local paths).

Output:
  <out-dir>/audio/<youtube_id>.wav        (the WAV files)
  data/splits/discogs_vi_audio.csv        (track_id, clique_id, filepath)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", required=True,
                        help="Private HF dataset repo, e.g. gabelev/discogs-vi-csi-subset")
    parser.add_argument("--subset-csv", default="data/splits/discogs_vi_subset.csv",
                        help="The pre-crawl subset CSV (for clique_id mapping).")
    parser.add_argument("--log-csv", default="data/splits/discogs_vi_download_log.csv",
                        help="The crawl log; we filter to status=ok rows.")
    parser.add_argument("--out-dir", default="data/raw/discogs_vi",
                        help="Local dir to materialize the HF audio/ folder under.")
    parser.add_argument("--out-csv", default="data/splits/discogs_vi_audio.csv")
    parser.add_argument("--skip-download", action="store_true",
                        help="Trust an existing local copy; only re-emit the CSV.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                or (Path.home() / ".cache" / "huggingface" / "token").exists()):
            sys.exit("HF auth missing: set HF_TOKEN or run `huggingface-cli login`.")
        print(f"Snapshot-downloading {args.hf_repo} (audio/ only) -> {out_dir}")
        t0 = time.time()
        try:
            snapshot_download(
                repo_id=args.hf_repo,
                repo_type="dataset",
                allow_patterns=["audio/*.wav"],
                local_dir=str(out_dir),
            )
        except HfHubHTTPError as e:
            sys.exit(f"snapshot_download failed: {e}")
        elapsed = time.time() - t0
        print(f"  done in {elapsed:.1f}s")

    audio_subdir = out_dir / "audio"
    if not audio_subdir.exists():
        sys.exit(f"Expected {audio_subdir} after download, missing.")

    # Build youtube_id -> clique_id map (clique_id needed for splits later)
    yid_to_clique: dict[str, str] = {}
    with open(args.subset_csv) as f:
        for r in csv.DictReader(f):
            yid_to_clique[r["youtube_id"]] = r["clique_id"]

    # Get successful youtube_ids from log
    ok_ids: list[str] = []
    if Path(args.log_csv).exists():
        with open(args.log_csv) as f:
            for r in csv.DictReader(f):
                if r.get("status") == "ok":
                    ok_ids.append(r["youtube_id"])

    # If log is missing or empty, fall back to whatever's on disk
    if not ok_ids:
        ok_ids = [p.stem for p in audio_subdir.glob("*.wav")]
        print(f"  (log empty; treating {len(ok_ids)} local WAVs as candidates)")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    missing = 0
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["track_id", "clique_id", "filepath"])
        w.writeheader()
        for yid in ok_ids:
            wav = audio_subdir / f"{yid}.wav"
            if not wav.exists():
                missing += 1
                continue
            clique_id = yid_to_clique.get(yid, "")
            if not clique_id:
                continue
            w.writerow({
                "track_id": yid,
                "clique_id": clique_id,
                "filepath": str(wav.resolve()),
            })
            written += 1

    print(f"Wrote {out_csv}: {written} rows, {missing} expected WAVs missing on disk")


if __name__ == "__main__":
    main()

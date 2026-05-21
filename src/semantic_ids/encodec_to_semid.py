"""Aggregate EnCodec frame-level codes to per-track 3-token Semantic IDs.

Strategies:
  majority_vote (default) — per codebook level, take the most frequent code across frames
  center_frame            — take the code at the middle frame
  first_frame             — take the code at frame 0

Input: NPZ from src/features/extract_encodec.py, keyed by track_id,
       values shape [n_codebooks, n_frames] (int).
Output: CSV with columns track_id, c1, c2, c3 (first 3 codebooks only).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def aggregate(codes: np.ndarray, strategy: str) -> list[int]:
    """codes: [n_codebooks, n_frames]. Returns first-3 codebooks aggregated."""
    n_cb = min(3, codes.shape[0])
    out: list[int] = []
    for cb in range(n_cb):
        row = codes[cb]
        if strategy == "majority_vote":
            vals, cnts = np.unique(row, return_counts=True)
            out.append(int(vals[int(np.argmax(cnts))]))
        elif strategy == "center_frame":
            out.append(int(row[row.shape[0] // 2]))
        elif strategy == "first_frame":
            out.append(int(row[0]))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    while len(out) < 3:
        out.append(0)  # pad if model had fewer than 3 codebooks (warn elsewhere)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes-npz", required=True)
    parser.add_argument("--out", required=True,
                        help="Output CSV: track_id, c1, c2, c3")
    parser.add_argument("--strategy", default="majority_vote",
                        choices=["majority_vote", "center_frame", "first_frame"])
    args = parser.parse_args()

    print(f"Loading {args.codes_npz}")
    with np.load(args.codes_npz) as npz:
        keys = list(npz.files)
        print(f"  {len(keys)} entries")
        if not keys:
            raise SystemExit(
                f"ERROR: {args.codes_npz} has 0 entries — upstream extract_encodec "
                "produced no codes. Check the extraction log for audio-load errors "
                "(missing torchaudio backend / libsndfile1 / ffmpeg)."
            )
        rows = []
        for tid in keys:
            codes = npz[tid]
            c1, c2, c3 = aggregate(codes, args.strategy)
            rows.append({"track_id": tid, "c1": c1, "c2": c2, "c3": c3})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["track_id", "c1", "c2", "c3"])
        w.writeheader()
        w.writerows(rows)

    # Quick stats
    c1_unique = len(set(r["c1"] for r in rows))
    c2_unique = len(set(r["c2"] for r in rows))
    c3_unique = len(set(r["c3"] for r in rows))
    semid_unique = len(set((r["c1"], r["c2"], r["c3"]) for r in rows))
    print(f"Strategy: {args.strategy}")
    print(f"  Codebook utilization: c1={c1_unique}, c2={c2_unique}, c3={c3_unique}")
    print(f"  Unique 3-token SemIDs: {semid_unique}/{len(rows)} "
          f"(clash rate {1 - semid_unique/len(rows):.3f})")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

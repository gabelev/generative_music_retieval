"""Aggregate per-file CLEWS segment embeddings into a single NPZ keyed by track_id.

CLEWS (Serra et al. 2025) inference writes a .pt tensor per input audio file:
each tensor has shape [n_segments, 1024], where the 1024-d vectors come from
the model's projection head and n_segments depends on the input clip length.
Our crawl provides 30-sec clips, so n_segments is typically 1 (or 2).

This script:
  - walks a directory of CLEWS .pt outputs
  - mean-pools segments to a single 1024-d vector per track
  - saves NPZ keyed by track_id (filename stem) — same shape contract as
    extract_mert.py's output, so all downstream code (train_rqvae, analyze_ids,
    bi-encoder, error_analysis) works without changes
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


def load_segments(pt_path: Path) -> np.ndarray:
    """Load a CLEWS .pt file, return [n_segments, 1024] float32 array."""
    obj = torch.load(str(pt_path), map_location="cpu", weights_only=True)
    # CLEWS output shape varies by version; handle the common forms.
    if isinstance(obj, torch.Tensor):
        t = obj
    elif isinstance(obj, dict):
        # Some CLEWS variants nest the tensor under a key like 'embeddings' or 'feats'.
        for k in ("embeddings", "embedding", "feats", "features", "z"):
            if k in obj and isinstance(obj[k], torch.Tensor):
                t = obj[k]
                break
        else:
            raise ValueError(f"{pt_path}: dict without a known embedding key. Keys: {list(obj)}")
    else:
        raise ValueError(f"{pt_path}: unsupported type {type(obj)!r}")
    if t.dim() == 1:
        t = t.unsqueeze(0)
    return t.detach().float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clews-dir", required=True,
                        help="Directory containing CLEWS .pt outputs (one per track).")
    parser.add_argument("--out", required=True,
                        help="Output .npz keyed by track_id (file stem).")
    parser.add_argument("--allow-keys", default=None,
                        help="Optional path to a text file with one track_id per line; "
                             "only those are kept (filters Discogs-VI vs Covers80 split).")
    parser.add_argument("--pool", default="mean", choices=["mean", "first", "center"],
                        help="How to reduce segment dim to one vector per track.")
    parser.add_argument("--expected-dim", type=int, default=1024)
    args = parser.parse_args()

    clews_dir = Path(args.clews_dir)
    if not clews_dir.exists():
        sys.exit(f"--clews-dir {clews_dir} does not exist.")

    allowed: set[str] | None = None
    if args.allow_keys:
        with open(args.allow_keys) as f:
            allowed = {line.strip() for line in f if line.strip()}
        print(f"Filtering to {len(allowed)} allowed track_ids from {args.allow_keys}")

    out: dict[str, np.ndarray] = {}
    bad = 0
    t0 = time.time()
    files = sorted(clews_dir.glob("*.pt"))
    print(f"Found {len(files)} CLEWS .pt files in {clews_dir}")

    for i, p in enumerate(files, 1):
        tid = p.stem
        if allowed is not None and tid not in allowed:
            continue
        try:
            segs = load_segments(p)
        except Exception as e:
            print(f"  fail {tid}: {e!r}", file=sys.stderr)
            bad += 1
            continue
        if segs.shape[-1] != args.expected_dim:
            print(f"  warn {tid}: unexpected last dim {segs.shape[-1]} (expected {args.expected_dim})",
                  file=sys.stderr)
        if args.pool == "mean":
            v = segs.mean(axis=0)
        elif args.pool == "first":
            v = segs[0]
        elif args.pool == "center":
            v = segs[segs.shape[0] // 2]
        out[tid] = v.astype(np.float32)
        if i % 500 == 0:
            print(f"  [{i}/{len(files)}] {len(out)} kept ({(i / max(1, time.time() - t0)):.1f}/s)")

    if not out:
        sys.exit("No embeddings collected. Check --clews-dir and --allow-keys filters.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    dim = next(iter(out.values())).shape[0]
    print(f"Done. {len(out)} tracks, dim={dim}, {bad} failures. Wrote {out_path}")


if __name__ == "__main__":
    main()

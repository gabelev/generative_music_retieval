"""EnCodec 24kHz RVQ code extraction.

For each track in a CSV with track_id + filepath columns:
  - load audio, resample to 24kHz mono, center-clip 30 sec
  - encode with EnCodec at target bandwidth (default 3.0 kbps -> 4 codebooks)
  - save the [n_codebooks, n_frames] code matrix to a single NPZ keyed by track_id

Aggregation to a 3-token SemID is a separate step (semantic_ids/encodec_to_semid.py),
because we want to compare aggregation strategies after the fact without re-running
the (slower) audio encode.

Bandwidth -> codebook count for EnCodec 24kHz:
  1.5 kbps -> 2 codebooks   (NOT enough for our 3-level SemID)
  3.0 kbps -> 4 codebooks   (default: use first 3)
  6.0 kbps -> 8 codebooks
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio

TARGET_SR = 24000
CLIP_SAMPLES = TARGET_SR * 30


def load_audio_clip(path: Path) -> torch.Tensor:
    """Load -> mono -> resample 24kHz -> center 30-sec clip. Returns [1, T] tensor."""
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    if wav.shape[1] > CLIP_SAMPLES:
        start = (wav.shape[1] - CLIP_SAMPLES) // 2
        wav = wav[:, start : start + CLIP_SAMPLES]
    elif wav.shape[1] < CLIP_SAMPLES:
        pad = CLIP_SAMPLES - wav.shape[1]
        wav = torch.nn.functional.pad(wav, (0, pad))
    return wav.contiguous().float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cliques-csv", required=True)
    parser.add_argument("--out", required=True,
                        help=".npz keyed by track_id; values are [n_codebooks, n_frames] int arrays.")
    parser.add_argument("--bandwidth", type=float, default=3.0,
                        help="Target EnCodec bandwidth in kbps. 3.0 -> 4 codebooks (we use first 3).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--track-id-col", default="track_id")
    parser.add_argument("--filepath-col", default="filepath")
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from encodec import EncodecModel  # imported lazily — heavy import

    if args.bandwidth < 3.0:
        print(f"warn: bandwidth {args.bandwidth} kbps gives <3 codebooks; SemID requires 3.",
              file=sys.stderr)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Device: {device}, bandwidth: {args.bandwidth} kbps")

    print("Loading EnCodec 24kHz")
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(args.bandwidth)
    model = model.to(device).eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, np.ndarray] = {}
    if out_path.exists():
        with np.load(out_path) as npz:
            existing = {k: npz[k] for k in npz.files}
        print(f"Resume: {len(existing)} entries already in {out_path}")

    with open(args.cliques_csv) as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]
    todo = [r for r in rows if r[args.track_id_col] not in existing]
    print(f"Loaded {len(rows)} rows; {len(todo)} to extract")
    if not todo:
        return

    codes_out = dict(existing)
    counts = {"ok": 0, "load_error": 0}
    t0 = time.time()

    for i in range(0, len(todo), args.batch_size):
        batch = todo[i : i + args.batch_size]
        wavs: list[torch.Tensor] = []
        track_ids: list[str] = []
        for r in batch:
            try:
                wavs.append(load_audio_clip(Path(r[args.filepath_col])))
                track_ids.append(r[args.track_id_col])
            except Exception as e:
                counts["load_error"] += 1
                print(f"  load fail {r[args.track_id_col]}: {e!r}", file=sys.stderr)

        if not wavs:
            continue

        # All clips are exactly CLIP_SAMPLES long after load_audio_clip, so we can stack.
        x = torch.stack(wavs, dim=0).to(device)  # [B, 1, T]
        with torch.no_grad():
            encoded = model.encode(x)
        # encoded is a list of (codes, scale) tuples; for unconditional 24k model -> 1 frame chunk
        # codes: [B, n_codebooks, n_frames]
        codes_all = torch.cat([c for c, _ in encoded], dim=-1)  # concat across time chunks
        codes_all = codes_all.cpu().numpy().astype(np.int16)

        for j, tid in enumerate(track_ids):
            codes_out[tid] = codes_all[j]
            counts["ok"] += 1

        done = min(i + args.batch_size, len(todo))
        if done % args.save_every < args.batch_size or done == len(todo):
            np.savez(out_path, **codes_out)
            rate = done / max(1, time.time() - t0)
            print(f"[{done}/{len(todo)}] {counts}  ({rate:.2f}/s)  saved {len(codes_out)} -> {out_path.name}")

    np.savez(out_path, **codes_out)
    print(f"Done. Wrote {len(codes_out)} entries to {out_path}")
    print(f"Counts: {counts}")


if __name__ == "__main__":
    main()

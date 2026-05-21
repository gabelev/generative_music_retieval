"""MERT-v1-95M embedding extraction.

For each track in a CSV with `track_id` and `filepath` columns:
  - load audio, resample to 24kHz mono
  - take 30-sec center clip
  - run through frozen MERT-v1-95M
  - mean-pool hidden states from layers 7..12 across time -> 768-d vector

Saves a NPZ at --out, keyed by track_id. Resume-safe: existing NPZ entries
are loaded at startup and not recomputed.
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
from transformers import AutoModel, Wav2Vec2FeatureExtractor

MODEL_NAME = "m-a-p/MERT-v1-95M"
TARGET_SR = 24000
CLIP_SAMPLES = TARGET_SR * 30
LAYERS = list(range(7, 13))  # 7..12 inclusive


def load_audio_clip(path: Path) -> np.ndarray:
    """Load -> mono -> resample to 24kHz -> center 30-sec clip. Returns float32 1-D array."""
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    wav = wav.squeeze(0)
    if wav.shape[0] > CLIP_SAMPLES:
        start = (wav.shape[0] - CLIP_SAMPLES) // 2
        wav = wav[start : start + CLIP_SAMPLES]
    elif wav.shape[0] < CLIP_SAMPLES:
        pad = CLIP_SAMPLES - wav.shape[0]
        wav = torch.nn.functional.pad(wav, (0, pad))
    return wav.contiguous().float().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cliques-csv", required=True,
                        help="CSV with columns including track_id and filepath.")
    parser.add_argument("--out", required=True,
                        help="Output .npz path; keys are track_ids, values are 768-d arrays.")
    parser.add_argument("--device", default=None,
                        help="cuda | mps | cpu. Default: cuda if available else cpu.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--track-id-col", default="track_id")
    parser.add_argument("--filepath-col", default="filepath")
    parser.add_argument("--save-every", type=int, default=200,
                        help="Periodically flush the NPZ every N tracks.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke-test: only process the first N rows.")
    args = parser.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    print(f"Device: {device}, dtype: {args.dtype}")

    print(f"Loading {MODEL_NAME}")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = model.to(device=device, dtype=dtype).eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: load existing
    existing: dict[str, np.ndarray] = {}
    if out_path.exists():
        with np.load(out_path) as npz:
            existing = {k: npz[k] for k in npz.files}
        print(f"Resume: {len(existing)} embeddings already in {out_path}")

    with open(args.cliques_csv) as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    todo = [r for r in rows if r[args.track_id_col] not in existing]
    print(f"Loaded {len(rows)} rows; {len(todo)} to extract")
    if not todo:
        print("Nothing to do.")
        return

    embeddings = dict(existing)
    counts = {"ok": 0, "load_error": 0}
    t0 = time.time()

    for i in range(0, len(todo), args.batch_size):
        batch = todo[i : i + args.batch_size]
        wavs: list[np.ndarray] = []
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

        inputs = processor(wavs, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        inputs = {k: v.to(device=device) for k, v in inputs.items()}
        if "input_values" in inputs:
            inputs["input_values"] = inputs["input_values"].to(dtype=dtype)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # outputs.hidden_states is tuple of (n_layers + 1) tensors [B, T, 768]
        hidden = torch.stack([outputs.hidden_states[L] for L in LAYERS], dim=0)
        # mean across layers (dim 0), then mean across time (dim 2)
        embed = hidden.mean(dim=0).mean(dim=1)  # [B, 768]
        embed = embed.float().cpu().numpy()
        for j, tid in enumerate(track_ids):
            embeddings[tid] = embed[j]
            counts["ok"] += 1

        done = min(i + args.batch_size, len(todo))
        if done % args.save_every < args.batch_size or done == len(todo):
            np.savez(out_path, **embeddings)
            rate = (done - sum(1 for r in todo[:i] if r[args.track_id_col] in existing)) / max(1, time.time() - t0)
            print(f"[{done}/{len(todo)}] {counts}  ({rate:.2f}/s)  saved {len(embeddings)} -> {out_path.name}")

    np.savez(out_path, **embeddings)
    print(f"Done. Wrote {len(embeddings)} embeddings to {out_path}")
    print(f"Counts: {counts}")
    if counts["ok"] == 0 and len(todo) > 0:
        raise SystemExit(
            f"ERROR: 0/{len(todo)} MERT extractions succeeded. "
            f"Likely missing torchaudio backend — try `apt-get install -y libsndfile1 ffmpeg` "
            f"and `pip install soundfile`."
        )


if __name__ == "__main__":
    main()

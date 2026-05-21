"""Constrained beam-search inference for T5 generative retrieval.

At decoder position i, only `<Li_C*>` tokens are allowed; after n_levels
tokens, EOS is forced. Beams are mapped back to Semantic IDs, which are then
resolved to ranked track_ids via the SemID->track lookup table.

Inputs:
  --ckpt-dir    : output_dir from train.py (has model, tokenizer, run_config.json)
  --test-csv    : a build_splits *_test.csv (queries are deduped)
  --semids-csv  : SemID lookup table for ALL tracks (for SemID -> track resolution)
  --out         : JSON file mapping query_track_id -> ranked [track_id, ...]

Notes:
  - For computing metrics, you also need a cliques CSV. That's an eval-time
    concern (src/evaluation/evaluate.py); inference only emits predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

from src.model.dataset import PREFIX, semid_string, semid_token


def load_run_config(ckpt_dir: Path) -> dict:
    cfg_path = ckpt_dir / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"{cfg_path} missing — was this dir produced by src/model/train.py?"
        )
    with cfg_path.open() as f:
        return json.load(f)


def build_level_token_ids(tokenizer, n_levels: int, codebook_size: int) -> list[list[int]]:
    """Returns [level -> [token_id, ...]] for the constrained decoder mask."""
    out: list[list[int]] = []
    for l in range(n_levels):
        ids = []
        for c in range(codebook_size):
            tid = tokenizer.convert_tokens_to_ids(semid_token(l, c))
            if tid == tokenizer.unk_token_id:
                raise RuntimeError(f"SemID token {semid_token(l, c)} not in vocab.")
            ids.append(tid)
        out.append(ids)
    return out


def parse_semid_token(token: str) -> tuple[int, int] | None:
    """'<L1_C42>' -> (1, 42); returns None on non-match."""
    if not (token.startswith("<L") and token.endswith(">") and "_C" in token):
        return None
    try:
        lvl_str, code_str = token[2:-1].split("_C")
        return int(lvl_str), int(code_str)
    except (ValueError, IndexError):
        return None


def beams_to_semids(
    sequences: torch.Tensor, tokenizer, n_levels: int
) -> list[tuple[int, ...]]:
    """Convert beam output token sequences (each [seq_len]) to SemID tuples."""
    semids: list[tuple[int, ...]] = []
    for seq in sequences:
        codes: list[int] = []
        for tok_id in seq.tolist():
            tok = tokenizer.convert_ids_to_tokens(tok_id)
            parsed = parse_semid_token(tok) if tok else None
            if parsed is not None and parsed[0] == len(codes):
                codes.append(parsed[1])
            if len(codes) == n_levels:
                break
        if len(codes) == n_levels:
            semids.append(tuple(codes))
    return semids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--semids-csv", required=True,
                        help="Full SemID lookup: track_id, c1, c2, c3 (all dataset tracks).")
    parser.add_argument("--out", required=True,
                        help="JSON output: {query_track_id: [track_id, ...]} ranked.")
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--num-beam-groups", type=int, default=4,
                        help="If >1, use diverse beam search (Vijayakumar et al.). "
                             "Prevents beams from collapsing to the same high-logit path — "
                             "critical when the underlying model exhibits mode collapse.")
    parser.add_argument("--diversity-penalty", type=float, default=0.5,
                        help="Diverse beam search diversity strength. Ignored if num_beam_groups=1.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    ckpt = Path(args.ckpt_dir)
    cfg = load_run_config(ckpt)
    n_levels = int(cfg["n_levels"])
    codebook_size = int(cfg["codebook_size"])

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    print(f"Loading model + tokenizer from {ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = T5ForConditionalGeneration.from_pretrained(str(ckpt)).to(device).eval()

    level_token_ids = build_level_token_ids(tokenizer, n_levels, codebook_size)
    level_token_sets = [set(ids) for ids in level_token_ids]
    eos_id = tokenizer.eos_token_id

    def prefix_allowed_tokens_fn(batch_id: int, sent: torch.Tensor) -> list[int]:
        # sent[0] is decoder_start_token (T5 pad); position is len(sent) - 1.
        pos = sent.shape[0] - 1
        if 0 <= pos < n_levels:
            return level_token_ids[pos]
        return [eos_id]

    # Build SemID -> track_id lookup (for resolving beams to candidate tracks)
    semid_to_tracks: dict[tuple[int, ...], list[str]] = defaultdict(list)
    with open(args.semids_csv) as f:
        for r in csv.DictReader(f):
            key = (int(r["c1"]), int(r["c2"]), int(r["c3"]))
            semid_to_tracks[key].append(r["track_id"])
    print(f"SemID lookup: {len(semid_to_tracks)} unique SemIDs covering "
          f"{sum(len(v) for v in semid_to_tracks.values())} tracks")

    # Deduplicate queries from the test CSV
    queries: dict[str, tuple[int, ...]] = {}
    with open(args.test_csv) as f:
        for r in csv.DictReader(f):
            qid = r["query_track_id"]
            if qid not in queries:
                queries[qid] = (int(r["query_c1"]), int(r["query_c2"]), int(r["query_c3"]))
    print(f"Inference on {len(queries)} unique queries")

    predictions: dict[str, list[str]] = {}
    with torch.no_grad():
        for qid, qsem in tqdm(queries.items()):
            input_text = f"{PREFIX} {semid_string(*qsem)}"
            enc = tokenizer(input_text, return_tensors="pt").to(device)
            gen_kwargs = dict(
                num_beams=args.beam_width,
                num_return_sequences=args.beam_width,
                max_new_tokens=n_levels + 1,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                early_stopping=True,
                do_sample=False,
            )
            if args.num_beam_groups > 1:
                gen_kwargs.update(
                    num_beam_groups=args.num_beam_groups,
                    diversity_penalty=args.diversity_penalty,
                )
            outputs = model.generate(**enc, **gen_kwargs)
            beam_semids = beams_to_semids(outputs, tokenizer, n_levels)

            ranked_tracks: list[str] = []
            seen: set[str] = set()
            for sem in beam_semids:
                for tid in semid_to_tracks.get(sem, []):
                    if tid == qid or tid in seen:
                        continue
                    seen.add(tid)
                    ranked_tracks.append(tid)
                    if len(ranked_tracks) >= args.top_k:
                        break
                if len(ranked_tracks) >= args.top_k:
                    break
            predictions[qid] = ranked_tracks

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(predictions, f, indent=2)
    n_with_results = sum(1 for v in predictions.values() if v)
    print(f"Wrote {out_path}: {len(predictions)} queries, "
          f"{n_with_results} have at least one prediction.")


if __name__ == "__main__":
    main()

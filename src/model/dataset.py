"""Dataset + tokenizer utilities for T5 generative retrieval.

Input  : "retrieve cover: <L0_C42> <L1_C17> <L2_C203>"
Target :                  "<L0_C88> <L1_C3>  <L2_C156>"

The Semantic ID tokens are added to T5's vocabulary as single tokens via
`tokenizer.add_tokens(...)`. The model's input/output embeddings are then
resized to match (done in train.py).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

PREFIX = "retrieve cover:"


def semid_token(level: int, code: int) -> str:
    return f"<L{level}_C{code}>"


def semid_string(c1: int, c2: int, c3: int) -> str:
    return f"{semid_token(0, c1)} {semid_token(1, c2)} {semid_token(2, c3)}"


def build_semid_vocab(n_levels: int, codebook_size: int) -> list[str]:
    """Return the list of SemID tokens to add to a T5 tokenizer."""
    return [semid_token(l, c) for l in range(n_levels) for c in range(codebook_size)]


def extend_tokenizer_with_semids(
    tokenizer: PreTrainedTokenizerBase,
    n_levels: int,
    codebook_size: int,
) -> int:
    """Adds SemID tokens (if not already present). Returns number of new tokens added."""
    new_tokens = build_semid_vocab(n_levels, codebook_size)
    existing = set(tokenizer.get_vocab().keys())
    to_add = [t for t in new_tokens if t not in existing]
    if to_add:
        tokenizer.add_tokens(to_add, special_tokens=False)
    return len(to_add)


class SemIDPairsDataset(Dataset):
    """Reads a build_splits CSV and emits (input_ids, attention_mask, labels) per row.

    CSV columns: clique_id, query_track_id, target_track_id,
                 query_c1..c3, target_c1..c3
    """

    def __init__(
        self,
        csv_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_input_len: int = 16,
        max_target_len: int = 8,
    ):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        with open(csv_path) as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.rows[idx]
        query = semid_string(int(r["query_c1"]), int(r["query_c2"]), int(r["query_c3"]))
        target = semid_string(int(r["target_c1"]), int(r["target_c2"]), int(r["target_c3"]))
        input_text = f"{PREFIX} {query}"

        enc = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tgt = self.tokenizer(
            target,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = tgt.input_ids.squeeze(0)
        # mask pads so loss ignores them
        labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)

        return {
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "labels": labels,
        }

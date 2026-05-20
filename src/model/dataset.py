"""PyTorch Dataset for (query, target_semid) training pairs.

Two input modes:
  - semid:          query is the query track's SemID as token IDs
  - mert_projection: query is the 768-d MERT embedding (continuous)
"""
from __future__ import annotations

from typing import Literal

# TODO: implement HF-tokenizer-compatible Dataset; resolve SemID -> token IDs;
# for mert_projection mode, return embedding + encoder_attention_mask.


class CoverSongPairs:
    def __init__(self, split_csv: str, mode: Literal["semid", "mert_projection"] = "semid"):
        raise NotImplementedError

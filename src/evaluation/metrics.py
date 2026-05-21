"""CSI retrieval metrics: MR1, MRR, MAP, Recall@k.

A predicted track_id is "relevant" if it belongs to the same clique as the
query (and isn't the query itself).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def _relevance_positions(
    predictions: list[str], query_id: str, query_clique: str, clique_of: dict[str, str],
) -> list[int]:
    """1-based positions of relevant predictions (excluding the query itself)."""
    positions: list[int] = []
    for rank, pid in enumerate(predictions, 1):
        if pid == query_id:
            continue
        if clique_of.get(pid) == query_clique:
            positions.append(rank)
    return positions


def mean_rank_1(
    predictions: dict[str, list[str]], clique_of: dict[str, str],
) -> float:
    ranks: list[float] = []
    for qid, preds in predictions.items():
        qc = clique_of.get(qid)
        if qc is None:
            continue
        pos = _relevance_positions(preds, qid, qc, clique_of)
        ranks.append(float(pos[0]) if pos else float(len(preds) + 1))
    return float(np.mean(ranks)) if ranks else float("nan")


def mrr(predictions: dict[str, list[str]], clique_of: dict[str, str]) -> float:
    rrs: list[float] = []
    for qid, preds in predictions.items():
        qc = clique_of.get(qid)
        if qc is None:
            continue
        pos = _relevance_positions(preds, qid, qc, clique_of)
        rrs.append(1.0 / pos[0] if pos else 0.0)
    return float(np.mean(rrs)) if rrs else float("nan")


def mean_average_precision(
    predictions: dict[str, list[str]], clique_of: dict[str, str],
) -> float:
    aps: list[float] = []
    for qid, preds in predictions.items():
        qc = clique_of.get(qid)
        if qc is None:
            continue
        pos = _relevance_positions(preds, qid, qc, clique_of)
        if not pos:
            aps.append(0.0)
            continue
        # AP = mean over relevant positions of (precision@i)
        ap_terms = [(i + 1) / r for i, r in enumerate(pos)]
        aps.append(float(np.mean(ap_terms)))
    return float(np.mean(aps)) if aps else float("nan")


def recall_at_k(
    predictions: dict[str, list[str]], clique_of: dict[str, str], ks: Iterable[int],
) -> dict[int, float]:
    out: dict[int, list[float]] = {k: [] for k in ks}
    for qid, preds in predictions.items():
        qc = clique_of.get(qid)
        if qc is None:
            continue
        pos = _relevance_positions(preds, qid, qc, clique_of)
        for k in out:
            out[k].append(1.0 if pos and pos[0] <= k else 0.0)
    return {k: float(np.mean(v)) if v else float("nan") for k, v in out.items()}


def compute_all(
    predictions: dict[str, list[str]],
    clique_of: dict[str, str],
    ks: Iterable[int] = (1, 5, 10),
) -> dict:
    return {
        "n_queries": sum(1 for q in predictions if q in clique_of),
        "MR1": mean_rank_1(predictions, clique_of),
        "MRR": mrr(predictions, clique_of),
        "MAP": mean_average_precision(predictions, clique_of),
        **{f"Recall@{k}": v for k, v in recall_at_k(predictions, clique_of, ks).items()},
    }

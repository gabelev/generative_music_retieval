"""CSI retrieval metrics: MR1, MRR, MAP, NAR, Recall@k.

A predicted track_id is "relevant" if it belongs to the same clique as the
query (and isn't the query itself).
"""
from __future__ import annotations

from collections import defaultdict
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


def normalized_average_rank(
    predictions: dict[str, list[str]],
    clique_of: dict[str, str],
    n_candidates: int,
    unfound_rank: int | None = None,
) -> float:
    """Normalized Average Rank per CLEWS (Serra et al. 2025, Appendix B).

      NAR_q = 100 / (M * (n_candidates - M)) * sum_i (rank(m_i) - i)

    over sorted true-positive ranks m_1 <= m_2 <= ... <= m_M, with 1-indexed i.
    Lower is better; 0 = all positives at the top, 100 = all at the bottom.

    True positives outside the prediction list (we keep top-K only) get
    `unfound_rank`. Default: one past the prediction list (worst observable);
    we recommend `unfound_rank = len(preds) + 1` or `n_candidates` (conservative).
    The choice is documented in the paper for reproducibility.
    """
    # Pre-index clique membership so we can enumerate every true positive per query
    by_clique: dict[str, list[str]] = defaultdict(list)
    for tid, cid in clique_of.items():
        by_clique[cid].append(tid)

    scores: list[float] = []
    for qid, preds in predictions.items():
        qc = clique_of.get(qid)
        if qc is None:
            continue
        positives = [t for t in by_clique[qc] if t != qid]
        M = len(positives)
        if M == 0 or M >= n_candidates:
            continue
        # 1-indexed rank of each positive within the prediction list, with the query removed.
        rank_of: dict[str, int] = {}
        rank = 0
        for p in preds:
            if p == qid:
                continue
            rank += 1
            if p in rank_of:
                continue
            rank_of[p] = rank
        default = unfound_rank if unfound_rank is not None else len(preds) + 1
        ranks = sorted(rank_of.get(p, default) for p in positives)
        score = sum(r - (i + 1) for i, r in enumerate(ranks))
        scores.append(100.0 * score / (M * (n_candidates - M)))
    return float(np.mean(scores)) if scores else float("nan")


def compute_all(
    predictions: dict[str, list[str]],
    clique_of: dict[str, str],
    ks: Iterable[int] = (1, 5, 10),
    n_candidates: int | None = None,
) -> dict:
    out: dict = {
        "n_queries": sum(1 for q in predictions if q in clique_of),
        "MR1": mean_rank_1(predictions, clique_of),
        "MRR": mrr(predictions, clique_of),
        "MAP": mean_average_precision(predictions, clique_of),
        **{f"Recall@{k}": v for k, v in recall_at_k(predictions, clique_of, ks).items()},
    }
    if n_candidates is not None:
        out["NAR"] = normalized_average_rank(predictions, clique_of, n_candidates)
        out["n_candidates"] = n_candidates
    return out

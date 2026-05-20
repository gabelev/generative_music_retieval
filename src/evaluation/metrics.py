"""CSI retrieval metrics: MR1, MRR, MAP, Recall@k.

All metrics treat any track in the query's clique (excluding the query itself)
as a relevant hit.
"""
from __future__ import annotations

# TODO: implement per CLAUDE.md Task 13.


def mean_rank_1(predictions, ground_truth_cliques):
    raise NotImplementedError


def mrr(predictions, ground_truth_cliques):
    raise NotImplementedError


def mean_average_precision(predictions, ground_truth_cliques):
    raise NotImplementedError


def recall_at_k(predictions, ground_truth_cliques, k):
    raise NotImplementedError

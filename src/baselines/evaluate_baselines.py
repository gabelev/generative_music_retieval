"""Run all non-generative baselines (bi-encoder + any CQT-based) and emit
their MR1/MRR/MAP/Recall@k under the same eval harness used for T5.
"""
from __future__ import annotations

import argparse

# TODO: orchestrate baseline runs and dispatch to evaluation.metrics.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()

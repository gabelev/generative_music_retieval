# Generative Music Retrieval

Research repo for **Generative Retrieval for Cover Song Identification via
Audio-Derived Semantic IDs** (ICECCME 2026, IEEE).

A T5 model is trained to autoregressively generate a discrete **Semantic ID**
of a cover version given a query track, replacing the traditional
encode-and-search-the-index pipeline with a single generative model whose
parameters *are* the index.

## Pipeline

```
audio -> MERT-v1-95M (frozen) -> 768-d embedding -> RQ-VAE -> <c1><c2><c3>
                                                          |
                                                          v
                            T5-small (fine-tuned) -> <c1'><c2'><c3'>  (a cover)
```

## Conditions compared

| Condition       | SemID source                                  |
|-----------------|-----------------------------------------------|
| `random_ids`    | uniform random 3-token tuples (baseline)      |
| `mert_ids`      | RQ-VAE quantization of MERT-v1-95M embeddings |
| `encodec_ids`   | EnCodec 24kHz RVQ codes (first 3 codebooks)   |

Plus a MERT bi-encoder + FAISS retrieval baseline.

## Datasets

- **Covers80** (dev): 80 cliques x 2 versions = 160 tracks
- **Discogs-VI-YT subset** (main eval): ~5-10K tracks across ~2-3K cliques,
  audio crawled via yt-dlp from YouTube IDs in the Discogs-VI metadata

## Layout

```
configs/      base + per-condition YAML
src/data/     dataset download + YouTube crawl + clique-level splits
src/features/ MERT + EnCodec extraction
src/semantic_ids/  RQ-VAE training, EnCodec aggregation, random IDs, diagnostics
src/model/    T5 fine-tuning + constrained beam-search inference
src/baselines/ bi-encoder + FAISS
src/evaluation/ MR1, MRR, MAP, Recall@k, t-SNE, prefix-overlap figures
paper/        IEEE LaTeX source + figures
scripts/      run_covers80_dev.sh (quick), run_full_pipeline.sh (main)
docs/         CLAUDE.md (task breakdown), ICECCME_Plan_Final.md (paper plan)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
# Dev cycle on Covers80 (sanity check pipeline)
bash scripts/run_covers80_dev.sh

# Main paper run on Discogs-VI-YT subset
bash scripts/run_full_pipeline.sh
```

Progress on individual tasks is tracked in chaos-dimension under the
`generative-retrieval` workstream.

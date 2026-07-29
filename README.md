# Generative Music Retrieval

Official code for **Generative Retrieval for Cover Song Identification via
Audio-Derived Semantic IDs**, accepted at the International Conference on
Electrical, Computer, Communications and Mechatronics Engineering
(**ICECCME 2026**, IEEE; 15–17 October 2026, Bali, Indonesia).

Gabriel L. Levine (Graduate Center, CUNY; glevine@gc.cuny.edu) and
Sarah Ita Levitan (Hunter College, CUNY).

A T5 model is trained to autoregressively generate a discrete **Semantic ID**
of a cover version given a query track, replacing the traditional
encode-and-search-the-index pipeline with a single generative model whose
parameters *are* the index. This is the first formulation of cover song
identification (CSI) as generative retrieval over a discrete Semantic ID
vocabulary, and it replicates the central TIGER/PLUM finding — that the
Semantic ID construction is the binding constraint — in the audio domain.

## Pipeline

```
query audio (30 s, 24 kHz mono)
        |
        v   offline indexing: audio encoder + quantizer
   [ CLEWS -> k-means RQ-VAE ]   <- best condition
   [ MERT  -> k-means RQ-VAE ]
   [ EnCodec (end-to-end RVQ codes) ]
   [ random (uniform sample) ]
        |
        v
   Semantic ID <c1 c2 c3>
        |
        v   T5-small (fine-tuned) + constrained beam search
   cover Semantic ID -> track-ID lookup
```

Each condition maps a 30-second clip to a 3-token Semantic ID through a
different encoder/quantizer path. A fine-tuned T5-small autoregressively
generates the Semantic ID of a cover, resolved to a track via a lookup table.

## Conditions compared

Four Semantic ID constructions spanning the spectrum of audio representation
quality, with the T5 model and decoding strategy held constant:

| Condition     | SemID source                                                   |
|---------------|----------------------------------------------------------------|
| `random_ids`  | uniform random 3-token tuples, K=256 (no structure, memorize)  |
| `encodec_ids` | EnCodec RVQ codes at 3 kbps, per-codebook majority vote, K=1024 |
| `mert_ids`    | sequential k-means RQ-VAE over MERT-v1-95M embeddings, K=256    |
| `clews_ids`   | sequential k-means RQ-VAE over CLEWS embeddings, K=256          |

Plus a MERT bi-encoder + FAISS (`IndexFlatIP`) retrieval baseline —
the index-and-retrieve paradigm generative retrieval aims to replace.

Note: unlike TIGER/PLUM (which train a neural RQ-VAE with reconstruction and
commitment losses), we use sequential k-means on pre-computed embeddings, which
isolates the effect of the input representation from the quantizer. Results are
therefore a lower bound on what a learned RQ-VAE could achieve.

## Key results

**Generative retrieval beats a strong embedding baseline by more than 3× on a
fixed catalog.** T5-CLEWS reaches **MRR 0.71** (MAP 0.69, R@1 0.70) on a fixed,
pre-indexed 3,192-track Discogs-VI catalog, versus **MRR 0.22** for a MERT
bi-encoder + FAISS (`IndexFlatIP`) retriever over the same pool — replacing the
search index with model parameters.

**A single before-training diagnostic predicts the entire ranking.** Before any
T5 is trained, we measure whether compositional structure survives
quantization: the ratio at which covers share their first Semantic ID token
versus non-covers (within-clique / cross-clique). That ratio rises with the
version-invariance quality of the source encoder — 1.06× → 1.8× → 4.3× → 91× —
and the downstream retrieval MRR follows the *exact same ordering*, turning four
training runs into a claim about *why* the encoder is the binding constraint.

| Condition        | W/C prefix ratio (c1) | Known-catalog MRR |
|------------------|-----------------------|-------------------|
| random           | 1.06×                 | 0.11              |
| EnCodec direct   | 1.8×                  | 0.11              |
| MERT-RQ-VAE      | 4.3×                  | 0.27              |
| CLEWS-RQ-VAE     | **91×**               | **0.71**          |
| Bi-enc. (MERT)   | —                     | 0.22              |

**Known-catalog vs. unseen-song (an honest scoping result).** Because the
model's parameters *are* the index, generative retrieval can only emit Semantic
IDs it saw in training. On a known-catalog split (all song groups indexed, only
query-to-cover pairs held out) T5-CLEWS dominates. On an unseen-song split
(whole groups held out) **every generative condition scores exactly 0.000 on
every metric**, while the bi-encoder — which embeds on the fly — is unaffected.
This is the paper's most useful practical finding: generative retrieval works
for a fixed catalog with a strong source encoder and fails for an open one, so
embedding-based retrieval remains necessary for a growing catalog. Reporting it
scopes the 3× win honestly rather than reporting only the win.

## Datasets

- **Discogs-VI-YT subset** (main eval): audio crawled with yt-dlp (May 2026)
  from YouTube IDs in the Discogs-VI metadata. Of 2,500 sampled groups (12,006
  candidate IDs), 3,413 tracks were recovered; filtering to groups with ≥2
  surviving versions yields **696 song groups across 3,192 tracks**. Audio is
  standardized to 30-second center clips at 24 kHz mono. The known-catalog
  split holds out pairs per group (42,620 train pairs, 2,324 test queries); the
  unseen-song split holds out whole groups (487/104/105 train/val/test).
- **Covers80** (cross-dataset eval): 80 cliques × 2 versions = 160 tracks. No
  Covers80 track appears in training; codebooks trained on Discogs-VI are
  reused, so it serves purely as a structure-transfer evaluation set.

## Layout

```
configs/      base + per-condition YAML (random/encodec/mert/clews)
src/data/     dataset download + YouTube crawl + clique-level splits
src/features/ MERT + EnCodec + CLEWS extraction
src/semantic_ids/  RQ-VAE (k-means) training, EnCodec aggregation, random IDs, diagnostics
src/model/    T5 fine-tuning + constrained beam-search inference
src/baselines/ MERT bi-encoder + FAISS
src/evaluation/ MR1, MRR, MAP, Recall@k, NAR, prefix-overlap + t-SNE figures
paper/        IEEE LaTeX source + figures
scripts/      run_covers80_dev.sh (quick), run_full_pipeline.sh (main), pod runners
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

## Citation

```bibtex
@inproceedings{levine2026generative,
  title     = {Generative Retrieval for Cover Song Identification via Audio-Derived Semantic IDs},
  author    = {Levine, Gabriel L. and Levitan, Sarah Ita},
  booktitle = {Proc. Int. Conf. Electrical, Computer, Communications and Mechatronics Engineering (ICECCME)},
  year      = {2026},
  address   = {Bali, Indonesia},
}
```

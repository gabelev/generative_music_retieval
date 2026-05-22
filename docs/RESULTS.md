# Results

Canonical empirical record for the ICECCME 2026 submission. Each section
records what was measured, when, with which script, against which inputs.
T5 training rows are placeholders pending the GPU-pod run.

---

## 1. Dataset construction

### 1.1 Covers80
- **Source**: LabROSA tarball <http://labrosa.ee.columbia.edu/projects/coversongs/covers80/covers80.tgz> (mtime 2007-08-08; static, no link rot)
- **Script**: `src/data/download_covers80.py`
- **Output**: `data/splits/covers80_cliques.csv`
- **Counts**: 80 cliques × 2 versions = **160 tracks, 0 missing**
- **Audio**: 32 kbps mp3 (legacy quality)

### 1.2 Discogs-VI-YT (Araz et al., ISMIR 2024)
- **Metadata source**: Zenodo record 13983028 → `main.zip` (1.4 GB) → `Discogs-VI-YT-20240701-light.json` (46 MB)
- **Full metadata size**: 493,049 (clique, version, YouTube ID) rows across 98,785 cliques
- **Subset sampled**: 2,500 cliques (≥2 versions, seed 42) → 12,006 candidate YouTube IDs
- **Crawl date**: May 2026 (yt-dlp + ffmpeg, 30-sec center-clip 24kHz mono WAV)
- **Crawl outcome**:

| status | count | share |
|---|---|---|
| ok (audio in HF dataset) | **3,413** | 28.4% |
| video_unavailable (link rot) | 3,404 | 28.3% |
| yt_dlp_failed (mostly "Sign in to confirm you're not a bot") | 5,630 | 46.9% |
| hf_upload_failed (from pre-batching run, retryable) | 93 | 0.8% |

- **Audio storage**: private HF dataset `gabelev/discogs-vi-csi-subset` (`audio/<youtube_id>.wav`)
- **After filtering cliques to ≥2 surviving versions**: **696 usable cliques, 3,192 tracks**
- **Clique-level split** (70/15/15, seeded): train 487, val 104, test 105
- **Total ordered training pairs**: ~42,000

### 1.3 Per-track feature extraction (RunPod, RTX 4090, ~5.3 min wall time)
- **Script**: `scripts/run_extraction_pod.sh`
- **MERT-v1-95M** embeddings (mean-pool layers 7–12, 768-d): `data/embeddings/mert/{covers80,discogs_vi}_embeddings.npz`
- **EnCodec 24kHz @ 3.0 kbps** RVQ codes (4 codebooks, 1024 each): `data/embeddings/encodec/{covers80,discogs_vi}_codes.npz`

---

## 2. Semantic ID construction (3 conditions)

| condition | source | codebook | how |
|---|---|---|---|
| `random_ids` | uniform random | 256 per level | `src/semantic_ids/random_ids.py` |
| `mert_ids` | RQ-VAE on MERT embeddings | 256 per level | `src/semantic_ids/train_rqvae.py` (sequential k-means, 3 levels) |
| `encodec_ids` | first 3 EnCodec codebooks, majority-vote across frames | 1024 per level | `src/semantic_ids/encodec_to_semid.py` |

RQ-VAE codebooks were trained on Discogs-VI MERT embeddings; Covers80 was encoded with those same codebooks for cross-dataset consistency.

### 2.1 Codebook utilization and clash rate

| dataset / condition | clash rate | utilization c1 / c2 / c3 |
|---|---|---|
| Discogs-VI random | 0.0003 | 256 / 256 / 256 (of 256) |
| Discogs-VI MERT-RQ-VAE | 0.014 | 256 / 250 / 256 (of 256) |
| Discogs-VI EnCodec | 0.008 | 578 / 556 / 485 (of 1024) — ~50% used |
| Covers80 random | 0.000 | 114 / 120 / 114 (of 256) |
| Covers80 MERT-RQ-VAE | 0.012 | 75 / 30 / 37 (of 256) |
| Covers80 EnCodec | **0.41** | 57 / 30 / 29 (of 1024) — **degenerate** |

### 2.2 Prefix overlap: do covers share SemID prefixes?

For each condition, we measure **within-clique** vs **cross-clique** prefix-overlap rates. Random IDs are the sanity floor (within ≈ cross expected). MERT/EnCodec should show within > cross if their SemIDs encode compositional structure.

#### Discogs-VI (3,413 tracks, 30,603 within-clique pairs, 100K cross-clique sampled)

| condition | within p1 | cross p1 | ratio c1 | within p12 | cross p12 | ratio c1+c2 |
|---|---|---|---|---|---|---|
| random | 0.0039 | 0.0037 | 1.06× | 3.3e-5 | 1e-5 | 3.3× |
| **MERT-RQ-VAE** | **0.0277** | 0.0064 | **4.3×** | **2.25e-3** | 2e-4 | **11×** |
| EnCodec | 0.0077 | 0.0044 | 1.8× | 3.6e-4 | 5e-5 | 7× |

**Headline:** MERT-RQ-VAE SemIDs encode compositional similarity at 4.3× the random rate at c1 and 11× at c1+c2 — *before any T5 training*. EnCodec carries weaker but real structure.

#### Covers80 (160 tracks, 80 within-clique pairs, 100K cross-clique sampled)

| condition | within p1 | cross p1 | ratio c1 | within p12 | cross p12 |
|---|---|---|---|---|---|
| random | 0.0 | 0.0047 | (sample noise, N=80) | 0 | 0 |
| **MERT-RQ-VAE** | **0.025** | 0.0153 | **1.6×** | 0 | 0.00141 |
| EnCodec | 0.1125 | 0.1125 | **1.0× (no signal)** | 0.10 | 0.091 |

**Two cross-dataset findings:**

1. **MERT structure transfers across datasets** (1.6× at c1). Discogs-VI-trained RQ-VAE codebooks still organize Covers80 with positive within > cross signal.
2. **EnCodec direct codes collapse on low-bitrate (32 kbps mp3) audio.** within ≈ cross prefix overlap, 41% clash rate, codebooks barely used. Direct neural-codec codes are not robust to audio quality heterogeneity — supports the case for the learned RQ-VAE-over-MERT approach.

Reports saved at `runs/{discogs_vi,covers80}_{random,mert,encodec}_semid_report.json`.

---

## 3. Baseline: MERT bi-encoder + FAISS

Frozen MERT embeddings, L2-normalized, retrieved via FAISS `IndexFlatIP` (cosine). Candidate pool = all dataset tracks; queries = test split (or all of Covers80).

- **Script**: `src/baselines/biencoder.py`

| dataset | queries | MR1 | MRR | MAP | Recall@1 | Recall@5 | Recall@10 |
|---|---|---|---|---|---|---|---|
| Discogs-VI test | 390 | 15.05 | 0.193 | 0.178 | 0.146 | 0.246 | 0.308 |
| Covers80 (full set) | 160 | 16.90 | 0.113 | 0.113 | 0.069 | 0.156 | 0.200 |

Reports: `runs/metrics_biencoder_{discogs_vi,covers80}.json`.

**Note:** Covers80 is a structurally harder eval (1 positive per query vs ~3 for Discogs-VI; lower audio quality). The two datasets' numbers are not directly comparable.

---

## 4. T5 generative retrieval

For each SemID condition, T5-small is fine-tuned to autoregressively generate a cover's SemID given the query's SemID, then evaluated with constrained beam search (beam width 20, 4 diverse beam groups, position i restricted to `<Li_C*>` tokens).

- **Training script**: `src/model/train.py` (HuggingFace `Seq2SeqTrainer`, early stop on val loss, patience 3)
- **Inference script**: `src/model/inference.py`
- **Driver**: `scripts/run_t5_training_pod.sh`
- **Evaluation regime**: transductive (DSI/TIGER-style) — the full Discogs-VI corpus is indexed; train/val/test split is on `(query→cover)` pairs (`build_splits.py --split-by pair`).

### 4.1 Two bugs found and fixed (runs 1-2 mode-collapsed)

The first two training runs collapsed (the model emitted a single constant SemID for every query). Root causes:

1. **Embedding init.** HF `resize_token_embeddings` mean-initializes new tokens, so all 768 SemID tokens started *identical* — the encoder produced byte-identical output for every query. Fixed by `_reinit_semid_embeddings()`, which samples each SemID embedding from the original vocabulary's per-dimension Gaussian.
2. **Split design.** The original clique-level split left test cliques entirely unseen; generative retrieval cannot retrieve an un-indexed corpus. Fixed by the transductive pair-split.

Run 3 (LR 3e-4, label smoothing 0, embedding re-init, pair-split) trained correctly: MERT predicts 850 distinct tracks across 2,324 queries (vs. 1 in the collapsed runs).

### 4.2 Discogs-VI in-distribution test (2,324 query tracks)

| Model | MR1 | MRR | MAP | R@1 | R@5 | R@10 |
|---|---|---|---|---|---|---|
| Bi-encoder (MERT + FAISS) | 14.0 | 0.214 | 0.181 | 0.152 | 0.278 | **0.355** |
| T5 random          | 18.1 | 0.112 | 0.100 | 0.095 | 0.122 | 0.148 |
| T5 EnCodec direct  | 17.9 | 0.122 | 0.106 | 0.102 | 0.140 | 0.150 |
| **T5 MERT-RQ-VAE** | **13.3** | **0.271** | **0.254** | **0.237** | **0.311** | 0.345 |

T5 MERT-RQ-VAE beats the bi-encoder on MR1, MRR, MAP, R@1, R@5; loses R@10 by 0.010. The R@1 gain is the headline: 0.237 vs 0.152 (+56% relative). The condition ordering (random ≈ EnCodec ≪ MERT) matches the pre-training SemID prefix-overlap ordering exactly — structured audio-derived IDs are what carries generative retrieval.

### 4.3 Cold-start (inductive) experiment: clique-level split

To measure the transductive boundary directly, we re-ran all 3 conditions with a **clique-level split** (`build_splits.py --split-by clique`): 487 train / 104 val / 105 test cliques, the test cliques *entirely unseen* during training. Same training recipe (embedding re-init, LR 3e-4, etc.).

| Model | MR1 | MRR | MAP | R@1 | R@5 | R@10 |
|---|---|---|---|---|---|---|
| Bi-encoder (MERT+FAISS) | 15.0 | **0.193** | 0.178 | **0.146** | 0.246 | **0.308** |
| T5 random          | 20.6 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| T5 EnCodec direct  | 15.4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| T5 MERT-RQ-VAE     | 17.7 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

All 3 generative conditions score **exactly 0.000** on every retrieval metric. The models are *not* collapsed — they emit 29-59 distinct tracks across 390 queries — but every emitted SemID belongs to a *training* clique. The model cannot reach the 105 held-out cliques it never indexed. The bi-encoder, having no training phase, retrieves held-out-clique covers just as well as in-distribution ones (MRR 0.193).

**Conclusion**: generative retrieval is strictly transductive — it wins (MRR 0.271 vs 0.214) when the corpus is fixed and indexed, and fails completely (0.000) on new songs. The bi-encoder is the robust choice for an open/growing catalog. This boundary is the paper's central scoping result.

### 4.4 Error analysis: graceful degradation and the acoustic-similarity failure mode

For the transductive T5 MERT-RQ-VAE model, we split test queries into hits (867) and misses (1,457) and measured MERT cosine similarity (`src/evaluation/error_analysis.py`):

| comparison | mean cos sim |
|---|---|
| query → true cover | 0.886 |
| query → T5 top-1 prediction on a MISS | **0.900** |
| query → random track | 0.869 |

Two findings: (1) **graceful degradation** — T5's wrong predictions (0.900) are far more acoustically similar to the query than random tracks (0.869), so the model fails by retrieving plausible near-misses, not noise. (2) **The failure mode** — T5's misses are *more* acoustically similar to the query than the true covers (0.900 vs 0.886). The model fails by retrieving acoustic near-duplicates. This is direct evidence that MERT-RQ-VAE SemIDs encode acoustic surface similarity, which only approximates compositional identity; covers with heavy reinterpretation (key/tempo/instrumentation changes) are acoustically far from the query and systematically missed. Figure at `paper/figures/error_analysis.pdf`.

### 4.5 Covers80 cross-dataset: not applicable to generative retrieval

Consistent with §4.3, a Discogs-VI-trained model cannot retrieve Covers80 tracks (a different, un-indexed catalog) — all cross-dataset generative-retrieval lookups miss. Covers80 therefore serves only as (a) the cross-dataset SemID-structure check (§2.2) and (b) the bi-encoder cross-dataset baseline (§3).

---

## 5. Provenance

- Crawl date: 2026-05-20 / 21
- MERT/EnCodec extraction: 2026-05-21 (RunPod RTX 4090, 5.3 min wall)
- SemID generation + RQ-VAE + diagnostics + bi-encoder baselines: 2026-05-21 (local)
- T5 training: _pending_
- Seeds: 42 everywhere (data subset, RQ-VAE, T5 training, split assignment)

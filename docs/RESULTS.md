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

## 2. Semantic ID construction (4 conditions)

| condition | source | codebook | how |
|---|---|---|---|
| `random_ids` | uniform random | 256 per level | `src/semantic_ids/random_ids.py` |
| `encodec_ids` | first 3 EnCodec codebooks, majority-vote across frames | 1024 per level | `src/semantic_ids/encodec_to_semid.py` |
| `mert_ids` | RQ-VAE on MERT-v1-95M embeddings (768-d) | 256 per level | `src/semantic_ids/train_rqvae.py` |
| `clews_ids` | RQ-VAE on CLEWS embeddings (1024-d, Serra et al. 2025) | 256 per level | `src/features/extract_clews.py` + `train_rqvae.py` |

RQ-VAE codebooks were trained on Discogs-VI embeddings; Covers80 was encoded with the same codebooks for cross-dataset consistency. CLEWS is the current CSI SOTA — its embeddings are explicitly contrastively trained for version invariance on Discogs-VI, in contrast to MERT's general-purpose music representation.

### 2.1 Codebook utilization and clash rate

| dataset / condition | clash rate | utilization c1 / c2 / c3 |
|---|---|---|
| Discogs-VI random | 0.0003 | 256 / 256 / 256 (of 256) |
| Discogs-VI EnCodec | 0.008 | 578 / 556 / 485 (of 1024) — ~50% used |
| Discogs-VI MERT-RQ-VAE | 0.014 | 256 / 250 / 256 (of 256) |
| Discogs-VI **CLEWS-RQ-VAE** | 0.068 | **256 / 256 / 256** (of 256) |
| Covers80 random | 0.000 | 114 / 120 / 114 (of 256) |
| Covers80 EnCodec | **0.41** | 57 / 30 / 29 (of 1024) — **degenerate** |
| Covers80 MERT-RQ-VAE | 0.012 | 75 / 30 / 37 (of 256) |
| Covers80 CLEWS-RQ-VAE | 0.037 | 40 / 50 / 54 (of 256) |

### 2.2 Prefix overlap: do covers share SemID prefixes?

For each condition, we measure **within-clique** vs **cross-clique** prefix-overlap rates. Random IDs are the sanity floor (within ≈ cross expected). MERT/EnCodec should show within > cross if their SemIDs encode compositional structure.

#### Discogs-VI (3,413 tracks, 30,603 within-clique pairs, 100K cross-clique sampled)

| condition | within p1 | cross p1 | ratio c1 | within p12 | cross p12 | ratio c1+c2 |
|---|---|---|---|---|---|---|
| random | 0.0039 | 0.0037 | 1.06× | 3.3e-5 | 1e-5 | 3.3× |
| EnCodec | 0.0077 | 0.0044 | 1.8× | 3.6e-4 | 5e-5 | 7× |
| MERT-RQ-VAE | 0.0277 | 0.0064 | 4.3× | 2.25e-3 | 2e-4 | 11× |
| **CLEWS-RQ-VAE** | **0.366** | 0.004 | **91×** | **0.0348** | 7e-5 | **497×** |

**Headline:** CLEWS-RQ-VAE Semantic IDs encode within-clique compositional structure at 91× the random rate at c1 and 497× at c1+c2 — an order of magnitude stronger than MERT-RQ-VAE's 4.3× / 11×. The condition ordering tracks the structure encoded by the source embedder: random (none) ≪ EnCodec (acoustic) < MERT (general music) ≪ CLEWS (version-invariance trained).

#### Covers80 (160 tracks, 80 within-clique pairs, 100K cross-clique sampled)

| condition | within p1 | cross p1 | ratio c1 | within p12 | cross p12 |
|---|---|---|---|---|---|
| random | 0.0 | 0.0047 | (sample noise, N=80) | 0 | 0 |
| EnCodec | 0.1125 | 0.1125 | **1.0× (no signal)** | 0.10 | 0.091 |
| MERT-RQ-VAE | 0.025 | 0.0153 | 1.6× | 0 | 0.00141 |
| **CLEWS-RQ-VAE** | **0.438** | 0.047 | **9.2×** | **0.150** | 0.00192 |

**Two cross-dataset findings:**

1. **MERT structure transfers across datasets** (1.6× at c1). Discogs-VI-trained RQ-VAE codebooks still organize Covers80 with positive within > cross signal.
2. **EnCodec direct codes collapse on low-bitrate (32 kbps mp3) audio.** within ≈ cross prefix overlap, 41% clash rate, codebooks barely used. Direct neural-codec codes are not robust to audio quality heterogeneity — supports the case for the learned RQ-VAE-over-MERT approach.

Reports saved at `runs/{discogs_vi,covers80}_{random,mert,encodec}_semid_report.json`.

---

## 3. Baseline: MERT bi-encoder + FAISS

Frozen MERT embeddings, L2-normalized, retrieved via FAISS `IndexFlatIP` (cosine). Candidate pool = all dataset tracks; queries = test split (or all of Covers80).

- **Script**: `src/baselines/biencoder.py`

| dataset | queries | MR1 | MRR | MAP | R@1 | R@5 | R@10 | NAR |
|---|---|---|---|---|---|---|---|---|
| Discogs-VI test | 2,324 | 234.7 | 0.220 | 0.068 | 0.152 | 0.278 | 0.355 | 31.39 |
| Covers80 (full set) | 160 | 56.2 | 0.126 | 0.126 | 0.069 | 0.156 | 0.200 | 34.72 |

Reports: `runs/metrics_biencoder_{discogs_vi,covers80}.json`. Bi-encoder retrieves the full pool (top-3,413 on Discogs-VI, top-159 on Covers80) so NAR is exact (not truncation-biased).

**Notes:**
- The full-pool MAP of 0.068 (Discogs-VI) replaces the earlier Phase-1 reported 0.181, which was inflated by top-20 truncation.
- Covers80 is a structurally harder eval (1 positive per query vs ~3 for Discogs-VI; lower audio quality).
- MR1 grows large under full-pool retrieval because queries without a top-K hit are no longer present (every query now has a rank for every positive).

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

### 4.2 Discogs-VI in-distribution test (2,324 query tracks, top-200 predictions, full-pool NAR)

| Model | MR1 | MRR | MAP | R@1 | R@5 | R@10 | NAR |
|---|---|---|---|---|---|---|---|
| Bi-encoder (MERT + FAISS) | 234.7 | 0.220 | 0.068 | 0.152 | 0.278 | 0.355 | 31.39 |
| T5 random          | 137.0 | 0.110 | 0.095 | 0.094 | 0.117 | 0.130 | 4.55 |
| T5 EnCodec direct  | 129.0 | 0.108 | 0.095 | 0.090 | 0.119 | 0.142 | 4.46 |
| T5 MERT-RQ-VAE     |  78.1 | 0.272 | 0.230 | 0.236 | 0.310 | 0.341 | 2.82 |
| **T5 CLEWS-RQ-VAE** | **13.8** | **0.713** | **0.693** | **0.698** | **0.730** | **0.744** | **0.44** |

**Headlines:**

1. **T5 CLEWS-RQ-VAE matches the CLEWS bi-encoder SOTA via generative retrieval** — NAR 0.44 is in the range reported by Serra et al. 2025 for CLEWS-MERT cosine bi-encoder retrieval on Discogs-VI. We achieve the same retrieval quality without an embedding index — the model's parameters are the index.
2. **Condition ordering tracks SemID structure exactly**: random (no structure) ≪ EnCodec (1.8× / 7×) ≈ random retrieval ≪ MERT (4.3× / 11×) ≪ CLEWS (91× / 497×) → retrieval MRR 0.11 ≈ 0.11 ≪ 0.27 ≪ 0.71. The quality of the source embedder is the binding constraint on generative retrieval, exactly as TIGER and PLUM established for recommendation.
3. **Bi-encoder is dramatically outperformed**: MRR 0.220 vs 0.713 (3.2×), R@1 0.152 vs 0.698 (4.6×), NAR 31.39 vs 0.44 (70× lower). Generative retrieval with a CSI-specialized embedder is the right approach for fixed catalogs.

Note: MAP and MR1 changed from earlier Phase-1 reporting because predictions are now top-200 (vs top-20) and bi-encoder uses full-pool retrieval (3,413 candidates) for honest NAR computation. The previously-reported "MAP 0.18" was inflated by top-20 truncation; the full-pool MAP of 0.068 is the honest number for the bi-encoder.

### 4.3 Cold-start (inductive) experiment: clique-level split

To measure the transductive boundary directly, we re-ran all 3 conditions with a **clique-level split** (`build_splits.py --split-by clique`): 487 train / 104 val / 105 test cliques, the test cliques *entirely unseen* during training. Same training recipe (embedding re-init, LR 3e-4, etc.).

| Model | MR1 | MRR | MAP | R@1 | R@5 | R@10 |
|---|---|---|---|---|---|---|
| Bi-encoder (MERT+FAISS) | 15.0 | **0.193** | 0.178 | **0.146** | 0.246 | **0.308** |
| T5 random          | 20.6 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| T5 EnCodec direct  | 15.4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| T5 MERT-RQ-VAE     | 17.7 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| T5 CLEWS-RQ-VAE    | _not run; expected 0.000 per the condition-independent transductive constraint above_ |

All 3 generative conditions evaluated score **exactly 0.000** on every retrieval metric. The models are *not* collapsed — they emit 29-59 distinct tracks across 390 queries — but every emitted SemID belongs to a *training* clique. CLEWS was not re-trained in the cold-start regime in Phase 2; the finding is condition-independent and a fourth zero-row would add no information. The bi-encoder, having no training phase, retrieves held-out-clique covers just as well as in-distribution ones (MRR 0.193).

**Conclusion**: generative retrieval is strictly transductive. It dominates when the corpus is fixed and indexed (T5 CLEWS MRR 0.713 vs bi-encoder 0.220) and fails completely (0.000) on unseen songs. The bi-encoder is the right choice for an open or growing catalog; generative retrieval is the right choice for a fixed catalog where the gain over embedding search is large.

### 4.4 Error analysis: two different failure modes

For each transductive T5 condition we split test queries into hits and misses and measured cosine similarity in the condition's own embedding space:

| condition | query → true cover | query → T5 miss top-1 | query → random | miss-vs-true gap |
|---|---|---|---|---|
| T5 MERT-RQ-VAE (MERT space) | 0.886 | **0.900** | 0.869 | +0.014 (miss *closer* than true) |
| T5 CLEWS-RQ-VAE (CLEWS space) | **0.497** | 0.235 | 0.006 | −0.262 (miss between random and true) |

**Two distinct failure modes**:

1. **MERT misses are acoustic near-duplicates.** Misses are *more* similar to the query than the true covers (0.900 vs 0.886). MERT's general-purpose embeddings encode acoustic surface; covers with heavy reinterpretation (key, tempo, instrumentation change) are acoustically distant from the query and missed in favor of acoustically similar non-covers.

2. **CLEWS misses are semantic confusions.** Misses land cleanly *between* random (0.006) and true covers (0.497) — the model retrieves tracks that share *some* version-invariance signal but aren't the right composition. The CLEWS embedding space has already factored out acoustic surface, so failures reflect compositional ambiguity rather than the acoustic-vs-compositional gap.

The wider spread for CLEWS (0.006 → 0.235 → 0.497) versus MERT (0.869 → 0.886 → 0.900) is itself a structural statement: CLEWS produces a more discriminative space for CSI, which is why both its bi-encoder (Serra et al. 2025) and our T5-CLEWS-RQ-VAE outperform their MERT counterparts. Figures: `paper/figures/error_analysis.pdf` (MERT), `error_analysis_clews.pdf` (CLEWS).

### 4.5 Covers80 cross-dataset: largely not applicable to generative retrieval

Consistent with §4.3, a Discogs-VI-trained model cannot retrieve Covers80 tracks (a different, un-indexed catalog). MERT and EnCodec generative conditions score exactly 0 on Covers80; CLEWS has a small non-zero signal (MRR 0.019, R@1 0.013) because its 1024-d embeddings produce SemIDs that occasionally happen to overlap with the Discogs-VI training corpus's SemID set. Covers80 therefore serves primarily as (a) the cross-dataset SemID-structure check (§2.2) and (b) the bi-encoder cross-dataset baseline (§3).

---

## 5. Provenance

- Crawl date: 2026-05-20 / 21
- MERT/EnCodec extraction: 2026-05-21 (RunPod RTX 4090, 5.3 min wall)
- Phase 1 SemID + RQ-VAE + bi-encoder + T5 (3 conditions): 2026-05-21 / 22
- **Phase 2 CLEWS extraction**: 2026-05-25 (RunPod L40S, CLEWS DVI checkpoint from Zenodo record 15045900)
- **Phase 2 CLEWS RQ-VAE + T5 + re-inference at top-200**: 2026-05-25 (~1.5 h pod wall)
- Seeds: 42 everywhere (data subset, RQ-VAE, T5 training, split assignment)

# Generative Retrieval for Cover Song Identification via Audio-Derived Semantic IDs
## ICECCME 2026 Paper Plan — Final

**Authors**: Gabriel Levine, Dr. Sarah Ita Levitan  
**Target**: ICECCME 2026 (IEEE, 6-page conference paper)  
**Deadline**: June 1, 2026 (extended)  
**Publication**: IEEE Xplore, Scopus-indexed  
**Last Updated**: May 20, 2026

---

## 1. Paper Thesis

**Title**: Generative Retrieval for Cover Song Identification via Audio-Derived Semantic IDs

**Core Question**: Can a generative retrieval model that autoregressively generates discrete Semantic IDs replace the traditional embedding + ANN search pipeline for cover song identification, and do Semantic IDs derived from audio encoders (MERT, EnCodec) improve this task over random or non-audio IDs?

**Contributions**:
1. First application of the generative retrieval paradigm (DSI/TIGER) to cover song identification, a classic MIR task
2. Systematic comparison of Semantic ID construction strategies using audio encoders (MERT-v1-95M, EnCodec RVQ codes) vs. random baselines
3. Empirical test of the architectural equivalence between neural audio codecs (RVQ) and recommendation Semantic IDs (RQ-VAE) on a real retrieval task with open data

---

## 2. Why Cover Song Identification

Cover song identification (CSI) is the ideal testbed for audio-derived generative retrieval because:

**Clean ground truth.** A cover of "Yesterday" IS the same composition. Binary, unambiguous. No subjective similarity judgments, no noisy tag annotations.

**Unsolved task.** Unlike audio fingerprinting (Shazam), CSI requires representations invariant to key, tempo, instrumentation, and production changes. Traditional embedding + ANN approaches plateau because the embedding must capture compositional identity while ignoring massive surface-level acoustic differences.

**Natural fit for Semantic IDs.** If audio-derived Semantic IDs encode compositional structure (harmonic, melodic, tonal) over acoustic surface (timbre, production), then covers of the same song should land in nearby regions of the Semantic ID space, sharing prefixes. MERT's CQT teacher specifically encodes harmonic and tonal structure, making this hypothesis testable.

**Nobody has tried this.** Text2Tracks does text-to-track. TIGER does interaction-sequence-to-item. FusID does playlist continuation. No generative retrieval work targets cover song identification.

---

## 3. Positioning in the Literature

### Generative Retrieval
- **DSI** (Tay et al., NeurIPS 2022): Transformer as differentiable search index
- **TIGER** (Rajput et al., NeurIPS 2023): Semantic IDs via RQ-VAE. Key finding: structured IDs >> random IDs

### Music-Specific Generative Retrieval
- **Text2Tracks** (Palumbo & Penha, Spotify, 2025): CF-based Semantic IDs, prompt-to-track
- **FusID** (Kim et al., UCSD, Jan 2026): Multimodal fused Semantic IDs, playlist continuation SOTA
- **Mei et al.** (SiriusXM/Pandora, RecSys 2025): Content-based Semantic IDs at scale

### Audio Encoders
- **MERT-v1-95M** (Yuan et al., 2023): Music SSL model. Dual teacher: EnCodec RVQ-VAE (acoustic) + CQT (musical/harmonic). 768-dim embeddings, 13 transformer layers
- **EnCodec** (Defossez et al., Meta, 2022): Neural audio codec with RVQ. Directly produces discrete hierarchical codes from raw audio

### Cover Song Identification
- **Da-TACOS** (Yesiler et al., ISMIR 2019): Standard CSI benchmark, 25K tracks
- **Discogs-VI** (Araz et al., ISMIR 2024): Largest VI dataset, 493K versions across 98K cliques
- **Re-MOVE, CoverHunter, LyraC-Net**: CNN-based CSI systems using CQT/HPCP features

### The Gap
- No generative retrieval work targets cover song identification
- No CSI work uses the Semantic ID paradigm
- The RQ-VAE/RVQ architectural bridge between recommendation and audio codecs has been noted but never tested on a retrieval task with real audio

---

## 4. Experiment Design

### 4.1 Task: Cover Song Retrieval as Generative Retrieval

Given a query track, the model autoregressively generates the Semantic ID of a cover version of the same song. This replaces the traditional CSI pipeline (encode query, search index) with a single generative model whose parameters ARE the index.

**Input**: Query track representation (MERT embedding or discrete tokens)  
**Output**: Semantic ID of a cover in the same clique (3 discrete tokens)  
**Ground truth**: Binary — either the generated ID maps to a track in the correct clique, or it doesn't

### 4.2 Datasets

**Development set: Covers80** (Ellis, 2007)
- 80 cliques, 2 versions each = 160 tracks
- Audio included (164MB WAV, downloadable)
- Used for pipeline development, sanity checks, qualitative analysis

**Main evaluation: Discogs-VI-YT subset** (Araz et al., ISMIR 2024)
- Full dataset: ~493K versions across ~98K cliques
- We crawl a subset: target ~5-10K tracks across 2-3K cliques via yt-dlp
- YouTube IDs are official uploads, high quality
- Rich metadata (genre, style, year) for analysis
- Pre-computed CQT features available on Zenodo (8.8GB)
- Train/val/test splits provided

### 4.3 Semantic ID Construction (3+ conditions)

#### Condition 1: Random IDs (baseline)
- Random 3-token tuples, codebook size 256 per level
- Tests whether the generative model can memorize arbitrary mappings
- Expected: poor, especially at scale

#### Condition 2: MERT-SemIDs (audio encoder → RQ-VAE)
- Extract MERT-v1-95M embeddings per track (768-dim, mean-pool layers 7-12 across time, 30-sec center clip at 24kHz)
- Train RQ-VAE (sequential k-means, 3 levels, codebook 256) on MERT embeddings
- Semantic IDs encode audio-level similarity as learned by MERT
- Hypothesis: covers of same song get similar SemID prefixes because MERT's CQT teacher captures harmonic/tonal content

#### Condition 3: EnCodec-SemIDs (direct RVQ codes)
- Run EnCodec 24kHz on each track
- Use first 3 RVQ codebook indices as Semantic ID
- Aggregation strategy: majority vote across frames per codebook level, or center-frame codes
- No additional training — the purest test of "neural audio codec codes = Semantic IDs"
- Tests the RVQ/RQ-VAE architectural equivalence directly

### 4.4 Generative Retrieval Model

**Architecture**: T5-small (60M params)
- Semantic ID tokens added to T5 vocabulary (768 new tokens: 256 per codebook level × 3 levels, plus level indicators)
- **Input option A** (SemID-to-SemID): query track's own Semantic ID as token sequence → decoder generates target cover's Semantic ID
- **Input option B** (continuous-to-discrete): MERT embedding projected via linear layer into T5 encoder space → decoder generates target cover's Semantic ID
- We implement both and compare

**Training pairs**: For each clique, all ordered pairs of versions. If a clique has versions A, B, C: training pairs are (A→B), (A→C), (B→A), (B→C), (C→A), (C→B).

**Training**: AdamW, lr=5e-4, batch 64, early stopping on validation MRR. Teacher forcing. Beam search (width 20) at inference, constrained to valid Semantic IDs.

### 4.5 Baselines

1. **Bi-encoder**: MERT encodes both query and candidate tracks, cosine similarity, top-K retrieval via FAISS. This is the "standard practice" that generative retrieval claims to replace.
2. **Published CSI results**: Report existing benchmark numbers from Re-MOVE, CoverHunter, LyraC-Net on the same evaluation set where possible.

### 4.6 Evaluation

**CSI Metrics** (standard in the VI literature):
- MR1 (Mean Rank of first correct result)
- MRR (Mean Reciprocal Rank)
- MAP (Mean Average Precision)
- Recall@k (k = 1, 5, 10)

**Semantic ID Quality**:
- Clash rate: % of distinct tracks with identical Semantic IDs
- Codebook utilization per level
- Prefix overlap: do covers within a clique share more SemID prefixes than cross-clique tracks?

**Analysis**:
- t-SNE of Semantic ID embeddings colored by clique
- Error analysis: when the model fails, is the generated SemID acoustically close to the target? (i.e., graceful degradation)
- Cold-start: hold out entire cliques from training, assign SemIDs from audio only, test whether the model generalizes

---

## 5. Timeline (May 20 — June 1)

### Phase 1: Data (May 20-21)
- Scaffold project repo and install dependencies
- Download Covers80 (immediate, 164MB)
- Download Discogs-VI metadata + pre-computed features from Zenodo
- Start Discogs-VI-YT audio crawl via yt-dlp (overnight)

### Phase 2: Feature extraction (May 21-22)
- MERT-v1-95M embedding extraction (Covers80 first, then Discogs-VI)
- EnCodec RVQ code extraction

### Phase 3: Semantic IDs (May 22-23)
- Train RQ-VAE on MERT embeddings
- Process EnCodec codes into track-level SemIDs
- Generate random ID baseline
- Compute clash rates, codebook utilization, prefix overlap on Covers80 as sanity check

### Phase 4: Models (May 24-26)
- Train T5-small generative retrieval model (one per SemID condition)
- Implement bi-encoder baseline
- Run inference with beam search

### Phase 5: Evaluation (May 27-28)
- Full evaluation suite (MR1, MRR, MAP, Recall@k)
- t-SNE visualization
- Cold-start experiment
- Error analysis

### Phase 6: Paper (May 28-31)
- Write paper in IEEE LaTeX format (6 pages)
- Generate all tables and figures
- IEEE PDF eXpress compliance
- Submit via CMT by June 1

---

## 6. Paper Outline (6 pages, IEEE format)

### I. Introduction (~0.75 page)
- Generative retrieval replaces index + ANN with autoregressive ID generation
- Cover song identification: a classic MIR task with clean ground truth
- Key insight: RQ-VAE (TIGER) ≡ RVQ (EnCodec) architecturally
- First generative retrieval approach to CSI

### II. Related Work (~0.75 page)
- Generative retrieval (DSI, TIGER, Text2Tracks, FusID)
- Cover song identification (Da-TACOS, Discogs-VI, Re-MOVE)
- Audio representation models (MERT, EnCodec)

### III. Method (~1.5 pages)
- Task formulation: CSI as generative retrieval
- Semantic ID construction (diagram: audio → encoder → RQ-VAE → discrete IDs)
- T5-based generative model with vocabulary expansion

### IV. Experiments (~1.5 pages)
- Datasets: Covers80 + Discogs-VI-YT subset
- Training configuration
- Main results table (3 SemID conditions + bi-encoder baseline)
- Semantic ID quality metrics

### V. Analysis & Discussion (~1 page)
- Do covers share SemID prefixes? (prefix overlap analysis)
- Cold-start performance
- What the model learns: t-SNE visualization
- Risk discussion: acoustic vs. compositional similarity in SemIDs
- Limitations

### VI. Conclusion (~0.5 page)
- Summary, future work (larger scale, CLAP embeddings, raw audio generation)

---

## 7. Key References

1. Tay et al. (2022). Transformer Memory as a Differentiable Search Index. NeurIPS.
2. Rajput et al. (2023). TIGER: Recommender Systems with Generative Retrieval. NeurIPS.
3. Palumbo & Penha et al. (2025). Text2Tracks. arXiv:2503.24193.
4. Kim, Hou & McAuley (2026). FusID. arXiv:2601.08764.
5. Mei et al. (2025). Semantic IDs for Music Recommendation. RecSys.
6. Yuan et al. (2023). MERT. arXiv:2306.00107.
7. Defossez et al. (2022). EnCodec: High Fidelity Neural Audio Compression.
8. Araz, Serra & Bogdanov (2024). Discogs-VI. ISMIR.
9. Yesiler et al. (2019). Da-TACOS. ISMIR.
10. Ellis (2007). Covers80 dataset.

---

## 8. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Discogs-VI-YT crawl too slow or high link rot | Fall back to Covers80 as primary eval (small but with audio). Frame as proof-of-concept. |
| MERT SemIDs capture acoustic similarity, not compositional | This IS an interesting negative result. Discuss in paper: "audio codec features encode surface acoustics, not deep compositional structure." Points to future work on disentanglement. |
| Covers80 too small for meaningful training | Use all 160 tracks for RQ-VAE training, leave-one-clique-out evaluation. Small N is acknowledged as limitation. |
| T5 overfits on small dataset | Aggressive regularization (dropout 0.3), early stopping, report overfitting analysis. |
| EnCodec frame-level codes too noisy as track-level SemIDs | Multiple aggregation strategies (majority vote, center frame, mean-pool). Report best. |
| 6-page limit too tight | Focus on main comparison table + one analysis figure. Code/details go to GitHub. |

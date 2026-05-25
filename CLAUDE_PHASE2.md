# CLAUDE.md -- Phase 2: CLEWS SemID Condition + Evaluation Upgrades

## Context

This is Phase 2 of the generative retrieval for cover song identification project. Phase 1 (complete) built the full pipeline: Discogs-VI-YT crawl, MERT/EnCodec/random SemID conditions, T5 training, bi-encoder baseline, evaluation. Phase 2 adds a fourth SemID condition using CLEWS (current SOTA CSI model) embeddings, adds the NAR metric used by the CSI community, and prepares final paper numbers.

Everything runs on a single RunPod GPU (RTX 4090 or A100). The existing codebase and data are on HuggingFace at `gabelev/discogs-vi-csi-subset`.

## What Already Exists

- `src/features/extract_mert.py` -- MERT embedding extraction (768-dim)
- `src/features/extract_encodec.py` -- EnCodec code extraction
- `src/semantic_ids/train_rqvae.py` -- Sequential k-means RQ-VAE (3 levels, codebook 256)
- `src/semantic_ids/random_ids.py` -- Random SemID baseline
- `src/semantic_ids/encodec_to_semid.py` -- EnCodec majority-vote aggregation
- `src/semantic_ids/analyze_ids.py` -- Clash rate, codebook utilization, prefix overlap
- `src/model/train.py` -- T5-small training with bf16, embedding re-init
- `src/model/inference.py` -- Constrained beam search (width 20)
- `src/model/dataset.py` -- SemID-token vocab extension + pair dataset
- `src/baselines/biencoder.py` -- MERT bi-encoder + FAISS
- `src/evaluation/evaluate.py` -- MRR, MAP, Recall@k, MR1
- `src/evaluation/visualize.py` -- Prefix overlap bars, t-SNE
- `src/evaluation/error_analysis.py` -- Cosine similarity analysis of T5 misses
- `data/splits/discogs_vi_subset.csv` -- 3,413 tracks, 696 cliques
- `data/splits/covers80_cliques.csv` -- 160 tracks, 80 cliques
- `data/embeddings/mert/` -- MERT embeddings (.npz)
- `data/semantic_ids/` -- All existing SemID CSVs
- `configs/` -- YAML configs with inheritance (base.yaml + per-condition overrides)
- `scripts/run_t5_training_pod.sh` -- Chains training + inference for all conditions

## Task List

### Task 1: Add NAR Metric

Add Normalized Average Rank to the evaluation suite. Use the corrected formula from CLEWS (Serra et al. 2025, Appendix B):

```python
def nar(ranks, n_candidates):
    """
    ranks: list of lists. For each query, the 1-indexed ranks of all true positives.
    n_candidates: total number of candidates (excluding the query itself).
    Returns: mean NAR across queries (lower is better, 0 = perfect, 100 = worst).
    """
    nar_scores = []
    for r_list in ranks:
        M = len(r_list)
        if M == 0:
            continue
        sorted_ranks = sorted(r_list)
        score = sum(rank - (i + 1) for i, rank in enumerate(sorted_ranks))
        score = 100.0 * score / (M * (n_candidates - M))
        nar_scores.append(score)
    return sum(nar_scores) / len(nar_scores)
```

Files to modify:
- `src/evaluation/metrics.py` -- add `nar()` function
- `src/evaluation/evaluate.py` -- compute and report NAR alongside existing metrics
- Rerun evaluation on all existing predictions to get NAR numbers

Output: updated `runs/metrics_*.json` files with NAR field, updated `docs/RESULTS.md`.

### Task 2: Install CLEWS and Extract Embeddings

Clone the CLEWS repo (https://github.com/sony/clews) and use their pretrained DVI checkpoint to extract 1024-dim embeddings for all our tracks.

Steps:
1. `git clone https://github.com/sony/clews`
2. Install requirements (check `install_requirements.sh`, needs python>=3.10, pytorch, nnAudio, etc.)
3. Download their pretrained checkpoint. It should be in `logs/model/checkpoint_best.ckpt` or downloadable from the repo. Check the README and `models/` directory.
4. Download all audio from HF dataset `gabelev/discogs-vi-csi-subset` to a local directory. Audio files are WAV 24kHz mono, 30-sec clips. CLEWS expects 16kHz mono, up to 10 min. Our 30-sec 24kHz files should work but may need resampling to 16kHz.
5. Run CLEWS inference:
```bash
OMP_NUM_THREADS=1 python inference.py \
  --checkpoint=logs/model/checkpoint_best.ckpt \
  --path_in=data/audio/ \
  --path_out=cache/clews_embeddings/
```
6. CLEWS outputs per-file .pt tensors. Each is a set of segment embeddings (8 segments of 20s from a 2.5min block, 1024-dim each). For our 30-sec clips, expect 1-2 segments. Mean-pool segments to get one 1024-dim vector per track.
7. Collect into a single NPZ file keyed by track_id: `data/embeddings/clews/discogs_vi_embeddings.npz` and `data/embeddings/clews/covers80_embeddings.npz`.

Write: `src/features/extract_clews.py` that wraps steps 5-7 (load .pt files, mean-pool segments, save NPZ). Keep it consistent with `extract_mert.py` output format.

IMPORTANT: If the 30-sec clips cause issues with CLEWS (it expects 2.5 min blocks in training), the segments will be short. That's fine. The model will still produce embeddings. If there are errors, try repeat-padding the audio to 2.5 min before passing to CLEWS.

### Task 3: CLEWS RQ-VAE and SemID Analysis

Train RQ-VAE on CLEWS embeddings and run the full SemID quality analysis.

1. Run `src/semantic_ids/train_rqvae.py` with CLEWS embeddings as input:
   - Input dim: 1024 (vs 768 for MERT). The RQ-VAE (sequential k-means) is dim-agnostic, so this should just work.
   - Same settings: 3 levels, codebook size 256
   - Train on Discogs-VI embeddings, apply same codebooks to Covers80
2. Output: `data/semantic_ids/discogs_vi_clews.csv` and `data/semantic_ids/covers80_clews.csv`
3. Run `src/semantic_ids/analyze_ids.py` on CLEWS SemIDs for both datasets
4. Compare prefix overlap ratios to MERT (expect CLEWS to be significantly higher since it's trained for version invariance)

Output: `runs/clews_semid_report.json` with clash rate, codebook utilization, within/cross prefix overlap ratios.

### Task 4: T5 Training on CLEWS SemIDs

Train T5-small generative retrieval model with CLEWS-derived SemIDs.

1. Add a new config: `configs/clews_ids.yaml`:
```yaml
inherits: base.yaml
condition: clews
embedding_source: clews
codebook_size: 256
n_levels: 3
```
2. Run the same training pipeline as MERT/EnCodec/random conditions:
   - Transductive pair-split (same splits as other conditions)
   - Embedding re-init from original-vocab Gaussian (same fix from Phase 1 run #3)
   - bf16, AdamW lr=3e-4, early stopping patience 3
   - Constrained beam search width 20 at inference
3. Run inference on Discogs-VI test set and Covers80

Output: model weights, predictions JSONs in `runs/`.

### Task 5: Full Evaluation Suite

Run evaluation across all 4 SemID conditions + bi-encoder baseline with the updated metrics (now including NAR).

1. Run `src/evaluation/evaluate.py` on all prediction files:
   - T5 random (Discogs-VI, Covers80)
   - T5 MERT-RQ-VAE (Discogs-VI, Covers80)
   - T5 EnCodec (Discogs-VI, Covers80)
   - T5 CLEWS-RQ-VAE (Discogs-VI, Covers80) [NEW]
   - Bi-encoder (Discogs-VI, Covers80)
2. For each, report: MRR, MAP, NAR, Recall@1, Recall@5, Recall@10, MR1
3. Run error analysis on CLEWS condition (same as MERT error analysis):
   - When T5 misses, what's the CLEWS cosine similarity between query and true cover vs query and T5's wrong answer?
   - Compare to the MERT error analysis numbers (0.886 vs 0.900)
4. Run prefix overlap visualization for CLEWS (add to existing bar chart)
5. Update t-SNE with CLEWS SemIDs

Output: All metrics in `runs/metrics_*.json`, updated `docs/RESULTS.md`, updated figures in `paper/figures/`.

### Task 6: Update Paper Tables and Figures

Update the LaTeX paper with new numbers:

1. Add CLEWS row to Table I (SemID quality): clash rate, codebook utilization, prefix overlap ratios
2. Add CLEWS row to Table II (retrieval results): all metrics for T5 CLEWS condition
3. Add NAR column to Table II
4. Update prefix_overlap bar chart (Fig 3) to include CLEWS condition
5. Update t-SNE plot (if generated)
6. Add CLEWS to the pipeline diagram legend/caption

Files to modify: `paper/main.tex`

## RunPod Execution Script

All tasks chain into a single script. Run in tmux.

```bash
#!/bin/bash
set -e

# --- Setup ---
pip install -r requirements.txt --break-system-packages
apt-get update && apt-get install -y libsndfile1 git

# --- Pull data from HF ---
# (existing scripts handle this)

# --- Task 1: NAR metric ---
# (code changes to metrics.py and evaluate.py, then rerun evals)

# --- Task 2: CLEWS embeddings ---
cd /workspace
git clone https://github.com/sony/clews
cd clews
bash install_requirements.sh
# Download checkpoint (check repo for location)
# Download audio from HF
# Run inference
# Mean-pool and save NPZ
cd /workspace/gen-retrieval-csi

# --- Task 3: CLEWS RQ-VAE ---
python src/semantic_ids/train_rqvae.py --source clews --dataset discogs_vi
python src/semantic_ids/train_rqvae.py --source clews --dataset covers80 --codebooks data/semantic_ids/clews_codebooks.pkl
python src/semantic_ids/analyze_ids.py --condition clews --dataset discogs_vi
python src/semantic_ids/analyze_ids.py --condition clews --dataset covers80

# --- Task 4: T5 training ---
python src/model/train.py --config configs/clews_ids.yaml
python src/model/inference.py --config configs/clews_ids.yaml --split test
python src/model/inference.py --config configs/clews_ids.yaml --split covers80

# --- Task 5: Evaluation ---
python src/evaluation/evaluate.py --predictions runs/predictions_clews_discogs_vi.json --cliques data/splits/discogs_vi_test_cliques.csv
python src/evaluation/evaluate.py --predictions runs/predictions_clews_covers80.json --cliques data/splits/covers80_cliques.csv
# Rerun all other conditions with NAR
for cond in random mert encodec; do
  python src/evaluation/evaluate.py --predictions runs/predictions_${cond}_discogs_vi.json --cliques data/splits/discogs_vi_test_cliques.csv
  python src/evaluation/evaluate.py --predictions runs/predictions_${cond}_covers80.json --cliques data/splits/covers80_cliques.csv
done
python src/evaluation/evaluate.py --predictions runs/predictions_biencoder_discogs_vi.json --cliques data/splits/discogs_vi_test_cliques.csv

# Error analysis
python src/evaluation/error_analysis.py --condition clews --dataset discogs_vi

# Figures
python src/evaluation/visualize.py --all

# --- Push results to HF ---
# (existing push scripts)

echo "Phase 2 complete."
```

## Key Risks

1. **CLEWS checkpoint availability**: The repo says checkpoints are included in `models/` directory. If they're not in the git repo (too large), they may be on a model hosting service. Check the README carefully. If unavailable, fall back to DVINet+ (also has public checkpoints from the same research group).

2. **Audio format mismatch**: CLEWS expects 16kHz mono. Our audio is 24kHz mono, 30-sec clips. CLEWS training used 2.5-min blocks cut into 8x20-sec segments. For 30-sec clips, CLEWS will produce ~1-2 segments. This is fine for embedding extraction. If it errors, resample to 16kHz with torchaudio or ffmpeg before passing in.

3. **Embedding dimensionality**: CLEWS outputs 1024-dim. MERT is 768-dim. RQ-VAE is dim-agnostic (k-means works in any dimension). No code changes needed in train_rqvae.py.

4. **CLEWS repo dependencies**: May conflict with our existing env. If so, create a separate venv for CLEWS inference, extract embeddings, then switch back.

## Expected Outcomes

If CLEWS embeddings encode version invariance well (which they should, it's the SOTA CSI model), we expect:
- Prefix overlap ratios significantly higher than MERT (maybe 8-15x at c1, 30-50x at c1+c2 vs MERT's 4.3x/11x)
- T5 CLEWS MRR substantially higher than T5 MERT (target: 0.35-0.50 range vs 0.271)
- Error analysis showing T5 CLEWS misses are compositionally closer to true covers than MERT misses
- A clean ranking: CLEWS >> MERT >> EnCodec ~ random, matching intrinsic SemID quality

This confirms the paper's thesis: the quality of Semantic IDs (and specifically, whether they encode the right invariances for the task) is the binding constraint on generative retrieval performance.

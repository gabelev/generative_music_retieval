# CLAUDE.md — Generative Retrieval for Cover Song Identification

## Project Overview

We are building a generative retrieval system for cover song identification (CSI). Instead of the traditional pipeline (encode query audio → search an embedding index via ANN), we train a T5 model to autoregressively generate the Semantic ID of a cover version given a query track.

This is for a 6-page IEEE conference paper targeting ICECCME 2026 (deadline June 1).

## Key Concepts

**Semantic IDs**: Each track in the corpus gets a discrete tuple of 3 tokens, e.g. `<42><17><203>`. These are constructed by encoding audio through a pretrained model and quantizing the embedding via RQ-VAE (residual quantization). Similar-sounding tracks should get similar Semantic IDs.

**Generative retrieval**: A T5 model takes a query track's representation as input and generates the Semantic ID of a matching cover as output. The model's parameters replace the traditional embedding index.

**Cover song identification**: Given a query recording, find other recordings of the same underlying musical composition. Covers differ in key, tempo, instrumentation, production — but share harmonic and melodic structure.

## Architecture

```
Query Track Audio
    │
    ▼
MERT-v1-95M (pretrained, frozen)
    │
    ▼
768-dim embedding (mean-pool layers 7-12, mean across time)
    │
    ├──→ RQ-VAE (3-level, codebook 256) ──→ Semantic ID: <c1><c2><c3>
    │
    └──→ Linear projection into T5 encoder dim
              │
              ▼
         T5-small (fine-tuned)
              │
              ▼
         Semantic ID of cover: <c1'><c2'><c3'>
```

## Repo Structure

```
gen-retrieval-csi/
├── CLAUDE.md                  # This file
├── README.md                  # Project overview for GitHub
├── requirements.txt           # Python dependencies
├── configs/
│   ├── base.yaml              # Shared hyperparameters
│   ├── random_ids.yaml        # Random SemID condition
│   ├── mert_ids.yaml          # MERT RQ-VAE condition
│   └── encodec_ids.yaml       # EnCodec direct codes condition
├── data/
│   ├── raw/                   # Downloaded audio (gitignored)
│   │   ├── covers80/
│   │   └── discogs_vi/
│   ├── embeddings/            # Extracted features (gitignored)
│   │   ├── mert/
│   │   └── encodec/
│   ├── semantic_ids/          # Generated SemID mappings
│   └── splits/                # Train/val/test splits as CSVs
├── src/
│   ├── data/
│   │   ├── download_covers80.py
│   │   ├── download_discogs_vi.py
│   │   ├── crawl_youtube.py
│   │   └── build_splits.py
│   ├── features/
│   │   ├── extract_mert.py
│   │   └── extract_encodec.py
│   ├── semantic_ids/
│   │   ├── train_rqvae.py
│   │   ├── encodec_to_semid.py
│   │   ├── random_ids.py
│   │   └── analyze_ids.py      # Clash rate, utilization, prefix overlap
│   ├── model/
│   │   ├── dataset.py          # PyTorch Dataset for (query, target_semid) pairs
│   │   ├── train.py            # T5 fine-tuning loop
│   │   ├── inference.py        # Beam search with constrained decoding
│   │   └── t5_with_projection.py  # T5 with linear projection for continuous input
│   ├── baselines/
│   │   ├── biencoder.py        # MERT cosine similarity + FAISS
│   │   └── evaluate_baselines.py
│   ├── evaluation/
│   │   ├── metrics.py          # MR1, MRR, MAP, Recall@k
│   │   ├── evaluate.py         # Main evaluation script
│   │   └── visualize.py        # t-SNE, prefix overlap plots
│   └── utils.py
├── paper/
│   ├── main.tex
│   ├── references.bib
│   └── figures/
├── scripts/
│   ├── run_full_pipeline.sh
│   └── run_covers80_dev.sh     # Quick dev cycle on small dataset
└── .gitignore
```

## Dependencies

```
# requirements.txt
torch>=2.0
torchaudio>=2.0
transformers>=4.30
encodec>=0.1.1
scikit-learn>=1.3
faiss-cpu>=1.7
numpy
pandas
matplotlib
seaborn
deepdish
pyyaml
tqdm
yt-dlp
```

## Task Breakdown (in execution order)

### Task 1: Scaffold and Setup

Create the repo structure above. Initialize `requirements.txt`. Create a minimal `README.md` with project title and description.

### Task 2: Download Covers80

Download from `http://labrosa.ee.columbia.edu/projects/coversongs/covers80/` or Kaggle (`https://www.kaggle.com/datasets/arpanpathak/original-and-cover-song-pairs`).

Expected structure after download:
```
data/raw/covers80/
├── covers32k/
│   ├── The Beatles+Come Together/       # Artist+Song directories
│   │   ├── beatles.mp3                  # Version A
│   │   └── aerosmith.mp3               # Version B
│   └── ...
├── list1.txt                            # Version A filenames
└── list2.txt                            # Version B filenames (same order = same clique)
```

**Build clique mappings**: Parse `list1.txt` and `list2.txt`. Each line pair at the same index is a clique. Output: `data/splits/covers80_cliques.csv` with columns `track_id, clique_id, filepath, version`.

### Task 3: Download Discogs-VI-YT Metadata

Download from Zenodo: `https://zenodo.org/records/13983028`
- `main.zip` (1.4 GB) — contains `Discogs-VI-YT-20240701-light.json` (the lightweight JSON with clique IDs, version IDs, YouTube IDs)
- `intermediary.zip` (8.8GB) — pre-computed CQT features (optional, lower priority)

Parse the JSON: it's a dict keyed by `clique_id`, each value a list of `{version_id, track_title, youtube_id}`. Sample a manageable subset: 2,500 cliques with at least 2 versions each → ~12K total candidate tracks.

Output: `data/splits/discogs_vi_subset.csv` with columns `version_id, clique_id, youtube_id`.

### Task 4: Crawl Discogs-VI-YT Audio (with HF stream upload)

Use `src/data/crawl_youtube.py`: yt-dlp downloads bestaudio per ID; ffmpeg
center-clips to 30 sec and transcodes to 24kHz mono WAV; the WAV is staged
locally and uploaded to a private Hugging Face dataset repo in batches of
100 per commit (HF free tier = 128 commits/hour on dataset repos — single-file
commits blow the rate limit fast). Local files are deleted after each batch
commit so persistent local disk stays under ~150 MB.

Required CLI tools / runtime:
- `yt-dlp` (modern build supporting `--js-runtimes`)
- `ffmpeg`, `ffprobe`
- `node` (or `deno`) — yt-dlp needs a JS runtime to decrypt YouTube's signatures;
  without one yt-dlp silently fails formats with "Video unavailable"

Auth: `HF_TOKEN` env var or `huggingface-cli login`.

**Reality check — link rot and bot detection.** YouTube IDs in the Discogs-VI metadata (timestamped 2024-07-01) rot fast, especially for cover songs (DMCA takedowns). Empirically, of a 12,006-track crawl in May 2026:
- 3,413 ok (28.4%)
- 3,404 video_unavailable (28.3% — deletions/takedowns/region-locks)
- 5,270 sign-in / bot-detection ("Sign in to confirm you're not a bot") (43.9%)
- ~250 other transient yt-dlp errors

Run inside `tmux` or `screen`, with `caffeinate -dis` to keep the Mac awake.
Track success/failure rates in `data/splits/discogs_vi_download_log.csv`
(cols: version_id, clique_id, youtube_id, status, hf_path, error).

### Task 4b: Filter to surviving cliques

After the crawl, derive `data/splits/discogs_vi_usable.csv` keeping only
cliques with ≥2 successfully-downloaded versions. Drop orphan tracks whose
clique-mates didn't make it through.

Typical yield from the May 2026 crawl: 696 usable cliques across 3,192
tracks (~42K training ordered pairs at 70% split). Distribution is heavy-
tailed: most cliques have 2-3 versions, a long tail extends to 60-130 versions
(e.g., classic covers like "Yesterday" or "Hallelujah").

### Task 5: MERT-v1-95M Embedding Extraction

```python
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import torch, torchaudio

model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
model.eval()

def extract_mert_embedding(audio_path):
    audio, sr = torchaudio.load(audio_path)
    if sr != 24000:
        audio = torchaudio.transforms.Resample(sr, 24000)(audio)
    audio = audio.squeeze()
    
    # Take 30-sec center clip
    max_samples = 24000 * 30
    if audio.shape[0] > max_samples:
        start = (audio.shape[0] - max_samples) // 2
        audio = audio[start:start + max_samples]
    
    inputs = processor(audio, sampling_rate=24000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Mean-pool layers 7-12, then mean across time
    hidden = torch.stack(outputs.hidden_states[7:13])  # [6, 1, T, 768]
    embedding = hidden.mean(dim=[0, 2]).squeeze()  # [768]
    return embedding.numpy()
```

Run on all tracks. Save as HDF5 or a single `.npz`:
```
data/embeddings/mert/covers80_embeddings.npz  # {track_id: 768-dim array}
data/embeddings/mert/discogs_vi_embeddings.npz
```

### Task 6: EnCodec Code Extraction

```python
from encodec import EncodecModel
from encodec.utils import convert_audio
import torchaudio

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(1.5)
model.eval()

def extract_encodec_codes(audio_path):
    audio, sr = torchaudio.load(audio_path)
    audio = convert_audio(audio, sr, 24000, 1)
    
    # 30-sec center clip
    max_samples = 24000 * 30
    if audio.shape[1] > max_samples:
        start = (audio.shape[1] - max_samples) // 2
        audio = audio[:, start:start + max_samples]
    
    with torch.no_grad():
        encoded = model.encode(audio.unsqueeze(0))
    
    codes = encoded[0][0].squeeze()  # [n_codebooks, n_frames]
    
    # Aggregation: majority vote per codebook level
    from scipy import stats
    sem_id = []
    for cb in range(min(3, codes.shape[0])):
        mode_val = stats.mode(codes[cb].numpy(), keepdims=False).mode
        sem_id.append(int(mode_val))
    
    return sem_id, codes.numpy()  # Return both SemID and raw codes
```

Save: `data/embeddings/encodec/covers80_codes.npz`

### Task 7: Train RQ-VAE on MERT Embeddings

Sequential k-means residual quantization:

```python
from sklearn.cluster import MiniBatchKMeans
import numpy as np

def train_rqvae(embeddings, n_levels=3, codebook_size=256):
    codebooks = []
    residual = embeddings.copy()
    for level in range(n_levels):
        kmeans = MiniBatchKMeans(n_clusters=codebook_size, batch_size=4096, random_state=42)
        kmeans.fit(residual)
        codebooks.append(kmeans)
        assigned = kmeans.predict(residual)
        centroids = kmeans.cluster_centers_[assigned]
        residual = residual - centroids
    return codebooks

def encode_rqvae(embeddings, codebooks):
    sem_ids = []
    residual = embeddings.copy()
    for kmeans in codebooks:
        assigned = kmeans.predict(residual)
        sem_ids.append(assigned)
        centroids = kmeans.cluster_centers_[assigned]
        residual = residual - centroids
    return np.stack(sem_ids, axis=1)  # [N, n_levels]
```

Also generate random IDs for the baseline condition:
```python
random_ids = np.random.randint(0, 256, size=(n_tracks, 3))
```

Save all SemID mappings: `data/semantic_ids/{condition}_semids.csv` with columns `track_id, c1, c2, c3`.

### Task 8: Analyze Semantic IDs

Before training the generative model, validate the SemIDs:

1. **Clash rate**: `n_unique_semids / n_tracks` — should be high (close to 1.0)
2. **Codebook utilization**: per level, what fraction of 256 entries are used?
3. **Prefix overlap within cliques**: For each clique, do the versions share the same first token (c1)? Same first two tokens (c1, c2)? Compare to random chance.
4. **Sanity check on Covers80**: Print a few cliques showing both versions' SemIDs. Are they close?

### Task 9: Build Training Data

For each clique with versions [A, B, C, ...], generate all ordered pairs:
- (A_semid → B_semid), (A_semid → C_semid), (B_semid → A_semid), etc.

If using continuous input: pair (A_mert_embedding, B_semid), etc.

Split by clique (not by track) to prevent leakage:
- Train: 70% of cliques
- Val: 15% of cliques
- Test: 15% of cliques

Output: `data/splits/{dataset}_{condition}_train.csv`, `_val.csv`, `_test.csv`

### Task 10: Train T5-small

Key implementation details:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Add SemID tokens to vocabulary
# 3 levels x 256 codes = 768 tokens, plus 3 level indicators
sem_tokens = [f"<L{l}_C{c}>" for l in range(3) for c in range(256)]
sem_tokens += ["<L0>", "<L1>", "<L2>"]
tokenizer.add_tokens(sem_tokens)

model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.resize_token_embeddings(len(tokenizer))
```

**Input format** (SemID-to-SemID mode):
```
"retrieve cover: <L0_C42> <L1_C17> <L2_C203>"
→ target: "<L0_C88> <L1_C3> <L2_C156>"
```

**Training loop**: Standard T5 seq2seq with teacher forcing. Use HuggingFace Trainer or custom loop.

**Hyperparameters**:
- lr: 5e-4, AdamW
- batch_size: 64
- max_epochs: 15
- early_stopping: patience 3 on val MRR
- dropout: 0.3

### Task 11: Inference with Constrained Beam Search

At inference time, use beam search (width 20) with constrained decoding:
- At position 0: only allow tokens `<L0_C0>` through `<L0_C255>`
- At position 1: only allow tokens `<L1_C0>` through `<L1_C255>`
- At position 2: only allow tokens `<L2_C0>` through `<L2_C255>`

Map generated SemIDs back to track IDs. If the SemID doesn't map to any track (invalid generation), it's a miss.

For each query, generate top-K beams and compute metrics against all tracks in the correct clique.

### Task 12: Bi-encoder Baseline

```python
import faiss

# embeddings: [N, 768] MERT embeddings for all tracks
index = faiss.IndexFlatIP(768)  # Inner product (cosine after L2 norm)
faiss.normalize_L2(embeddings)
index.add(embeddings)

# Query
query_emb = extract_mert_embedding(query_path)
faiss.normalize_L2(query_emb.reshape(1, -1))
distances, indices = index.search(query_emb.reshape(1, -1), k=20)
```

### Task 13: Evaluation

```python
def mean_rank_1(predictions, ground_truth_cliques):
    """Mean rank of first correct result."""
    ranks = []
    for query_id, pred_ids in predictions.items():
        true_clique = ground_truth_cliques[query_id]
        for rank, pid in enumerate(pred_ids, 1):
            if ground_truth_cliques.get(pid) == true_clique and pid != query_id:
                ranks.append(rank)
                break
        else:
            ranks.append(len(pred_ids) + 1)  # Not found
    return np.mean(ranks)

def mrr(predictions, ground_truth_cliques):
    """Mean reciprocal rank."""
    rrs = []
    for query_id, pred_ids in predictions.items():
        true_clique = ground_truth_cliques[query_id]
        for rank, pid in enumerate(pred_ids, 1):
            if ground_truth_cliques.get(pid) == true_clique and pid != query_id:
                rrs.append(1.0 / rank)
                break
        else:
            rrs.append(0.0)
    return np.mean(rrs)
```

Also compute MAP and Recall@k (k=1,5,10).

### Task 14: Visualization

1. **t-SNE**: Embed all Semantic IDs (as 3-dim discrete vectors, expanded via codebook centroids) and plot colored by clique. Do covers cluster?
2. **Prefix overlap heatmap**: For each clique, fraction of version pairs sharing c1, (c1,c2), (c1,c2,c3). Compare across conditions.
3. **Results table**: LaTeX-formatted comparison table for the paper.

## Important Notes

- All audio processing at 24kHz mono (MERT requirement)
- 30-second center clips from each track
- Split by clique, never by track, to prevent data leakage
- Covers80 + Discogs-VI together form the paper's eval. Covers80 (80 cliques)
  serves as a CROSS-DATASET held-out test set — the model never sees any
  Covers80 track during training — while Discogs-VI provides in-distribution
  train/val/test splits (487/104/105 cliques) and the larger training corpus.
- This setup is empirically determined by what the May 2026 yt-dlp crawl
  yielded (28% per-track success rate due to YouTube link rot + bot detection),
  not an a priori design choice. It happens to be a stronger paper story:
  cross-dataset generalization is more interesting than single-dataset eval.
- The paper deadline is June 1. Prioritize getting end-to-end results over perfection in any single component.
- Use chaos-dimension (chaosdimension.fyi) to track progress on tasks in the generative-retrieval workstream.

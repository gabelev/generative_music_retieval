#!/usr/bin/env bash
# Full pipeline on Discogs-VI-YT subset (main paper results).
# Assumes Covers80 dev cycle already passes.
set -euo pipefail

cd "$(dirname "$0")/.."

# 1. Metadata + crawl (run overnight)
python -m src.data.download_discogs_vi --n-cliques 2500 --min-versions 2
python -m src.data.crawl_youtube \
    --subset-csv data/splits/discogs_vi_subset.csv \
    --out-dir data/raw/discogs_vi

# 2. Features
python -m src.features.extract_mert \
    --audio-dir data/raw/discogs_vi \
    --cliques-csv data/splits/discogs_vi_subset.csv \
    --out data/embeddings/mert/discogs_vi_embeddings.npz

python -m src.features.extract_encodec \
    --audio-dir data/raw/discogs_vi \
    --cliques-csv data/splits/discogs_vi_subset.csv \
    --out data/embeddings/encodec/discogs_vi_codes.npz

# 3. SemIDs
python -m src.semantic_ids.random_ids \
    --cliques-csv data/splits/discogs_vi_subset.csv \
    --out data/semantic_ids/discogs_vi_random.csv
python -m src.semantic_ids.train_rqvae \
    --embeddings data/embeddings/mert/discogs_vi_embeddings.npz \
    --out-codebooks data/semantic_ids/discogs_vi_mert_codebooks.pkl \
    --out-semids data/semantic_ids/discogs_vi_mert.csv
python -m src.semantic_ids.encodec_to_semid \
    --codes-npz data/embeddings/encodec/discogs_vi_codes.npz \
    --out data/semantic_ids/discogs_vi_encodec.csv

# 4. Splits, training, eval (mirrors run_covers80_dev.sh)
for cond in random_ids mert_ids encodec_ids; do
    python -m src.data.build_splits --config configs/${cond}.yaml --dataset discogs_vi
    python -m src.model.train  --config configs/${cond}.yaml --out-dir runs/discogs_vi_${cond}
    python -m src.model.inference \
        --ckpt-dir runs/discogs_vi_${cond} \
        --test-csv data/splits/discogs_vi_${cond}_test.csv \
        --out runs/discogs_vi_${cond}_preds.json
    python -m src.evaluation.evaluate \
        --predictions runs/discogs_vi_${cond}_preds.json \
        --cliques-csv data/splits/discogs_vi_subset.csv \
        --out runs/discogs_vi_${cond}_metrics.json
done

# 5. Bi-encoder baseline
python -m src.baselines.biencoder \
    --embeddings data/embeddings/mert/discogs_vi_embeddings.npz \
    --cliques-csv data/splits/discogs_vi_subset.csv \
    --out runs/discogs_vi_biencoder_preds.json
python -m src.evaluation.evaluate \
    --predictions runs/discogs_vi_biencoder_preds.json \
    --cliques-csv data/splits/discogs_vi_subset.csv \
    --out runs/discogs_vi_biencoder_metrics.json

# 6. Figures
python -m src.evaluation.visualize \
    --semids-csv data/semantic_ids/discogs_vi_mert.csv \
    --cliques-csv data/splits/discogs_vi_subset.csv \
    --out-dir paper/figures

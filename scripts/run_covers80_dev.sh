#!/usr/bin/env bash
# Quick end-to-end dev cycle on Covers80 (160 tracks).
# Use this to sanity-check the full pipeline before scaling to Discogs-VI.
set -euo pipefail

cd "$(dirname "$0")/.."

# 1. Data
python -m src.data.download_covers80 --out-dir data/raw/covers80

# 2. Features
python -m src.features.extract_mert \
    --audio-dir data/raw/covers80 \
    --cliques-csv data/splits/covers80_cliques.csv \
    --out data/embeddings/mert/covers80_embeddings.npz

python -m src.features.extract_encodec \
    --audio-dir data/raw/covers80 \
    --cliques-csv data/splits/covers80_cliques.csv \
    --out data/embeddings/encodec/covers80_codes.npz

# 3. Semantic IDs (3 conditions)
python -m src.semantic_ids.random_ids \
    --cliques-csv data/splits/covers80_cliques.csv \
    --out data/semantic_ids/covers80_random.csv

python -m src.semantic_ids.train_rqvae \
    --embeddings data/embeddings/mert/covers80_embeddings.npz \
    --out-codebooks data/semantic_ids/covers80_mert_codebooks.pkl \
    --out-semids data/semantic_ids/covers80_mert.csv

python -m src.semantic_ids.encodec_to_semid \
    --codes-npz data/embeddings/encodec/covers80_codes.npz \
    --out data/semantic_ids/covers80_encodec.csv

# 4. SemID diagnostics
for cond in random mert encodec; do
    python -m src.semantic_ids.analyze_ids \
        --semids-csv data/semantic_ids/covers80_${cond}.csv \
        --cliques-csv data/splits/covers80_cliques.csv \
        --out runs/covers80_${cond}_semid_report.json
done

# 5. Splits + training pairs
for cond in random_ids mert_ids encodec_ids; do
    python -m src.data.build_splits --config configs/${cond}.yaml --dataset covers80
done

# 6. Train T5 per condition
for cond in random_ids mert_ids encodec_ids; do
    python -m src.model.train --config configs/${cond}.yaml --out-dir runs/covers80_${cond}
done

# 7. Inference + eval
for cond in random_ids mert_ids encodec_ids; do
    python -m src.model.inference \
        --ckpt-dir runs/covers80_${cond} \
        --test-csv data/splits/covers80_${cond}_test.csv \
        --out runs/covers80_${cond}_preds.json
    python -m src.evaluation.evaluate \
        --predictions runs/covers80_${cond}_preds.json \
        --cliques-csv data/splits/covers80_cliques.csv \
        --out runs/covers80_${cond}_metrics.json
done

# 8. Bi-encoder baseline
python -m src.baselines.biencoder \
    --embeddings data/embeddings/mert/covers80_embeddings.npz \
    --cliques-csv data/splits/covers80_cliques.csv \
    --out runs/covers80_biencoder_preds.json
python -m src.evaluation.evaluate \
    --predictions runs/covers80_biencoder_preds.json \
    --cliques-csv data/splits/covers80_cliques.csv \
    --out runs/covers80_biencoder_metrics.json

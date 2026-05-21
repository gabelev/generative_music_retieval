#!/usr/bin/env bash
# Local-only: generate random-ID baselines, build per-condition Semantic IDs
# analysis reports, and emit clique-level train/val/test splits.
#
# Assumes you've already produced (likely on a GPU pod, then pulled back):
#   data/embeddings/mert/{covers80,discogs_vi}_embeddings.npz
#   data/embeddings/encodec/{covers80,discogs_vi}_codes.npz
#   data/semantic_ids/{covers80,discogs_vi}_encodec.csv   (from extract_encodec -> encodec_to_semid)
#   data/semantic_ids/{covers80,discogs_vi}_mert.csv      (from train_rqvae)
#
# Usage:
#   bash scripts/run_semids_local.sh

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p runs data/semantic_ids data/splits

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then
    echo "Auto-activating .venv"
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

banner() {
    echo
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
}

require() {
    if [ ! -f "$1" ]; then
        echo "MISSING: $1 — produce this artifact upstream before re-running."
        exit 1
    fi
}

# ----- preflight: make sure upstream artifacts exist -----
banner "Preflight"
require data/embeddings/mert/covers80_embeddings.npz
require data/embeddings/mert/discogs_vi_embeddings.npz
require data/embeddings/encodec/covers80_codes.npz
require data/embeddings/encodec/discogs_vi_codes.npz
require data/semantic_ids/covers80_encodec.csv
require data/semantic_ids/discogs_vi_encodec.csv
require data/semantic_ids/covers80_mert.csv
require data/semantic_ids/discogs_vi_mert.csv
require data/splits/covers80_cliques.csv
require data/splits/discogs_vi_subset.csv

# discogs_vi_audio.csv lives on the pod; rebuild it locally from subset CSV +
# MERT NPZ keys (the embeddings' track_ids are the successfully-crawled set).
if [ ! -f data/splits/discogs_vi_audio.csv ]; then
    echo "Deriving data/splits/discogs_vi_audio.csv from local subset CSV + MERT NPZ"
    python <<'PY'
import csv
import numpy as np

with open("data/splits/discogs_vi_subset.csv") as f:
    subset = {r["youtube_id"]: r["clique_id"] for r in csv.DictReader(f)}

with np.load("data/embeddings/mert/discogs_vi_embeddings.npz") as npz:
    yids = list(npz.files)

n_ok = 0
with open("data/splits/discogs_vi_audio.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["track_id", "clique_id", "filepath"])
    w.writeheader()
    for yid in yids:
        cid = subset.get(yid)
        if not cid:
            continue
        w.writerow({"track_id": yid, "clique_id": cid, "filepath": ""})
        n_ok += 1
print(f"Wrote data/splits/discogs_vi_audio.csv: {n_ok} rows "
      f"(from {len(yids)} MERT-extracted tracks, {len(subset)} subset rows).")
PY
fi
echo "All upstream artifacts present."

# ----- 1. random-ID baselines -----
banner "1. Random Semantic IDs (baselines)"
python -m src.semantic_ids.random_ids \
    --cliques-csv data/splits/discogs_vi_audio.csv \
    --out data/semantic_ids/discogs_vi_random.csv
python -m src.semantic_ids.random_ids \
    --cliques-csv data/splits/covers80_cliques.csv \
    --out data/semantic_ids/covers80_random.csv

# ----- 2. SemID quality diagnostics (Discogs-VI: full prefix-overlap signal) -----
banner "2. SemID diagnostics: Discogs-VI"
for cond in random mert encodec; do
    echo "-- $cond --"
    python -m src.semantic_ids.analyze_ids \
        --semids-csv "data/semantic_ids/discogs_vi_${cond}.csv" \
        --cliques-csv data/splits/discogs_vi_audio.csv \
        --out "runs/discogs_vi_${cond}_semid_report.json"
    echo
done

banner "3. SemID diagnostics: Covers80"
for cond in random mert encodec; do
    echo "-- $cond --"
    python -m src.semantic_ids.analyze_ids \
        --semids-csv "data/semantic_ids/covers80_${cond}.csv" \
        --cliques-csv data/splits/covers80_cliques.csv \
        --out "runs/covers80_${cond}_semid_report.json"
    echo
done

# ----- 3. train/val/test splits (Discogs-VI only — Covers80 is held-out cross-dataset test) -----
banner "4. Build train/val/test splits (Discogs-VI)"
for cond in random mert encodec; do
    echo "-- $cond --"
    python -m src.data.build_splits \
        --cliques-csv data/splits/discogs_vi_audio.csv \
        --semids-csv "data/semantic_ids/discogs_vi_${cond}.csv" \
        --out-dir data/splits \
        --name "discogs_vi_${cond}_ids"
    echo
done

# ----- summary -----
banner "Summary"
echo "SemID files:"
ls -lh data/semantic_ids/*.csv 2>/dev/null
echo
echo "Quality reports:"
ls -lh runs/*_semid_report.json 2>/dev/null
echo
echo "Train/val/test split files:"
ls -lh data/splits/discogs_vi_*_train.csv data/splits/discogs_vi_*_val.csv data/splits/discogs_vi_*_test.csv 2>/dev/null
echo
echo "Headline diagnostic (prefix-overlap rates): paste these reports back so we can"
echo "compare within-vs-cross prefix sharing across conditions."
echo "  for f in runs/discogs_vi_*_semid_report.json; do echo \"==== \$f ====\"; cat \"\$f\"; done"

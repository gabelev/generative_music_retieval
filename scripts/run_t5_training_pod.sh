#!/usr/bin/env bash
# T5-small fine-tuning + constrained-beam-search inference for all 3 SemID
# conditions on Discogs-VI. Run on a GPU pod.
#
# Prereqs (on the pod):
#   1. bash scripts/setup_pod.sh        (creates venv with torch+CUDA + transformers)
#   2. huggingface-cli login            (so we can pull metadata + push results)
#
# All metadata (splits CSVs, SemID CSVs, RQ-VAE codebooks) is pulled from the
# private HF dataset repo at the start of this script. No scp from laptop needed.
#
# Usage:
#   bash scripts/run_t5_training_pod.sh
#
# Optional env overrides:
#   HF_REPO=...  EPOCHS=10  BATCH=32  PUSH_RESULTS=1  bash scripts/run_t5_training_pod.sh
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p runs

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then
    echo "Auto-activating .venv"
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

HF_REPO=${HF_REPO:-gabelev/discogs-vi-csi-subset}
EPOCHS=${EPOCHS:-15}
BATCH=${BATCH:-64}
BEAM=${BEAM:-20}
TOP_K=${TOP_K:-20}
PUSH_RESULTS=${PUSH_RESULTS:-1}     # if 1, push runs/preds + RESULTS.md back to HF at the end

banner() { echo; echo "=== $1 ==="; }

banner "Pulling metadata + SemIDs + codebooks from HF ($HF_REPO)"
python -m src.data.sync_hf_metadata --hf-repo "$HF_REPO" --mode download

run_condition() {
    local cond="$1"
    local codebook_size="$2"
    banner "Training: $cond (codebook ${codebook_size})"
    python -m src.model.train \
        --train-csv "data/splits/discogs_vi_${cond}_ids_train.csv" \
        --val-csv   "data/splits/discogs_vi_${cond}_ids_val.csv" \
        --out-dir   "runs/t5_${cond}" \
        --codebook-size "$codebook_size" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH"

    banner "Inference (in-distribution test): $cond"
    python -m src.model.inference \
        --ckpt-dir   "runs/t5_${cond}" \
        --test-csv   "data/splits/discogs_vi_${cond}_ids_test.csv" \
        --semids-csv "data/semantic_ids/discogs_vi_${cond}.csv" \
        --out        "runs/preds_discogs_vi_${cond}.json" \
        --beam-width "$BEAM" --top-k "$TOP_K"

    banner "Inference (cross-dataset, Covers80): $cond"
    # Build a synthetic test CSV from covers80_cliques + covers80 SemIDs so
    # inference treats every Covers80 track as a query against the Covers80 SemID pool.
    python <<PY
import csv
sem = {}
with open("data/semantic_ids/covers80_${cond}.csv") as f:
    for r in csv.DictReader(f):
        sem[r["track_id"]] = (int(r["c1"]), int(r["c2"]), int(r["c3"]))
out_rows = []
with open("data/splits/covers80_cliques.csv") as f:
    by_clique = {}
    for r in csv.DictReader(f):
        by_clique.setdefault(r["clique_id"], []).append(r["track_id"])
for cid, tids in by_clique.items():
    for q in tids:
        for t in tids:
            if q == t or q not in sem or t not in sem:
                continue
            qs, ts = sem[q], sem[t]
            out_rows.append({
                "clique_id": cid,
                "query_track_id": q, "target_track_id": t,
                "query_c1": qs[0], "query_c2": qs[1], "query_c3": qs[2],
                "target_c1": ts[0], "target_c2": ts[1], "target_c3": ts[2],
            })
with open("data/splits/covers80_${cond}_test.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader(); w.writerows(out_rows)
print(f"Wrote data/splits/covers80_${cond}_test.csv ({len(out_rows)} pairs)")
PY
    python -m src.model.inference \
        --ckpt-dir   "runs/t5_${cond}" \
        --test-csv   "data/splits/covers80_${cond}_test.csv" \
        --semids-csv "data/semantic_ids/covers80_${cond}.csv" \
        --out        "runs/preds_covers80_${cond}.json" \
        --beam-width "$BEAM" --top-k "$TOP_K"
}

run_condition random  256
run_condition mert    256
run_condition encodec 1024

banner "Summary"
ls -lh runs/t5_*/pytorch_model.bin runs/t5_*/model.safetensors 2>/dev/null || true
ls -lh runs/preds_*.json

if [ "$PUSH_RESULTS" = "1" ]; then
    banner "Pushing predictions + logs back to HF"
    python -m src.data.sync_hf_metadata \
        --hf-repo "$HF_REPO" --mode upload \
        --paths runs
fi

echo "Done. On your Mac:"
echo "  python -m src.data.sync_hf_metadata --hf-repo $HF_REPO --mode download --paths runs"
echo "  # then run src.evaluation.evaluate for each preds_*.json"

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
BEAM_GROUPS=${BEAM_GROUPS:-4}        # >1 enables diverse beam search (anti-mode-collapse)
DIVERSITY=${DIVERSITY:-0.5}
SKIP_TRAIN=${SKIP_TRAIN:-0}          # if 1, ALL conditions skip training (re-inference only)
# ONLY_TRAIN: comma-separated list of conditions to actually train; others use existing
# HF checkpoints (per-condition skip). Empty = obey SKIP_TRAIN for all conditions.
# Example: ONLY_TRAIN=clews trains only the new CLEWS condition, reuses Phase-1 weights
# for random/mert/encodec. Phase 1 weights are pulled by the HF download step at the top.
ONLY_TRAIN=${ONLY_TRAIN:-}
PUSH_RESULTS=${PUSH_RESULTS:-1}      # if 1, push runs/preds + RESULTS.md back to HF at the end
# SPLIT_TAG selects which split family to train/eval on:
#   ids    -> transductive pair-split (default, the main result)
#   clique -> inductive clique-split (cold-start experiment)
SPLIT_TAG=${SPLIT_TAG:-ids}
SUFFIX=""
[ "$SPLIT_TAG" != "ids" ] && SUFFIX="_${SPLIT_TAG}"

banner() { echo; echo "=== $1 ==="; }

banner "Pulling metadata + SemIDs + codebooks from HF ($HF_REPO)"
python -m src.data.sync_hf_metadata --hf-repo "$HF_REPO" --mode download

run_condition() {
    local cond="$1"
    local codebook_size="$2"

    # Decide whether to train this condition this run.
    local do_train=1
    if [ "$SKIP_TRAIN" = "1" ]; then
        do_train=0
    elif [ -n "$ONLY_TRAIN" ]; then
        case ",${ONLY_TRAIN}," in
            *,${cond},*) do_train=1 ;;
            *) do_train=0 ;;
        esac
    fi

    if [ "$do_train" = "1" ]; then
        banner "Training: $cond (codebook ${codebook_size}, split=${SPLIT_TAG})"
        python -m src.model.train \
            --train-csv "data/splits/discogs_vi_${cond}_${SPLIT_TAG}_train.csv" \
            --val-csv   "data/splits/discogs_vi_${cond}_${SPLIT_TAG}_val.csv" \
            --out-dir   "runs/t5_${cond}${SUFFIX}" \
            --codebook-size "$codebook_size" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH"
    else
        banner "Reusing existing checkpoint: runs/t5_${cond}${SUFFIX} (pulled from HF)"
    fi

    banner "Inference (in-distribution test): $cond (split=${SPLIT_TAG}, beam=$BEAM, groups=$BEAM_GROUPS)"
    python -m src.model.inference \
        --ckpt-dir   "runs/t5_${cond}${SUFFIX}" \
        --test-csv   "data/splits/discogs_vi_${cond}_${SPLIT_TAG}_test.csv" \
        --semids-csv "data/semantic_ids/discogs_vi_${cond}.csv" \
        --out        "runs/preds_discogs_vi_${cond}${SUFFIX}.json" \
        --beam-width "$BEAM" --top-k "$TOP_K" \
        --num-beam-groups "$BEAM_GROUPS" --diversity-penalty "$DIVERSITY"

    # Covers80 cross-dataset inference only applies to the transductive run.
    if [ "$SPLIT_TAG" != "ids" ]; then
        return 0
    fi
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
        --beam-width "$BEAM" --top-k "$TOP_K" \
        --num-beam-groups "$BEAM_GROUPS" --diversity-penalty "$DIVERSITY"
}

run_condition random  256
run_condition mert    256
run_condition encodec 1024
run_condition clews   256

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

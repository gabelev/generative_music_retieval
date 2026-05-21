#!/usr/bin/env bash
# Full feature-extraction pipeline for the RunPod GPU phase.
# Runs end-to-end, idempotent (each sub-step is resume-safe). Detach-safe.
#
# Prereqs (do these once before running this script):
#   1. git clone <repo> && cd <repo>
#   2. python -m venv .venv && source .venv/bin/activate
#   3. pip install -r requirements.txt
#   4. huggingface-cli login          # paste your HF token
#   5. From your laptop, scp these two files into data/splits/ on the pod:
#        - discogs_vi_subset.csv      (needed for clique_id mapping)
#        - discogs_vi_download_log.csv (optional; tightens audio filter)
#
# Usage:
#   bash scripts/run_extraction_pod.sh
#
# Or inside tmux, with logging:
#   tmux new -s extract
#   bash scripts/run_extraction_pod.sh 2>&1 | tee runs/extraction.log
#   # detach with Ctrl-b d ; reattach with: tmux attach -t extract

set -euo pipefail

HF_REPO=${HF_REPO:-gabelev/discogs-vi-csi-subset}
BATCH_SIZE=${BATCH_SIZE:-8}
MERT_DTYPE=${MERT_DTYPE:-float16}
ENCODEC_BANDWIDTH=${ENCODEC_BANDWIDTH:-3.0}

cd "$(dirname "$0")/.."
mkdir -p runs data/embeddings/mert data/embeddings/encodec data/semantic_ids data/splits

banner() {
    echo
    echo "================================================================"
    echo "  $1"
    echo "  $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo "================================================================"
}

time_step() {
    local label="$1"; shift
    local t0=$SECONDS
    echo
    echo "[$label] $@"
    "$@"
    local dt=$((SECONDS - t0))
    echo "[$label] done in ${dt}s"
}

# ----- preflight -----
banner "Preflight"
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python -c "import transformers, torchaudio, encodec, huggingface_hub" \
    || { echo "Missing deps. pip install -r requirements.txt"; exit 1; }
test -n "${HF_TOKEN:-}" || test -f "$HOME/.cache/huggingface/token" \
    || { echo "HF auth missing. Run: huggingface-cli login"; exit 1; }

# ----- 1. Covers80 -----
banner "1. Covers80 dataset"
time_step covers80 python -m src.data.download_covers80

# ----- 2. Discogs-VI audio from HF -----
banner "2. Discogs-VI audio (HF snapshot)"
if [ ! -f data/splits/discogs_vi_subset.csv ]; then
    echo "FATAL: data/splits/discogs_vi_subset.csv missing on the pod."
    echo "Copy it from your laptop with: scp data/splits/discogs_vi_subset.csv <pod>:..."
    exit 1
fi
time_step prep_discogs python -m src.data.prep_discogs_audio --hf-repo "$HF_REPO"

# ----- 3. MERT extraction -----
banner "3. MERT embeddings: Covers80"
time_step mert_covers80 python -m src.features.extract_mert \
    --cliques-csv data/splits/covers80_cliques.csv \
    --out data/embeddings/mert/covers80_embeddings.npz \
    --batch-size "$BATCH_SIZE" --dtype "$MERT_DTYPE"

banner "4. MERT embeddings: Discogs-VI"
time_step mert_discogs python -m src.features.extract_mert \
    --cliques-csv data/splits/discogs_vi_audio.csv \
    --out data/embeddings/mert/discogs_vi_embeddings.npz \
    --batch-size "$BATCH_SIZE" --dtype "$MERT_DTYPE"

# ----- 4. EnCodec extraction -----
banner "5. EnCodec codes: Covers80"
time_step encodec_covers80 python -m src.features.extract_encodec \
    --cliques-csv data/splits/covers80_cliques.csv \
    --out data/embeddings/encodec/covers80_codes.npz \
    --bandwidth "$ENCODEC_BANDWIDTH" \
    --batch-size "$BATCH_SIZE"

banner "6. EnCodec codes: Discogs-VI"
time_step encodec_discogs python -m src.features.extract_encodec \
    --cliques-csv data/splits/discogs_vi_audio.csv \
    --out data/embeddings/encodec/discogs_vi_codes.npz \
    --bandwidth "$ENCODEC_BANDWIDTH" \
    --batch-size "$BATCH_SIZE"

# ----- 5. EnCodec frame codes -> per-track SemID -----
banner "7. EnCodec -> SemID (majority vote)"
time_step semid_covers80 python -m src.semantic_ids.encodec_to_semid \
    --codes-npz data/embeddings/encodec/covers80_codes.npz \
    --out data/semantic_ids/covers80_encodec.csv \
    --strategy majority_vote
time_step semid_discogs python -m src.semantic_ids.encodec_to_semid \
    --codes-npz data/embeddings/encodec/discogs_vi_codes.npz \
    --out data/semantic_ids/discogs_vi_encodec.csv \
    --strategy majority_vote

# ----- final summary -----
banner "Summary"
echo "Artifacts (download these to your laptop after the pod runs):"
ls -lh data/embeddings/mert/*.npz \
       data/embeddings/encodec/*.npz \
       data/semantic_ids/*_encodec.csv 2>/dev/null || true
echo
echo "Total disk used by extracted features:"
du -sh data/embeddings data/semantic_ids 2>/dev/null || true
echo
echo "Done. Total wall time: ${SECONDS}s."

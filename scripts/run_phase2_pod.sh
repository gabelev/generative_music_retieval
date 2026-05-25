#!/usr/bin/env bash
# Phase 2 pod orchestration: extract CLEWS embeddings, build CLEWS SemIDs,
# train T5-CLEWS, re-infer all 4 conditions at beam width 200 for NAR-grade
# predictions, push everything to HF.
#
# Prereqs (on a fresh pod):
#   1. bash scripts/setup_pod.sh        (creates .venv with torch+CUDA + transformers)
#   2. huggingface-cli login
#
# Usage (in tmux):
#   bash scripts/run_phase2_pod.sh
#
# Tunables (env vars):
#   HF_REPO              gabelev/discogs-vi-csi-subset
#   CLEWS_REPO_URL       https://github.com/sony/clews
#   CLEWS_CHECKPOINT     path to .ckpt (auto-located if empty; fall back to DVINet+)
#   BEAM                 200 (for top-200 NAR-grade predictions)
#   TOP_K                200
#
# Cost estimate (L40S): ~3-4h total.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p runs data/embeddings/clews data/audio/clews_input cache/clews

HF_REPO=${HF_REPO:-gabelev/discogs-vi-csi-subset}
CLEWS_REPO_URL=${CLEWS_REPO_URL:-https://github.com/sony/clews}
CLEWS_DIR=${CLEWS_DIR:-/workspace/clews}
CLEWS_CHECKPOINT=${CLEWS_CHECKPOINT:-}
BEAM=${BEAM:-200}
TOP_K=${TOP_K:-200}

banner() { echo; echo "============================================================"; echo "  $1"; echo "  $(date -u +'%Y-%m-%dT%H:%M:%SZ')"; echo "============================================================"; }

# Make sure we're inside the main project venv for our code (not for CLEWS).
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then
    echo "Auto-activating .venv (main project env)"
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi
MAIN_PY=$(which python)
echo "Main project Python: $MAIN_PY"

# ----- Phase A: project metadata + audio -----
banner "A1. Pull project metadata from HF"
python -m src.data.sync_hf_metadata --hf-repo "$HF_REPO" --mode download

banner "A2. Pull Discogs-VI audio from HF (3,413 WAVs, ~5GB)"
python -m src.data.prep_discogs_audio --hf-repo "$HF_REPO" --out-dir data/raw/discogs_vi

banner "A3. Download Covers80 (LabROSA tarball, 160 mp3s)"
python -m src.data.download_covers80

# Stage all audio paths into one CLEWS-input dir as <track_id>.wav (symlinks).
banner "A4. Stage all audio under cache/clews_input/<track_id>.wav as symlinks"
python <<'PY'
import csv, os
from pathlib import Path
inp = Path("cache/clews_input"); inp.mkdir(parents=True, exist_ok=True)
n = 0
# Discogs-VI
audio_dir = Path("data/raw/discogs_vi/audio")
for p in audio_dir.glob("*.wav"):
    dst = inp / p.name
    if not dst.exists():
        os.symlink(p.resolve(), dst)
        n += 1
# Covers80 (mp3s; CLEWS handles non-wav typically, else we'd need ffmpeg conversion)
with open("data/splits/covers80_cliques.csv") as f:
    for r in csv.DictReader(f):
        src = Path(r["filepath"])
        if not src.exists():
            continue
        dst = inp / f"{r['track_id']}{src.suffix.lower()}"
        if not dst.exists():
            os.symlink(src.resolve(), dst)
            n += 1
print(f"staged {n} new symlinks; total files in cache/clews_input: "
      f"{sum(1 for _ in inp.iterdir())}")
PY

# ----- Phase B: CLEWS environment + inference -----
banner "B1. Clone CLEWS repo + install in a separate venv (.venv-clews)"
if [ ! -d "$CLEWS_DIR" ]; then
    git clone "$CLEWS_REPO_URL" "$CLEWS_DIR"
fi
PROJECT_DIR="$(pwd)"
CLEWS_PIP="$PROJECT_DIR/.venv-clews/bin/pip"
CLEWS_PY="$PROJECT_DIR/.venv-clews/bin/python"
if [ ! -d .venv-clews ]; then
    python -m venv .venv-clews
    "$CLEWS_PIP" install --upgrade pip
fi
# CLEWS dependencies: install via its requirements.txt or install_requirements.sh.
# Re-running is idempotent — pip skips already-installed packages.
if [ -f "$CLEWS_DIR/requirements.txt" ]; then
    "$CLEWS_PIP" install -r "$CLEWS_DIR/requirements.txt"
elif [ -f "$CLEWS_DIR/install_requirements.sh" ]; then
    # Run install_requirements.sh with our venv's bin/ first on PATH so its
    # `pip install ...` and `python ...` calls resolve to our venv (not system).
    (cd "$CLEWS_DIR" && PATH="$PROJECT_DIR/.venv-clews/bin:$PATH" bash install_requirements.sh)
fi
# Always make sure torch + torchaudio with CUDA are present (in case requirements.txt pinned CPU torch).
"$CLEWS_PIP" install --index-url https://download.pytorch.org/whl/cu124 torch torchaudio

banner "B2. Locate or download the CLEWS DVI checkpoint"
if [ -z "$CLEWS_CHECKPOINT" ]; then
    # Search common locations.
    for cand in "$CLEWS_DIR"/logs/model/checkpoint_best.ckpt \
                "$CLEWS_DIR"/models/dvi.ckpt \
                "$CLEWS_DIR"/checkpoints/*.ckpt; do
        if [ -f "$cand" ]; then CLEWS_CHECKPOINT="$cand"; break; fi
    done
fi
if [ -z "$CLEWS_CHECKPOINT" ] || [ ! -f "$CLEWS_CHECKPOINT" ]; then
    echo "WARN: CLEWS checkpoint not found locally. See CLEWS README for download link, OR fall back to DVINet+:"
    echo "      https://github.com/raraz15/Discogs-VINet"
    echo "      Then re-run this script with CLEWS_CHECKPOINT=<path>."
    exit 1
fi
echo "Using CLEWS checkpoint: $CLEWS_CHECKPOINT"

banner "B3. Smoke test CLEWS on 5 audio files"
mkdir -p "$PROJECT_DIR/cache/clews_smoke" "$PROJECT_DIR/cache/clews_smoke_input"
# `ls | head -5` under `set -o pipefail` bails because head closes the pipe early
# (ls exits 141 / SIGPIPE). Avoid the pipeline:
ls "$PROJECT_DIR/cache/clews_input" > /tmp/clews_all_keys.txt
head -5 /tmp/clews_all_keys.txt > /tmp/clews_smoke_keys.txt
while read -r f; do
    ln -sf "$PROJECT_DIR/cache/clews_input/$f" "$PROJECT_DIR/cache/clews_smoke_input/$f"
done < /tmp/clews_smoke_keys.txt
(cd "$CLEWS_DIR" && OMP_NUM_THREADS=1 "$CLEWS_PY" inference.py \
    --checkpoint="$CLEWS_CHECKPOINT" \
    --path_in="$PROJECT_DIR/cache/clews_smoke_input" \
    --path_out="$PROJECT_DIR/cache/clews_smoke") || {
        echo "Smoke test failed. Likely CLEWS sample-rate / clip-length issue."
        echo "  - CLEWS expects 16kHz mono, 2.5min blocks; ours is 24kHz mono, 30-sec."
        echo "  - Manual fix: resample to 16kHz and repeat-pad to 2.5min in cache/clews_input/."
        exit 1
}
echo "Smoke test OK. Files in cache/clews_smoke:"
ls "$PROJECT_DIR/cache/clews_smoke"

banner "B4. CLEWS inference on full corpus (~3,573 files)"
(cd "$CLEWS_DIR" && OMP_NUM_THREADS=1 "$CLEWS_PY" inference.py \
    --checkpoint="$CLEWS_CHECKPOINT" \
    --path_in="$PROJECT_DIR/cache/clews_input" \
    --path_out="$PROJECT_DIR/cache/clews")

banner "B5. Aggregate CLEWS segments -> NPZ (mean-pool)"
# Discogs-VI: allow keys = track IDs present in discogs_vi_audio.csv
python <<'PY'
import csv
with open("data/splits/discogs_vi_audio.csv") as f:
    keys = sorted({r["track_id"] for r in csv.DictReader(f)})
with open("/tmp/discogs_vi_keys.txt", "w") as out:
    out.write("\n".join(keys))
print(f"discogs_vi keys: {len(keys)}")
PY
python -m src.features.extract_clews \
    --clews-dir cache/clews \
    --out data/embeddings/clews/discogs_vi_embeddings.npz \
    --allow-keys /tmp/discogs_vi_keys.txt

# Covers80
python <<'PY'
import csv
with open("data/splits/covers80_cliques.csv") as f:
    keys = sorted({r["track_id"] for r in csv.DictReader(f)})
with open("/tmp/covers80_keys.txt", "w") as out:
    out.write("\n".join(keys))
print(f"covers80 keys: {len(keys)}")
PY
python -m src.features.extract_clews \
    --clews-dir cache/clews \
    --out data/embeddings/clews/covers80_embeddings.npz \
    --allow-keys /tmp/covers80_keys.txt

banner "B6. Push CLEWS embeddings to HF"
python -m src.data.sync_hf_metadata --hf-repo "$HF_REPO" --mode upload \
    --paths data/embeddings/clews \
    --patterns "*.npz"

# ----- Phase C: RQ-VAE + SemID + splits -----
banner "C1. Train RQ-VAE on Discogs-VI CLEWS embeddings"
python -m src.semantic_ids.train_rqvae \
    --embeddings data/embeddings/clews/discogs_vi_embeddings.npz \
    --out-codebooks data/semantic_ids/clews_codebooks.pkl \
    --out-semids   data/semantic_ids/discogs_vi_clews.csv

banner "C2. Encode Covers80 with the same codebooks"
python -m src.semantic_ids.train_rqvae \
    --embeddings   data/embeddings/clews/covers80_embeddings.npz \
    --train-embeddings data/embeddings/clews/discogs_vi_embeddings.npz \
    --out-codebooks data/semantic_ids/clews_codebooks_covers80_view.pkl \
    --out-semids   data/semantic_ids/covers80_clews.csv

banner "C3. SemID quality diagnostics"
python -m src.semantic_ids.analyze_ids \
    --semids-csv data/semantic_ids/discogs_vi_clews.csv \
    --cliques-csv data/splits/discogs_vi_audio.csv \
    --out runs/discogs_vi_clews_semid_report.json
python -m src.semantic_ids.analyze_ids \
    --semids-csv data/semantic_ids/covers80_clews.csv \
    --cliques-csv data/splits/covers80_cliques.csv \
    --out runs/covers80_clews_semid_report.json

banner "C4. Build transductive splits for CLEWS condition"
python -m src.data.build_splits \
    --cliques-csv data/splits/discogs_vi_audio.csv \
    --semids-csv data/semantic_ids/discogs_vi_clews.csv \
    --out-dir data/splits \
    --name discogs_vi_clews_ids \
    --split-by pair

banner "C5. Push splits + SemIDs + codebooks + reports to HF"
python -m src.data.sync_hf_metadata --hf-repo "$HF_REPO" --mode upload \
    --paths data/splits data/semantic_ids runs

# ----- Phase D: T5 training + inference (delegate to existing script) -----
banner "D. T5: train CLEWS only, re-infer all 4 conditions at BEAM=$BEAM"
ONLY_TRAIN=clews BEAM=$BEAM TOP_K=$TOP_K \
    bash scripts/run_t5_training_pod.sh

banner "Phase 2 complete."

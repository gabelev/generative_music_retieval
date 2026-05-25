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

# Per-phase skip flags. Set to 1 to force-skip a phase; default (empty/0) lets
# the script auto-detect whether the phase's outputs already exist and skip in
# that case. SKIP_ALL_AUTO=1 enables auto-skip for everything that has output;
# useful for resuming after a transient failure without redoing prior work.
SKIP_HF_META=${SKIP_HF_META:-0}                  # A1: metadata sync (cheap, default 0)
SKIP_AUDIO=${SKIP_AUDIO:-auto}                   # A2: Discogs-VI audio
SKIP_COVERS80=${SKIP_COVERS80:-auto}             # A3: Covers80 tarball
SKIP_STAGE=${SKIP_STAGE:-auto}                   # A4: cache/clews_input symlinks
SKIP_CLEWS_INSTALL=${SKIP_CLEWS_INSTALL:-auto}   # B1: clone CLEWS + venv + deps + torch<2.6
SKIP_CLEWS_INFERENCE=${SKIP_CLEWS_INFERENCE:-auto} # B3 + B4
SKIP_AGGREGATE=${SKIP_AGGREGATE:-auto}           # B5: extract_clews.py -> NPZ
SKIP_PUSH_CLEWS=${SKIP_PUSH_CLEWS:-0}            # B6: push CLEWS NPZs to HF (cheap, default 0)
SKIP_RQVAE=${SKIP_RQVAE:-auto}                   # C: RQ-VAE + analyze + splits
SKIP_T5=${SKIP_T5:-0}                            # D: T5 train + re-infer (always rerun by default)

# Helper: returns 0 (skip) if either explicit SKIP_<X>=1 or (SKIP_<X>=auto/empty
# AND the auto-detect command exits 0).
should_skip() {
    local flag_value="$1"; shift
    local label="$1"; shift
    if [ "$flag_value" = "1" ]; then
        echo "[skip] $label (explicit SKIP=1)"
        return 0
    fi
    if [ "$flag_value" = "auto" ] || [ -z "$flag_value" ]; then
        if "$@"; then
            echo "[skip] $label (auto-detected as already done)"
            return 0
        fi
    fi
    return 1
}

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
if ! should_skip "$SKIP_HF_META" "A1. HF metadata sync" false; then
    banner "A1. Pull project metadata from HF"
    python -m src.data.sync_hf_metadata --hf-repo "$HF_REPO" --mode download
fi

# A2 auto-skips when audio dir has > 3,400 wavs (target is 3,413).
audio_done() {
    [ -d data/raw/discogs_vi/audio ] && \
    [ "$(ls data/raw/discogs_vi/audio 2>/dev/null | wc -l)" -gt 3400 ]
}
if ! should_skip "$SKIP_AUDIO" "A2. Discogs-VI audio from HF" audio_done; then
    banner "A2. Pull Discogs-VI audio from HF (3,413 WAVs, ~5GB)"
    python -m src.data.prep_discogs_audio --hf-repo "$HF_REPO" --out-dir data/raw/discogs_vi
fi

# A3 auto-skips when Covers80 cliques CSV exists.
covers80_done() { [ -f data/splits/covers80_cliques.csv ]; }
if ! should_skip "$SKIP_COVERS80" "A3. Covers80 tarball" covers80_done; then
    banner "A3. Download Covers80 (LabROSA tarball, 160 mp3s)"
    python -m src.data.download_covers80
fi

# A4 auto-skips when cache/clews_input has > 3,400 symlinks.
stage_done() {
    [ -d cache/clews_input ] && \
    [ "$(ls cache/clews_input 2>/dev/null | wc -l)" -gt 3400 ]
}
if ! should_skip "$SKIP_STAGE" "A4. Stage audio symlinks" stage_done; then
    banner "A4. Stage all audio under cache/clews_input/<track_id>.wav as symlinks"
    python <<'PY'
import csv, os
from pathlib import Path
inp = Path("cache/clews_input"); inp.mkdir(parents=True, exist_ok=True)
n = 0
audio_dir = Path("data/raw/discogs_vi/audio")
for p in audio_dir.glob("*.wav"):
    dst = inp / p.name
    if not dst.exists():
        os.symlink(p.resolve(), dst); n += 1
with open("data/splits/covers80_cliques.csv") as f:
    for r in csv.DictReader(f):
        src = Path(r["filepath"])
        if not src.exists():
            continue
        dst = inp / f"{r['track_id']}{src.suffix.lower()}"
        if not dst.exists():
            os.symlink(src.resolve(), dst); n += 1
print(f"staged {n} new symlinks; total files in cache/clews_input: "
      f"{sum(1 for _ in inp.iterdir())}")
PY
fi

# ----- Phase B: CLEWS environment + inference -----
PROJECT_DIR="$(pwd)"
CLEWS_PIP="$PROJECT_DIR/.venv-clews/bin/pip"
CLEWS_PY="$PROJECT_DIR/.venv-clews/bin/python"

# Auto-skip B1 if torch is already pinned <2.6 in .venv-clews AND inference.py imports cleanly.
clews_install_done() {
    [ -x "$CLEWS_PY" ] || return 1
    [ -d "$CLEWS_DIR" ] || return 1
    local v
    v=$("$CLEWS_PY" -c "import torch, sys; sys.exit(0 if torch.__version__.split('+')[0] < '2.6' else 1)" 2>/dev/null) || return 1
    "$CLEWS_PY" -c "import sys; sys.path.insert(0, '$CLEWS_DIR'); import inference" 2>/dev/null
}
if ! should_skip "$SKIP_CLEWS_INSTALL" "B1. CLEWS install (clone + venv + deps + torch<2.6)" clews_install_done; then
    banner "B1. Clone CLEWS repo + install in a separate venv (.venv-clews)"
    if [ ! -d "$CLEWS_DIR" ]; then
        git clone "$CLEWS_REPO_URL" "$CLEWS_DIR"
    fi
    if [ ! -d .venv-clews ]; then
        python -m venv .venv-clews
        "$CLEWS_PIP" install --upgrade pip
    fi
    if [ -f "$CLEWS_DIR/requirements.txt" ]; then
        "$CLEWS_PIP" install -r "$CLEWS_DIR/requirements.txt"
    elif [ -f "$CLEWS_DIR/install_requirements.sh" ]; then
        (cd "$CLEWS_DIR" && PATH="$PROJECT_DIR/.venv-clews/bin:$PATH" bash install_requirements.sh)
    fi
    # Pin torch < 2.6 in the clews venv: PyTorch 2.6 changed torch.load to default
    # weights_only=True, which breaks CLEWS's checkpoint load.
    "$CLEWS_PIP" install --force-reinstall --index-url https://download.pytorch.org/whl/cu124 \
        "torch<2.6" "torchaudio<2.6"
    "$CLEWS_PIP" install omegaconf hydra-core lightning nnAudio einops librosa
fi

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

clews_inference_done() {
    [ -d "$PROJECT_DIR/cache/clews" ] && \
    [ "$(ls "$PROJECT_DIR/cache/clews" 2>/dev/null | grep -c '\.pt$')" -gt 3000 ]
}
if ! should_skip "$SKIP_CLEWS_INFERENCE" "B3 + B4. CLEWS smoke test + full inference" clews_inference_done; then
    banner "B3. Smoke test CLEWS on 5 audio files"
    rm -rf "$PROJECT_DIR/cache/clews_smoke" "$PROJECT_DIR/cache/clews_smoke_input"
    mkdir -p "$PROJECT_DIR/cache/clews_smoke_input"
    ls "$PROJECT_DIR/cache/clews_input" > /tmp/clews_all_keys.txt
    head -5 /tmp/clews_all_keys.txt > /tmp/clews_smoke_keys.txt
    while read -r f; do
        ln -sf "$PROJECT_DIR/cache/clews_input/$f" "$PROJECT_DIR/cache/clews_smoke_input/$f"
    done < /tmp/clews_smoke_keys.txt
    (cd "$CLEWS_DIR" && OMP_NUM_THREADS=1 "$CLEWS_PY" inference.py \
        --checkpoint="$CLEWS_CHECKPOINT" \
        --path_in="$PROJECT_DIR/cache/clews_smoke_input" \
        --path_out="$PROJECT_DIR/cache/clews_smoke" < /dev/null) || {
            echo "Smoke test failed. Likely CLEWS sample-rate / clip-length issue."
            echo "  - CLEWS expects 16kHz mono, 2.5min blocks; ours is 24kHz mono, 30-sec."
            echo "  - Manual fix: resample to 16kHz and repeat-pad to 2.5min in cache/clews_input/."
            exit 1
    }
    echo "Smoke test OK. Files in cache/clews_smoke:"
    ls "$PROJECT_DIR/cache/clews_smoke"

    banner "B4. CLEWS inference on full corpus (~3,573 files)"
    rm -rf "$PROJECT_DIR/cache/clews"
    (cd "$CLEWS_DIR" && OMP_NUM_THREADS=1 "$CLEWS_PY" inference.py \
        --checkpoint="$CLEWS_CHECKPOINT" \
        --path_in="$PROJECT_DIR/cache/clews_input" \
        --path_out="$PROJECT_DIR/cache/clews" < /dev/null)
fi

aggregate_done() {
    [ -f data/embeddings/clews/discogs_vi_embeddings.npz ] && \
    [ -f data/embeddings/clews/covers80_embeddings.npz ]
}
if ! should_skip "$SKIP_AGGREGATE" "B5. NPZ aggregation" aggregate_done; then
    banner "B5. Aggregate CLEWS segments -> NPZ (mean-pool)"
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
fi

if ! should_skip "$SKIP_PUSH_CLEWS" "B6. Push CLEWS embeddings to HF" false; then
    banner "B6. Push CLEWS embeddings to HF"
    python -m src.data.sync_hf_metadata --hf-repo "$HF_REPO" --mode upload \
        --paths data/embeddings/clews \
        --patterns "*.npz"
fi

# ----- Phase C: RQ-VAE + SemID + splits -----
rqvae_done() {
    [ -f data/semantic_ids/clews_codebooks.pkl ] && \
    [ -f data/semantic_ids/discogs_vi_clews.csv ] && \
    [ -f data/semantic_ids/covers80_clews.csv ] && \
    [ -f data/splits/discogs_vi_clews_ids_train.csv ]
}
if ! should_skip "$SKIP_RQVAE" "C. RQ-VAE + SemID + splits + sync" rqvae_done; then
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
fi

# ----- Phase D: T5 training + inference (delegate to existing script) -----
if [ "$SKIP_T5" = "1" ]; then
    echo "[skip] D. T5 train + re-infer (SKIP_T5=1)"
else
    banner "D. T5: train CLEWS only, re-infer all 4 conditions at BEAM=$BEAM"
    ONLY_TRAIN=clews BEAM=$BEAM TOP_K=$TOP_K \
        bash scripts/run_t5_training_pod.sh
fi

banner "Phase 2 complete."

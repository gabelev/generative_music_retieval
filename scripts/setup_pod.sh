#!/usr/bin/env bash
# One-shot pod environment setup: clean venv, GPU-enabled torch, all Python deps.
# Idempotent (re-running is safe). Run BEFORE scripts/run_extraction_pod.sh.
#
# Usage:
#   bash scripts/setup_pod.sh
#
# Override with env vars:
#   PYTHON=python3.11 TORCH_INDEX=https://download.pytorch.org/whl/cu124 bash scripts/setup_pod.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}

echo "=== System libs (audio backend deps for torchaudio) ==="
if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq libsndfile1 ffmpeg || \
        echo "  warn: apt-get install failed; continue if libsndfile1 + ffmpeg already present"
fi

echo "=== Cleaning stale venv ==="
rm -rf .venv

echo "=== Creating fresh venv (isolated from system) ==="
$PYTHON -m venv .venv
.venv/bin/pip install --upgrade pip

echo "=== Installing torch + torchaudio with CUDA wheel ($TORCH_INDEX) ==="
.venv/bin/pip install --index-url "$TORCH_INDEX" torch torchaudio

echo "=== Installing remaining Python deps ==="
# Avoid double-installing torch/torchaudio (already installed from CUDA index)
TMP_REQ=$(mktemp)
grep -vE "^(torch|torchaudio)" requirements.txt > "$TMP_REQ"
.venv/bin/pip install -r "$TMP_REQ"
.venv/bin/pip install --upgrade "transformers==4.46.3" "accelerate>=0.26" "soundfile>=0.12"
rm -f "$TMP_REQ"

echo "=== Verifying ==="
.venv/bin/python << 'EOF'
import torch, torchaudio, transformers, huggingface_hub, encodec
print(f"torch           : {torch.__version__}  cuda? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu             : {torch.cuda.get_device_name(0)}")
print(f"torchaudio      : {torchaudio.__version__}")
print(f"transformers    : {transformers.__version__}")
print(f"huggingface_hub : {huggingface_hub.__version__}")
print(f"encodec         : present")
from transformers import AutoModel, Wav2Vec2FeatureExtractor
print("imports OK: AutoModel + Wav2Vec2FeatureExtractor")
EOF

echo
echo "=== Setup complete ==="
echo "Now run: nohup bash scripts/run_extraction_pod.sh > runs/extraction.log 2>&1 & disown"

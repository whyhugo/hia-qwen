#!/usr/bin/env bash
# RunPod environment setup — run this ONCE on a cheap CPU pod (with the
# persistent network volume mounted at /workspace). It builds the venv and
# pre-downloads the Qwen2-Audio model onto the volume so the GPU pod can start
# training immediately without paying GPU time for downloads.
set -euo pipefail

REPO_DIR="${1:-/workspace/hia-qwen}"
cd "$REPO_DIR"

echo "=== [1/4] Python venv ==="
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

echo "=== [2/4] Install requirements ==="
python -m pip install -r requirements.txt

echo "=== [3/4] Pre-download Qwen2-Audio-7B-Instruct to HF cache (on the volume) ==="
# Use the Python API (version-robust). Do NOT `pip install -U huggingface_hub` —
# that can pull hub 1.x, which transformers<5 rejects and which renames the CLI.
export HF_HOME="${HF_HOME:-$REPO_DIR/.hf_cache}"
python - <<'PY'
from huggingface_hub import snapshot_download
print("downloaded to:", snapshot_download("Qwen/Qwen2-Audio-7B-Instruct"))
PY
echo "HF_HOME=$HF_HOME  (export this in the GPU pod too)"

echo "=== [4/4] Sanity: bundled data + HIA code load ==="
PYTHONPATH=src python - <<'PY'
import numpy as np, torch
from hia_qwen.hia_features import import_hia_class
HIA = import_hia_class("external/apa-hia-framework")
torch.load("external/hia_ckpt/best_audio_model.pth", map_location="cpu")
f = np.load("external/seq_data_librispeech/tr_feat.npy", mmap_mode="r")
ids = [l.split()[0] for l in open("external/speechocean762/train/wav.scp") if l.strip()]
assert f.shape[0] == len(ids), "feat/wav.scp row mismatch"
print(f"OK: HIA={HIA.__name__}, tr_feat={f.shape}, ids={len(ids)}")
PY

echo ""
echo "=== Setup complete. Stop this CPU pod, then start a GPU pod on the SAME volume. ==="

# Running HIA-Qwen on RunPod

The repo is self-contained for RunPod: all data + the HIA model code are bundled
under `external/` (~99 MB). The only thing fetched at runtime is the
Qwen2-Audio-7B-Instruct base model (~16 GB) from HuggingFace.

The workflow splits cheap setup (CPU) from expensive compute (GPU) using a
**persistent Network Volume** so you don't pay GPU rates for `pip install` and a
16 GB model download.

---

## 0. One-time: create a Network Volume

RunPod console → **Storage → Network Volumes → New**. Size **≥ 60 GB**
(repo ~0.1 GB + venv ~8 GB + Qwen ~16 GB + outputs ~5 GB, with headroom).
Pick a region/datacenter that also offers the GPU you want (volume and pod must
share a datacenter).

---

## 1. CPU pod — build environment + download model

Deploy a pod with the network volume attached at `/workspace`:
- Template: any PyTorch or CUDA image (e.g. `runpod/pytorch`), **2–8 vCPU, no GPU**.
- This is cheap (~$0.05–0.15/hr) and only runs for the setup.

In the pod's web terminal:

```bash
cd /workspace
# Private repo — use a GitHub token (or deploy key). Do NOT paste a token that
# you also use elsewhere; create a fine-scoped one for this.
git clone https://<GITHUB_TOKEN>@github.com/whyhugo/hia-qwen.git
cd hia-qwen

# Build venv, install deps, pre-download Qwen onto the volume, sanity-check data
bash scripts/runpod_setup.sh /workspace/hia-qwen
```

When it prints `Setup complete`, **stop (not terminate) the CPU pod**. The volume
keeps the venv, the bundled data, and the Qwen cache.

---

## 2. GPU pod — run training + eval

Deploy a new pod **on the same network volume** (`/workspace`).

### Which GPU?

| GPU | VRAM | Speed | Notes |
|-----|------|------:|-------|
| **A100 80GB** (recommended) | 80 GB | fastest, zero OOM risk | best for "speed up the experiment"; ~$1.5–2/hr |
| L40S | 48 GB | fast | good middle ground |
| RTX 4090 | 24 GB | ok | cheapest that fits; 4-bit needs `min_free_vram_gb: 22`, so it *just* fits — no headroom |

This is 4-bit QLoRA of a 7B model. It fits on 24 GB, but Stage 2 is 6 epochs /
15k steps (~hours). For wall-clock speed pick **A100 80GB**; for cost pick **4090**.
Avoid <24 GB cards.

### Run

```bash
cd /workspace/hia-qwen
source .venv/bin/activate
export HF_HOME=/workspace/hia-qwen/.hf_cache   # reuse the model cached in step 1

# Stage 1: multi-granularity alignment (projectors only). MUST finish & save projector.pt
python scripts/train_alignment.py --config configs/alignment_stage1_runpod.yaml \
    2>&1 | tee outputs/alignment_stage1_multi/run.log
test -f outputs/alignment_stage1_multi/projector.pt \
    && echo "✅ Stage 1 done" || { echo "❌ Stage 1 did not save projector.pt — check run.log"; exit 1; }

# Stage 2: joint projector + LoRA (hard-fails if Stage 1 projector is missing)
python scripts/train_joint_lora.py --config configs/joint_lora_stage2_runpod.yaml \
    2>&1 | tee outputs/joint_lora_stage2_multi/run.log

# Eval on SO762 test
python scripts/evaluate_hia_qwen.py --config configs/eval_stage2_runpod.yaml

# Inspect results
cat outputs/joint_lora_stage2_multi/eval_test/metrics.json
```

Tip: run long jobs under `tmux`/`nohup` so a dropped web terminal doesn't kill them.

---

## 3. Get results back

Results land under `outputs/` on the volume. Either:
- `git add outputs/joint_lora_stage2_multi/eval_test outputs/alignment_stage1_multi/run.log && git commit && git push`
  (note: `outputs/` is gitignored by default — use `git add -f`), or
- `runpodctl send` / `scp` the `outputs/.../metrics.json` files.

Then **stop the GPU pod** so you stop paying for it.

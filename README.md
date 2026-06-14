# HIA-Qwen APA Experiment

This repository implements a two-stage experiment that fuses the HIA pronunciation-assessment frontend with a Qwen2-Audio language backend through learnable soft prompts.

The experiment does not feed raw audio into Qwen. HIA reads GOP features and phone ids, then its hidden states are projected into Qwen's embedding space and injected as continuous soft tokens.

## Architecture

```text
GOP + phone ids -> frozen HIA -> HIA hidden states
                                -> projector MLPs -> Qwen embedding space
text prompt + HIA soft tokens   -> Qwen2-Audio decoder
                                -> APA scores
```

| Stage | Objective | Trainable | Frozen | Target |
| --- | --- | --- | --- | --- |
| Stage 1 | Alignment pre-training | HIA-to-Qwen projectors | HIA, Qwen2-Audio | `Score: {Score}` sentence total |
| Stage 2 | Joint LoRA fine-tuning | Projectors + Qwen LoRA | HIA, Qwen base weights | Multi-level JSON scores |

## HIA Feature Definition

The word-level soft prompt is not the IAM global `H_word [B,1,D]` token. The implementation extracts the HIA Word Branch hidden state:

```text
F_word = word_norm(word_conv((X + phn_residual + H_word).T).T)  # [B,L,D]
```

Then `F_word` is mean-pooled by `word_id` from `label_word[:,:,3]` to produce `[B,T_wrd,D]`, preserving the dynamic number of words per utterance.

Feature streams:

- Phone: valid positions from `F_phn [B,L,D]` -> `[B,T_phn,D]`
- Word: pooled Word Branch `F_word [B,L,D]` -> `[B,T_wrd,D]`
- Utterance: utterance branch `dec_out [B,1,D]`

## Repository Layout

```text
configs/
  alignment_stage1.yaml
  joint_lora_stage2.yaml
  eval_stage2.yaml
scripts/
  train_alignment.py
  train_joint_lora.py
  evaluate_hia_qwen.py
src/hia_qwen/
  data.py
  stage2_data.py
  hia_features.py
  modeling.py
  joint_modeling.py
  schema.py
tests/
  test_stage1_alignment.py
  test_stage2_pipeline.py
```

## Environment

The system `python3` can run local unit tests, but it cannot import `Qwen2AudioForConditionalGeneration` because its PyTorch is too old for the installed Transformers package:

```text
AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'
```

Create and use a project venv for full Qwen dry-runs and training:

```bash
cd /datas/store163/whyhugo/hia-qwen
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If a CUDA-specific PyTorch wheel is needed:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu121 'torch>=2.2.0'
python -m pip install -r requirements.txt
```

Verify Qwen2-Audio import:

```bash
python -c "from transformers import Qwen2AudioForConditionalGeneration; print('ok')"
```

## Tests

These tests avoid loading Qwen2-Audio and can run on system Python:

```bash
python3 -m unittest discover tests
```

Expected current result: `10 tests OK`.

## Stage 1: Alignment Pre-training

Stage 1 trains only the projector MLPs. HIA and Qwen2-Audio are frozen.

Dry-run:

```bash
source .venv/bin/activate
python scripts/train_alignment.py --config configs/alignment_stage1.yaml --dry-run --max-steps 2
```

Train:

```bash
python scripts/train_alignment.py --config configs/alignment_stage1.yaml
```

Output:

```text
outputs/alignment_stage1/projector.pt
```

## Stage 2: Joint LoRA Fine-tuning

Stage 2 trains projectors and Qwen LoRA adapters. HIA remains frozen and Qwen base weights remain frozen. Default LoRA modules are `q_proj`, `k_proj`, `v_proj`, and `o_proj`.

Dry-run:

```bash
python scripts/train_joint_lora.py --config configs/joint_lora_stage2.yaml --dry-run --max-steps 2
```

Train:

```bash
python scripts/train_joint_lora.py --config configs/joint_lora_stage2.yaml
```

Formal Stage-2 training should run after Stage 1 so it can initialize from `outputs/alignment_stage1/projector.pt`. Stage-2 dry-run can still validate the pipeline with scratch projectors if that file is absent.

## Stage-2 JSON Schema

Stage 2 outputs valid JSON only:

```json
{
  "sentence": {"accuracy": 8, "fluency": 9, "prosody": 9, "completeness": 10, "total": 8},
  "words": [{"text": "WE", "accuracy": 10, "stress": 10, "total": 10}],
  "phones": [{"phone": "W", "word_index": 0, "index": 0, "accuracy": 10}]
}
```

Scores use the SpeechOcean762 0-10 scale. Word stress remains 5 or 10 in the labels.

## Evaluation

Gold-target smoke test, no Qwen load required:

```bash
python scripts/evaluate_hia_qwen.py --config configs/eval_stage2.yaml --use-targets
```

Expected smoke result: `invalid_count = 0`, all populated RMSE values are `0.0`, and all non-constant populated PCC/SCC values are `1.0`.

Evaluate a trained Stage-2 checkpoint:

```bash
python scripts/evaluate_hia_qwen.py --config configs/eval_stage2.yaml
```

Evaluation writes:

```text
outputs/joint_lora_stage2/eval_test/predictions.jsonl
outputs/joint_lora_stage2/eval_test/scored_predictions.jsonl
outputs/joint_lora_stage2/eval_test/metrics.json
```

Metrics include PCC, SCC, RMSE, and invalid JSON count.

## Recommended Workflow

```bash
cd /datas/store163/whyhugo/hia-qwen
source .venv/bin/activate

python -m unittest discover tests
python scripts/evaluate_hia_qwen.py --config configs/eval_stage2.yaml --use-targets
python scripts/train_alignment.py --config configs/alignment_stage1.yaml --dry-run --max-steps 2
python scripts/train_alignment.py --config configs/alignment_stage1.yaml
python scripts/train_joint_lora.py --config configs/joint_lora_stage2.yaml --dry-run --max-steps 2
python scripts/train_joint_lora.py --config configs/joint_lora_stage2.yaml
python scripts/evaluate_hia_qwen.py --config configs/eval_stage2.yaml
```

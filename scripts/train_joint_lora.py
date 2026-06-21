#!/usr/bin/env python3
"""Stage-2 joint projector + LoRA fine-tuning for HIA-Qwen APA."""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from hia_qwen.hia_features import HiaFeatureExtractor
from hia_qwen.joint_modeling import HiaQwenJointModel, is_stage2_trainable_name
from hia_qwen.stage2_data import Stage2JsonDataset, collate_stage2_batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required to read the config.") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def require_vram(min_free_gb: float) -> None:
    if min_free_gb <= 0:
        return
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; Qwen2-Audio 7B training needs a GPU.")
    free_bytes, _ = torch.cuda.mem_get_info()
    free_gb = free_bytes / (1024**3)
    if free_gb < min_free_gb:
        raise SystemExit(f"Need {min_free_gb:.1f} GiB free VRAM; currently {free_gb:.1f} GiB.")


def check_transformers_import() -> None:
    try:
        import transformers  # noqa: F401
        from transformers import Qwen2AudioForConditionalGeneration  # noqa: F401
        import peft  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "Could not import Qwen2AudioForConditionalGeneration/PEFT. Use the project "
            "venv from ENVIRONMENT.md before running stage-2 dry-run/training.\n"
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def trainable_parameter_names(model: torch.nn.Module) -> list[str]:
    return [name for name, param in model.named_parameters() if param.requires_grad]


def gradient_parameter_names(model: torch.nn.Module) -> list[str]:
    return [name for name, param in model.named_parameters() if param.grad is not None]


def load_joint_checkpoint(model: Any, checkpoint_dir: Path) -> None:
    """Load projector weights and LoRA adapter weights from a saved checkpoint."""
    from peft import set_peft_model_state_dict
    try:
        from safetensors.torch import load_file as safetensors_load
        adapter_weights = safetensors_load(str(checkpoint_dir / "adapter_model.safetensors"))
    except Exception as exc:
        raise RuntimeError(f"Failed to load adapter_model.safetensors from {checkpoint_dir}: {exc}") from exc
    set_peft_model_state_dict(model.llm, adapter_weights)
    model.load_projectors(checkpoint_dir, strict=True)
    print(f"[resume] Loaded projectors + LoRA adapter from {checkpoint_dir}")


def parse_resume_epoch(checkpoint_dir: Path) -> int:
    """Return the completed epoch number encoded in the checkpoint directory name.

    Expects a path ending in 'epoch_N'. Raises ValueError if unparseable.
    """
    name = checkpoint_dir.name
    if not name.startswith("epoch_"):
        raise ValueError(
            f"Cannot parse epoch from checkpoint dir name {name!r}. "
            "Expected format: 'epoch_N' (e.g. outputs/.../checkpoints/epoch_3)"
        )
    try:
        return int(name.split("_", 1)[1])
    except ValueError as exc:
        raise ValueError(f"Cannot parse epoch number from {name!r}") from exc


def read_global_step_offset(train_log_path: Path) -> int:
    """Read the last step recorded in train_log.jsonl for TensorBoard continuity."""
    if not train_log_path.exists():
        return 0
    last_step = 0
    with train_log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    last_step = json.loads(line).get("step", last_step)
                except json.JSONDecodeError:
                    pass
    return last_step


def get_git_hash() -> str:
    """Return short git commit hash, or 'unknown' if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def make_exp_name(cfg: dict) -> str:
    """Generate a human-readable experiment name from key hyperparams + timestamp.

    Format: s2_r{rank}_lr{lr}_ep{epochs}_{YYYYMMDD_HHMM}
    Example: s2_r8_lr1e-4_ep3_20260604_1430
    """
    # Allow manual override in config
    if cfg.get("exp_name"):
        return str(cfg["exp_name"])
    train_cfg = cfg.get("train", {})
    lora_cfg = cfg.get("lora", {})
    lr = float(train_cfg.get("learning_rate", 1e-4))
    ep = int(train_cfg.get("num_epochs", 1))
    rank = int(lora_cfg.get("rank", 8))
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    return f"s2_r{rank}_lr{lr:.0e}_ep{ep}_{ts}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        metavar="CHECKPOINT_DIR",
        help="Resume from an epoch checkpoint, e.g. outputs/joint_lora_stage2/checkpoints/epoch_3. "
             "Loads projector + LoRA weights and continues from the next epoch.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    torch.manual_seed(int(cfg.get("seed", 17)))
    check_transformers_import()
    require_vram(float(cfg.get("min_free_vram_gb", 0)))

    data_cfg = cfg["data"]
    hia_cfg = cfg["hia"]
    qwen_cfg = cfg["qwen"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg.get("train", {})
    out_dir = Path(cfg.get("output_dir", "outputs/joint_lora_stage2"))

    exp_name = make_exp_name(cfg)
    start_time = datetime.datetime.now().isoformat()
    git_hash = get_git_hash()

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        run_meta = {
            **cfg,
            "exp_name": exp_name,
            "git_hash": git_hash,
            "start_time": start_time,
        }
        (out_dir / "run_config.json").write_text(
            json.dumps(run_meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[exp] name:     {exp_name}")
        print(f"[exp] git:      {git_hash}")
        print(f"[exp] out_dir:  {out_dir}")
        print(f"[exp] tb_dir:   runs/{exp_name}")

    dataset = Stage2JsonDataset(
        jsonl_path=data_cfg["train_jsonl"],
        seq_data_dir=data_cfg["seq_data_dir"],
        raw_data_root=data_cfg["raw_data_root"],
        split="train",
        task=data_cfg.get("task", "multi_all"),
        max_records=cfg.get("max_train_records"),
        use_hia=bool(data_cfg.get("use_hia", True)),
    )
    if not bool(data_cfg.get("use_hia", True)):
        print("[ablation] use_hia=False: HIA soft tokens NOT injected (pure LoRA).")
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=not args.dry_run,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate_stage2_batch,
    )
    print(f"Loaded {len(dataset)} aligned stage-2 JSON records.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hia = HiaFeatureExtractor(
        hia_repo=hia_cfg["repo"],
        checkpoint_path=hia_cfg["checkpoint"],
        embed_dim=int(hia_cfg.get("embed_dim", 48)),
        num_heads=int(hia_cfg.get("num_heads", 1)),
        depth=int(hia_cfg.get("depth", 3)),
        dropout=float(hia_cfg.get("dropout", 0.1)),
        seq_len=int(hia_cfg.get("seq_len", 50)),
        device=device,
    )

    dtype_name = qwen_cfg.get("torch_dtype", "bfloat16")
    dtype = torch.bfloat16 if dtype_name == "bfloat16" and torch.cuda.is_available() else torch.float32
    model = HiaQwenJointModel.from_pretrained(
        base_model=qwen_cfg.get("base_model", "Qwen/Qwen2-Audio-7B-Instruct"),
        hia_dim=hia.embed_dim,
        torch_dtype=dtype,
        device_map=qwen_cfg.get("device_map", "auto" if torch.cuda.is_available() else None),
        quantization=qwen_cfg.get("quantization"),
        local_files_only=bool(qwen_cfg.get("local_files_only", True)),
        projector_hidden_dim=qwen_cfg.get("projector_hidden_dim"),
        projector_dropout=float(qwen_cfg.get("projector_dropout", 0.0)),
    )
    projector_checkpoint = qwen_cfg.get("projector_checkpoint")
    if projector_checkpoint:
        path = Path(projector_checkpoint)
        if not path.exists():
            # Fail loudly: a configured-but-missing checkpoint almost always means
            # the Stage-1 run crashed before saving. Silently falling back to
            # random projectors invalidates the experiment (see the multi run that
            # trained projectors from scratch because Stage 1 died at step 974).
            raise SystemExit(
                f"Configured projector checkpoint does not exist: {path}\n"
                "Stage 1 likely did not finish and never saved projector.pt. "
                "Re-run Stage 1 to completion, or remove qwen.projector_checkpoint "
                "from the config to intentionally start projectors from scratch."
            )
        model.load_projectors(path, strict=True)
        print(f"Loaded stage-1 projectors from {path}")
    else:
        print("No projector checkpoint configured; starting projectors from scratch.")

    model.configure_lora(
        rank=int(lora_cfg.get("rank", 8)),
        alpha=int(lora_cfg.get("alpha", 16)),
        dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        prepare_kbit=qwen_cfg.get("quantization") == "4bit",
    )
    model.projectors.to(device)

    # -----------------------------------------------------------------------
    # Resume from checkpoint (must happen after configure_lora)
    # -----------------------------------------------------------------------
    start_epoch = 0
    global_step_offset = 0
    if args.resume_from is not None:
        resume_dir = args.resume_from.resolve()
        if not resume_dir.exists():
            raise SystemExit(f"--resume-from path does not exist: {resume_dir}")
        load_joint_checkpoint(model, resume_dir)
        start_epoch = parse_resume_epoch(resume_dir)
        global_step_offset = read_global_step_offset(out_dir / "train_log.jsonl")
        print(f"[resume] Resuming from epoch {start_epoch + 1} (completed epochs: {start_epoch})")
        print(f"[resume] TensorBoard step offset: {global_step_offset}")

    model.train()

    trainable = trainable_parameter_names(model)
    bad_trainable = [name for name in trainable if not is_stage2_trainable_name(name)]
    if bad_trainable:
        raise RuntimeError(f"Only projectors and LoRA may be trainable, got: {bad_trainable[:20]}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("Trainable modules:", ", ".join(trainable[:10]), "..." if len(trainable) > 10 else "")

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    grad_accum = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
    total_epochs = 1 if args.dry_run else int(train_cfg.get("num_epochs", 1))
    if args.resume_from is not None and not args.dry_run:
        if start_epoch >= total_epochs:
            raise SystemExit(
                f"Checkpoint is at epoch {start_epoch}, but num_epochs={total_epochs}. "
                "Increase num_epochs in the config before resuming."
            )
    max_steps = args.max_steps if args.max_steps is not None else cfg.get("max_steps")
    if args.dry_run and max_steps is None:
        max_steps = 2

    # -----------------------------------------------------------------------
    # TensorBoard writer + training log file
    # -----------------------------------------------------------------------
    writer: SummaryWriter | None = None
    train_log_f = None
    if not args.dry_run:
        tb_dir = ROOT / "runs" / exp_name
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        # Append mode: safe to resume or re-open
        train_log_f = (out_dir / "train_log.jsonl").open("a", encoding="utf-8")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    global_step = global_step_offset
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    train_start = time.time()

    for epoch in range(start_epoch, total_epochs):
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for batch in dataloader:
            features = hia(batch.gop.to(device), batch.phn_id.to(device), batch.word_id.to(device))
            outputs = model(batch.prompts, batch.targets, features)
            loss = outputs.loss
            (loss / grad_accum).backward()
            bad_grads = [name for name in gradient_parameter_names(model) if not is_stage2_trainable_name(name)]
            if bad_grads:
                raise RuntimeError(f"Frozen parameters received gradients: {bad_grads[:20]}")
            micro_step += 1
            if micro_step % grad_accum != 0:
                continue
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            loss_val = float(loss.detach().cpu().item())
            epoch_loss_sum += loss_val
            epoch_steps += 1

            log_record = {
                "epoch": epoch + 1,
                "step": global_step,
                "loss": loss_val,
                "phone_lengths": features.phone_lengths,
                "word_lengths": features.word_lengths,
                "raw_word_branch_shape": list(features.raw_word_branch.shape),
            }
            print(json.dumps(log_record))

            if writer is not None:
                writer.add_scalar("train/loss", loss_val, global_step)

            if train_log_f is not None:
                train_log_f.write(json.dumps(log_record) + "\n")
                train_log_f.flush()

            if max_steps is not None and global_step >= int(max_steps):
                break

        # -------------------------------------------------------------------
        # End of epoch: log + epoch checkpoint
        # -------------------------------------------------------------------
        if epoch_steps > 0 and writer is not None:
            epoch_avg_loss = epoch_loss_sum / epoch_steps
            writer.add_scalar("train/loss_epoch", epoch_avg_loss, epoch + 1)
            print(f"[epoch {epoch + 1}] avg_loss={epoch_avg_loss:.4f}  steps={epoch_steps}")

        if not args.dry_run:
            ckpt_dir = out_dir / "checkpoints" / f"epoch_{epoch + 1}"
            model.save_joint(ckpt_dir)
            print(f"[epoch {epoch + 1}] checkpoint saved → {ckpt_dir}")

        if max_steps is not None and global_step >= int(max_steps):
            break

    # -----------------------------------------------------------------------
    # Finalize
    # -----------------------------------------------------------------------
    elapsed = time.time() - train_start

    if train_log_f is not None:
        train_log_f.close()
    if writer is not None:
        writer.close()

    if args.dry_run:
        print("Stage-2 dry run complete: full forward/backward passed; no checkpoint saved.")
        return

    model.save_joint(out_dir)
    print(f"Stage-2 training complete. Projectors and LoRA adapter saved to: {out_dir}")
    print(f"Total time: {elapsed / 60:.1f} min  |  steps: {global_step}  |  exp: {exp_name}")

    # Write final summary
    summary = {
        "exp_name": exp_name,
        "git_hash": git_hash,
        "start_time": start_time,
        "end_time": datetime.datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "global_steps": global_step,
        "epochs_completed": total_epochs,
        "resumed_from_epoch": start_epoch if args.resume_from else None,
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

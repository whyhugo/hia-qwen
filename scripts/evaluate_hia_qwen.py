#!/usr/bin/env python3
"""Evaluate HIA-Qwen JSON predictions for stage-2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from torch.utils.data import DataLoader

from hia_qwen.hia_features import HiaFeatureExtractor
from hia_qwen.joint_modeling import HiaQwenJointModel
from hia_qwen.schema import compute_json_metrics, json_sanitize, parse_json_prediction, render_json_target
from hia_qwen.stage2_data import Stage2JsonDataset, collate_stage2_batch


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


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def score_predictions(records: List[Dict[str, Any]], predictions: Dict[str, str]) -> Dict[str, Any]:
    parsed_predictions = []
    scored = []
    invalid = []
    for record in records:
        text = predictions.get(record["id"])
        if text is None:
            error = "missing prediction"
            invalid.append({"id": record["id"], "error": error})
            parsed_predictions.append({"sentence": {}, "words": [], "phones": []})
            scored.append({"id": record["id"], "valid": False, "error": error})
            continue
        try:
            parsed = parse_json_prediction(text, record["labels"]["words"])
            parsed_predictions.append(parsed)
            scored.append({"id": record["id"], "prediction": text, "parsed": parsed, "valid": True})
        except Exception as exc:
            invalid.append({"id": record["id"], "error": str(exc), "prediction": text})
            parsed_predictions.append({"sentence": {}, "words": [], "phones": []})
            scored.append({"id": record["id"], "prediction": text, "valid": False, "error": str(exc)})
    metrics = compute_json_metrics(records, parsed_predictions)
    metrics["invalid_count"] = len(invalid)
    metrics["invalid_examples"] = invalid[:20]
    return {"metrics": metrics, "scored": scored}


def generate_predictions(cfg: Dict[str, Any], dataset: Stage2JsonDataset, max_new_tokens: int) -> List[Dict[str, str]]:
    try:
        from peft import PeftModel
    except Exception as exc:
        raise SystemExit(f"PEFT is required for checkpoint evaluation: {exc}") from exc

    hia_cfg = cfg["hia"]
    qwen_cfg = cfg["qwen"]
    eval_cfg = cfg.get("eval", {})
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
    checkpoint = Path(eval_cfg["checkpoint"])
    model.load_projectors(checkpoint, strict=True)
    model.llm = PeftModel.from_pretrained(model.llm, str(checkpoint))
    model.projectors.to(device)
    model.eval()
    batch_size = int(eval_cfg.get("batch_size", 1))
    total = len(dataset)

    # Length-bucketing: greedy decode runs each batch until its LONGEST row
    # finishes, so mixing short and long utterances wastes huge amounts of
    # generation on padding. Sort by phone count (longest first) so batches are
    # length-homogeneous; the longest batch also runs first, failing fast on OOM.
    # Order is irrelevant downstream — scoring looks predictions up by id.
    def _phone_count(record: Dict[str, Any]) -> int:
        return sum(len(word["phones"]) for word in record["labels"]["words"])

    order = sorted(range(len(dataset)), key=lambda i: _phone_count(dataset.records[i]), reverse=True)

    predictions: List[Dict[str, str]] = []
    checked = batch_size <= 1  # only need the padding self-check when batching
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch = collate_stage2_batch([dataset[i] for i in idx])
        features = hia(batch.gop.to(device), batch.phn_id.to(device), batch.word_id.to(device))
        # Correctness gate: on the first multi-row batch, verify that the
        # left-padded batched decode matches a per-row (no-padding) decode.
        # Aborts rather than silently emitting wrong scores.
        if not checked:
            k = min(4, len(batch.ids))
            batched_k = model.generate_json_batched(batch.prompts[:k], features, max_new_tokens=max_new_tokens)
            for i in range(k):
                fi = hia(
                    batch.gop[i : i + 1].to(device),
                    batch.phn_id[i : i + 1].to(device),
                    batch.word_id[i : i + 1].to(device),
                )
                ref = model.generate_json_batched([batch.prompts[i]], fi, max_new_tokens=max_new_tokens)[0]
                if ref.strip() != batched_k[i].strip():
                    raise SystemExit(
                        "Batched-generation self-check FAILED (padding bug). Re-run with "
                        f"eval.batch_size=1.\n--- single ---\n{ref}\n--- batched ---\n{batched_k[i]}"
                    )
            print(f"[eval] batched self-check passed on {k} records (batch_size={batch_size})")
            checked = True
        outputs = model.generate_json_batched(batch.prompts, features, max_new_tokens=max_new_tokens)
        for utt_id, output in zip(batch.ids, outputs):
            predictions.append({"id": utt_id, "prediction": output.strip()})
        if len(predictions) % 200 < batch_size:
            print(f"[eval] {len(predictions)}/{total} records generated", flush=True)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--use-targets", action="store_true")
    parser.add_argument("--predictions", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    eval_cfg = cfg.get("eval", {})
    out_dir = Path(eval_cfg.get("out_dir", "outputs/stage2_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = Stage2JsonDataset(
        jsonl_path=data_cfg["test_jsonl"],
        seq_data_dir=data_cfg["seq_data_dir"],
        raw_data_root=data_cfg["raw_data_root"],
        split="test",
        task=data_cfg.get("task", "multi_all"),
        max_records=eval_cfg.get("max_records"),
        use_hia=bool(data_cfg.get("use_hia", True)),
    )
    records = dataset.records

    if args.use_targets:
        generated = [{"id": record["id"], "prediction": render_json_target(record)} for record in records]
    elif args.predictions is not None:
        generated = read_jsonl(args.predictions)
    else:
        generated = generate_predictions(cfg, dataset, int(eval_cfg.get("max_new_tokens", 768)))
    prediction_map = {item["id"]: item["prediction"] for item in generated}
    result = score_predictions(records, prediction_map)

    write_jsonl(out_dir / "predictions.jsonl", generated)
    write_jsonl(out_dir / "scored_predictions.jsonl", result["scored"])
    (out_dir / "metrics.json").write_text(
        json.dumps(json_sanitize(result["metrics"]), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote predictions: {out_dir / 'predictions.jsonl'}")
    print(f"Wrote metrics: {out_dir / 'metrics.json'}")
    print(json.dumps(json_sanitize(result["metrics"]), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

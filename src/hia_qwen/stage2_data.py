"""Stage-2 dataset for JSON multi-granularity HIA-Qwen training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import (
    HIA_PHONE_TOKEN,
    HIA_UTT_TOKEN,
    HIA_WORD_TOKEN,
    normalize_gop,
    read_jsonl,
    read_kaldi_ids,
)
from .schema import build_json_instruction, render_json_target


@dataclass
class Stage2Batch:
    ids: List[str]
    prompts: List[str]
    targets: List[str]
    gop: torch.Tensor
    phn_id: torch.Tensor
    word_id: torch.Tensor
    records: List[Dict[str, Any]]


def build_stage2_prompt(record: Dict[str, Any], use_hia: bool = True) -> str:
    if use_hia:
        return (
            "You are a pronunciation evaluation teacher. Evaluate the learner's "
            "pronunciation using the reference text, reference phone sequence, and "
            "injected HIA acoustic soft tokens.\n"
            f"Reference text: {record['reference_text']}.\n"
            f"Reference phone sequence: {record['reference_phone_sequence']}.\n"
            f"HIA utterance feature: {HIA_UTT_TOKEN}\n"
            f"HIA word feature sequence: {HIA_WORD_TOKEN}\n"
            f"HIA phone feature sequence: {HIA_PHONE_TOKEN}\n"
            + build_json_instruction()
        )
    # Ablation: no HIA soft tokens injected. The prompt omits the placeholder
    # tokens, so merge_hia_soft_tokens has nothing to replace -> pure LoRA.
    return (
        "You are a pronunciation evaluation teacher. Evaluate the learner's "
        "pronunciation using the reference text and reference phone sequence.\n"
        f"Reference text: {record['reference_text']}.\n"
        f"Reference phone sequence: {record['reference_phone_sequence']}.\n"
        + build_json_instruction()
    )


class Stage2JsonDataset(Dataset):
    """SO762 multi_all JSON records aligned to HIA GOP numpy rows."""

    def __init__(
        self,
        jsonl_path: str | Path,
        seq_data_dir: str | Path,
        raw_data_root: str | Path,
        split: str = "train",
        task: str = "multi_all",
        max_records: int | None = None,
        use_hia: bool = True,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError(f"split must be train or test, got {split!r}")
        self.split = split
        self.task = task
        self.use_hia = use_hia
        self.jsonl_path = Path(jsonl_path)
        self.seq_data_dir = Path(seq_data_dir)
        self.raw_data_root = Path(raw_data_root)

        prefix = "tr" if split == "train" else "te"
        self.feat = np.load(self.seq_data_dir / f"{prefix}_feat.npy", mmap_mode="r")
        self.label_phn = np.load(self.seq_data_dir / f"{prefix}_label_phn.npy", mmap_mode="r")
        self.label_word = np.load(self.seq_data_dir / f"{prefix}_label_word.npy", mmap_mode="r")

        hia_ids = read_kaldi_ids(self.raw_data_root / split / "wav.scp")
        if len(hia_ids) != len(self.feat):
            raise ValueError(
                f"HIA row count mismatch: {len(hia_ids)} ids in wav.scp, "
                f"{len(self.feat)} feature rows"
            )
        self.row_by_id = {utt_id: i for i, utt_id in enumerate(hia_ids)}

        records = read_jsonl(self.jsonl_path)
        if max_records is not None:
            records = records[:max_records]
        for record in records:
            if record.get("task") != task:
                raise ValueError(f"Expected {task} records, got {record.get('task')!r}")
            if record["id"] not in self.row_by_id:
                raise KeyError(f"JSONL id not present in HIA wav.scp order: {record['id']}")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        row = self.row_by_id[record["id"]]
        gop = normalize_gop(np.asarray(self.feat[row]))
        phn_id = np.asarray(self.label_phn[row, :, 0]).copy()
        word_id = np.asarray(self.label_word[row, :, 3]).copy()
        return {
            "id": record["id"],
            "prompt": build_stage2_prompt(record, use_hia=self.use_hia),
            "target": render_json_target(record),
            "gop": torch.from_numpy(gop).float(),
            "phn_id": torch.from_numpy(phn_id).long(),
            "word_id": torch.from_numpy(word_id).long(),
            "record": record,
        }


def collate_stage2_batch(samples: Sequence[Dict[str, Any]]) -> Stage2Batch:
    return Stage2Batch(
        ids=[str(sample["id"]) for sample in samples],
        prompts=[str(sample["prompt"]) for sample in samples],
        targets=[str(sample["target"]) for sample in samples],
        gop=torch.stack([sample["gop"] for sample in samples], dim=0),
        phn_id=torch.stack([sample["phn_id"] for sample in samples], dim=0),
        word_id=torch.stack([sample["word_id"] for sample in samples], dim=0),
        records=[sample["record"] for sample in samples],
    )

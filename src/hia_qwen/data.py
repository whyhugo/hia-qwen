"""Dataset and text construction for stage-1 HIA/Qwen alignment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


HIA_UTT_TOKEN = "<|hia_utt|>"
HIA_WORD_TOKEN = "<|hia_word|>"
HIA_PHONE_TOKEN = "<|hia_phone|>"
HIA_SPECIAL_TOKENS = [HIA_UTT_TOKEN, HIA_WORD_TOKEN, HIA_PHONE_TOKEN]

_GOP_MEAN = 3.203
_GOP_STD = 4.045


@dataclass
class AlignmentBatch:
    ids: List[str]
    prompts: List[str]
    targets: List[str]
    gop: torch.Tensor
    phn_id: torch.Tensor
    word_id: torch.Tensor


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_kaldi_ids(path: Path) -> List[str]:
    ids: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line.split(maxsplit=1)[0])
    return ids


def normalize_gop(feat: np.ndarray) -> np.ndarray:
    """Match the HIA reproduction GOP normalization."""
    out = feat.copy()
    valid = feat[:, 0] != 0
    out[valid] = (feat[valid] - _GOP_MEAN) / (_GOP_STD + 1e-8)
    return out


def build_stage1_prompt(record: Dict[str, Any]) -> str:
    reference_text = record["reference_text"]
    phone_sequence = record["reference_phone_sequence"]
    return (
        "You are a pronunciation evaluation teacher. Evaluate the learner's "
        "pronunciation using the reference text, phone sequence, and injected "
        "HIA acoustic soft tokens.\n"
        f"Reference text: {reference_text}.\n"
        f"Reference phone sequence: {phone_sequence}.\n"
        "HIA utterance feature: "
        f"{HIA_UTT_TOKEN}\n"
        "HIA word feature sequence: "
        f"{HIA_WORD_TOKEN}\n"
        "HIA phone feature sequence: "
        f"{HIA_PHONE_TOKEN}\n"
        "Assess only sentence-level total. Return exactly this format:\n"
        "Score: {Score}"
    )


class AlignmentDataset(Dataset):
    """SO762 records aligned with the HIA GOP numpy tensors.

    Microsoft JSONL records and HIA `.npy` tensors are both derived from the same
    official splits, but the robust source of HIA row order is the split's
    `wav.scp`. This dataset maps every JSONL `id` to its HIA row and fails early
    if the two sources drift.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        seq_data_dir: str | Path,
        raw_data_root: str | Path,
        split: str = "train",
        max_records: int | None = None,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError(f"split must be train or test, got {split!r}")
        self.split = split
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
            if record.get("task") != "sentence_total":
                raise ValueError(
                    f"Stage 1 expects sentence_total records, got {record.get('task')!r}"
                )
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
            "prompt": build_stage1_prompt(record),
            "target": record["target"],
            "gop": torch.from_numpy(gop).float(),
            "phn_id": torch.from_numpy(phn_id).long(),
            "word_id": torch.from_numpy(word_id).long(),
        }


def collate_alignment_batch(samples: Sequence[Dict[str, Any]]) -> AlignmentBatch:
    return AlignmentBatch(
        ids=[str(sample["id"]) for sample in samples],
        prompts=[str(sample["prompt"]) for sample in samples],
        targets=[str(sample["target"]) for sample in samples],
        gop=torch.stack([sample["gop"] for sample in samples], dim=0),
        phn_id=torch.stack([sample["phn_id"] for sample in samples], dim=0),
        word_id=torch.stack([sample["word_id"] for sample in samples], dim=0),
    )


def iter_trainable_names(parameters: Iterable[tuple[str, torch.nn.Parameter]]) -> List[str]:
    return [name for name, param in parameters if param.requires_grad]

"""Frozen HIA feature extraction for Qwen soft-prompt alignment."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F


@dataclass
class HiaFeatureOutput:
    """Variable-length HIA features ready for projection."""

    phone_features: List[torch.Tensor]
    word_features: List[torch.Tensor]
    utt_features: List[torch.Tensor]
    raw_word_branch: torch.Tensor
    phone_lengths: List[int]
    word_lengths: List[int]


def import_hia_class(hia_repo: str | Path):
    src_dir = Path(hia_repo).resolve() / "src"
    if not src_dir.exists():
        raise FileNotFoundError(f"HIA src directory not found: {src_dir}")
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from models import HIA  # type: ignore

    return HIA


def pool_word_branch_by_word_id(
    word_branch: torch.Tensor,
    word_id: torch.Tensor,
) -> List[torch.Tensor]:
    """Mean-pool `F_word [B,L,D]` into dynamic `[T_wrd,D]` sequences.

    The IAM `H_word` tensor is a single global query `[B,1,D]`; it is useful as
    conditioning inside HIA but is not the word-level soft prompt. The word-level
    soft prompt comes from the Word Branch hidden state `F_word [B,L,D]`, pooled
    by `word_id` so each output token represents one reference word.
    """
    if word_branch.ndim != 3:
        raise ValueError(f"word_branch must be [B,L,D], got {tuple(word_branch.shape)}")
    if word_id.ndim != 2:
        raise ValueError(f"word_id must be [B,L], got {tuple(word_id.shape)}")
    if word_branch.shape[:2] != word_id.shape:
        raise ValueError(
            f"word_branch/word_id shape mismatch: {tuple(word_branch.shape)} vs "
            f"{tuple(word_id.shape)}"
        )

    pooled: List[torch.Tensor] = []
    for hidden, ids in zip(word_branch, word_id):
        valid_ids = ids[ids >= 0].unique(sorted=True)
        if valid_ids.numel() == 0:
            pooled.append(hidden.new_zeros((0, hidden.shape[-1])))
            continue
        pooled.append(torch.stack([hidden[ids == wid].mean(dim=0) for wid in valid_ids], dim=0))
    return pooled


class HiaFeatureExtractor(torch.nn.Module):
    """Frozen HIA wrapper that exposes intermediate branch features."""

    def __init__(
        self,
        hia_repo: str | Path,
        checkpoint_path: str | Path,
        embed_dim: int = 48,
        num_heads: int = 1,
        depth: int = 3,
        dropout: float = 0.1,
        seq_len: int = 50,
        gop_dim: int = 84,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        HIA = import_hia_class(hia_repo)
        self.model = HIA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            dropout=dropout,
            seq_len=seq_len,
            gop_dim=gop_dim,
        )
        state = torch.load(Path(checkpoint_path), map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        if device is not None:
            self.model.to(device)

    @property
    def embed_dim(self) -> int:
        return int(self.model.embed_dim)

    @torch.no_grad()
    def forward(self, gop: torch.Tensor, phn: torch.Tensor, word_id: torch.Tensor) -> HiaFeatureOutput:
        model = self.model
        gop = gop.to(next(model.parameters()).device)
        phn = phn.to(gop.device)
        word_id = word_id.to(gop.device)

        pad_mask = phn < 0
        phn_shifted = (phn + 1).clamp(0, 41).long()
        phn_onehot = F.one_hot(phn_shifted, num_classes=42).float()
        phn_embed = model.phn_proj(phn_onehot)
        gop_embed = model.gop_proj(gop)
        X = model.input_dropout(gop_embed + phn_embed + model.pos_embed[:, : phn.shape[1], :])

        H = model.transformer_encoder(X, src_key_padding_mask=pad_mask)
        H_phn, H_word_global, H_utt = model.iam(H, src_key_padding_mask=pad_mask)

        F_phn = X + H_phn
        F_phn = model.phn_conv(F_phn.transpose(1, 2)).transpose(1, 2)
        F_phn = model.phn_norm(F_phn)
        p_acc = model.phn_head(F_phn).squeeze(-1)

        phn_residual = model.phn_res_proj(p_acc.unsqueeze(-1))
        F_word = X + phn_residual + H_word_global
        F_word = model.word_conv(F_word.transpose(1, 2)).transpose(1, 2)
        F_word = model.word_norm(F_word)
        word_out = model.word_head(F_word)

        word_residual = model.word_res_proj(word_out)
        memory = X + word_residual + H_utt
        dec_out = model.utt_decoder(H_utt, memory, memory_key_padding_mask=pad_mask)
        dec_out = model.utt_conv(dec_out.transpose(1, 2)).transpose(1, 2)
        dec_out = model.utt_norm(dec_out)

        phone_features = [F_phn[i, ~pad_mask[i]].detach() for i in range(F_phn.shape[0])]
        word_features = [seq.detach() for seq in pool_word_branch_by_word_id(F_word, word_id)]
        utt_features = [dec_out[i].detach() for i in range(dec_out.shape[0])]
        return HiaFeatureOutput(
            phone_features=phone_features,
            word_features=word_features,
            utt_features=utt_features,
            raw_word_branch=F_word.detach(),
            phone_lengths=[int(t.shape[0]) for t in phone_features],
            word_lengths=[int(t.shape[0]) for t in word_features],
        )

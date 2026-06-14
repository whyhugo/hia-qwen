"""HIA soft-token projection and Qwen2-Audio alignment model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn

from .data import HIA_PHONE_TOKEN, HIA_SPECIAL_TOKENS, HIA_UTT_TOKEN, HIA_WORD_TOKEN
from .hia_features import HiaFeatureOutput


@dataclass
class ProjectorConfig:
    hia_dim: int = 48
    llm_dim: int = 4096
    hidden_dim: int | None = None
    dropout: float = 0.0


class HiaProjectors(nn.Module):
    def __init__(self, cfg: ProjectorConfig) -> None:
        super().__init__()
        hidden_dim = cfg.hidden_dim or cfg.llm_dim
        self.utt_projector = self._mlp(cfg.hia_dim, hidden_dim, cfg.llm_dim, cfg.dropout)
        self.word_projector = self._mlp(cfg.hia_dim, hidden_dim, cfg.llm_dim, cfg.dropout)
        self.phone_projector = self._mlp(cfg.hia_dim, hidden_dim, cfg.llm_dim, cfg.dropout)

    @staticmethod
    def _mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
        layers: List[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)


def _single_token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError(f"Special token was not registered as a single token: {token}")
    return int(token_id)


def _get_text_embedding_layer(llm: nn.Module) -> nn.Module:
    if hasattr(llm, "get_input_embeddings"):
        emb = llm.get_input_embeddings()
        if emb is not None:
            return emb
    if hasattr(llm, "model") and hasattr(llm.model, "embed_tokens"):
        return llm.model.embed_tokens
    if hasattr(llm, "language_model") and hasattr(llm.language_model, "model"):
        model = llm.language_model.model
        if hasattr(model, "embed_tokens"):
            return model.embed_tokens
    raise AttributeError("Could not locate Qwen text input embeddings.")


def make_chat_text(tokenizer: Any, prompt: str, target: str | None) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
        if target is None:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        messages.append({"role": "assistant", "content": target})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    eos = getattr(tokenizer, "eos_token", "") or ""
    if target is None:
        return f"User: {prompt}\nAssistant:"
    return f"User: {prompt}\nAssistant: {target}{eos}"


def tokenize_with_labels(
    tokenizer: Any,
    prompts: Sequence[str],
    targets: Sequence[str],
) -> Dict[str, torch.Tensor]:
    full_texts = [make_chat_text(tokenizer, p, t) for p, t in zip(prompts, targets)]
    prompt_texts = [make_chat_text(tokenizer, p, None) for p in prompts]
    full = tokenizer(full_texts, return_tensors="pt", padding=True)
    prompt = tokenizer(prompt_texts, return_tensors="pt", padding=True)
    labels = full["input_ids"].clone()
    for row in range(labels.shape[0]):
        prompt_len = int(prompt["attention_mask"][row].sum().item())
        labels[row, :prompt_len] = -100
    labels[full["attention_mask"] == 0] = -100
    return {
        "input_ids": full["input_ids"],
        "attention_mask": full["attention_mask"],
        "labels": labels,
    }


def merge_hia_soft_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    token_embeds: torch.Tensor,
    special_token_ids: Dict[str, int],
    projected: Dict[str, List[torch.Tensor]],
    pad_label_id: int = -100,
) -> Dict[str, torch.Tensor]:
    """Replace placeholder token embeddings with variable-length HIA embeddings."""
    merged_embeds: List[torch.Tensor] = []
    merged_labels: List[torch.Tensor] = []
    device = token_embeds.device

    replacement_by_id = {
        special_token_ids[HIA_UTT_TOKEN]: "utt",
        special_token_ids[HIA_WORD_TOKEN]: "word",
        special_token_ids[HIA_PHONE_TOKEN]: "phone",
    }

    for row in range(input_ids.shape[0]):
        pieces: List[torch.Tensor] = []
        label_pieces: List[torch.Tensor] = []
        valid_len = int(attention_mask[row].sum().item())
        for col in range(valid_len):
            token_id = int(input_ids[row, col].item())
            feature_name = replacement_by_id.get(token_id)
            if feature_name is None:
                pieces.append(token_embeds[row, col : col + 1])
                label_pieces.append(labels[row, col : col + 1])
                continue
            feature = projected[feature_name][row].to(device=device, dtype=token_embeds.dtype)
            pieces.append(feature)
            label_pieces.append(
                torch.full((feature.shape[0],), pad_label_id, dtype=labels.dtype, device=device)
            )
        merged_embeds.append(torch.cat(pieces, dim=0))
        merged_labels.append(torch.cat(label_pieces, dim=0))

    max_len = max(x.shape[0] for x in merged_embeds)
    hidden = token_embeds.shape[-1]
    batch = len(merged_embeds)
    final_embeds = token_embeds.new_zeros((batch, max_len, hidden))
    final_attention = attention_mask.new_zeros((batch, max_len))
    final_labels = labels.new_full((batch, max_len), pad_label_id)
    for row, (embeds, row_labels) in enumerate(zip(merged_embeds, merged_labels)):
        length = embeds.shape[0]
        final_embeds[row, :length] = embeds
        final_attention[row, :length] = 1
        final_labels[row, :length] = row_labels
    return {
        "inputs_embeds": final_embeds,
        "attention_mask": final_attention,
        "labels": final_labels,
    }


class HiaQwenAlignmentModel(nn.Module):
    """Frozen Qwen2-Audio decoder with trainable HIA projection layers."""

    def __init__(
        self,
        llm: nn.Module,
        tokenizer: Any,
        hia_dim: int,
        projector_hidden_dim: int | None = None,
        projector_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        embedding_layer = _get_text_embedding_layer(self.llm)
        llm_dim = int(embedding_layer.embedding_dim)
        self.projectors = HiaProjectors(
            ProjectorConfig(
                hia_dim=hia_dim,
                llm_dim=llm_dim,
                hidden_dim=projector_hidden_dim,
                dropout=projector_dropout,
            )
        )
        self.special_token_ids = {token: _single_token_id(tokenizer, token) for token in HIA_SPECIAL_TOKENS}
        for param in self.llm.parameters():
            param.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        base_model: str,
        hia_dim: int,
        torch_dtype: torch.dtype | None = None,
        device_map: str | None = None,
        quantization: str | None = None,
        local_files_only: bool = False,
        projector_hidden_dim: int | None = None,
        projector_dropout: float = 0.0,
    ) -> "HiaQwenAlignmentModel":
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2AudioForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            base_model,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        tokenizer = getattr(processor, "tokenizer", processor)
        tokenizer.add_special_tokens({"additional_special_tokens": HIA_SPECIAL_TOKENS})
        model_kwargs: Dict[str, Any] = {"trust_remote_code": True, "local_files_only": local_files_only}
        if torch_dtype is not None:
            model_kwargs["dtype"] = torch_dtype
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        if quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        llm = Qwen2AudioForConditionalGeneration.from_pretrained(base_model, **model_kwargs)
        llm.resize_token_embeddings(len(tokenizer))
        model = cls(
            llm=llm,
            tokenizer=tokenizer,
            hia_dim=hia_dim,
            projector_hidden_dim=projector_hidden_dim,
            projector_dropout=projector_dropout,
        )
        model.processor = processor
        return model

    def forward(
        self,
        prompts: Sequence[str],
        targets: Sequence[str],
        hia_features: HiaFeatureOutput,
    ) -> Any:
        device = next(self.projectors.parameters()).device
        tokenized = tokenize_with_labels(self.tokenizer, prompts, targets)
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        input_embeds = _get_text_embedding_layer(self.llm)(tokenized["input_ids"])

        projected = {
            "utt": [self.projectors.utt_projector(x.to(device)) for x in hia_features.utt_features],
            "word": [self.projectors.word_projector(x.to(device)) for x in hia_features.word_features],
            "phone": [self.projectors.phone_projector(x.to(device)) for x in hia_features.phone_features],
        }
        merged = merge_hia_soft_tokens(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            labels=tokenized["labels"],
            token_embeds=input_embeds,
            special_token_ids=self.special_token_ids,
            projected=projected,
        )
        return self.llm(**merged)

    def save_projectors(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "projectors": self.projectors.state_dict(),
                "special_token_ids": self.special_token_ids,
            },
            output / "projector.pt",
        )
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(output)

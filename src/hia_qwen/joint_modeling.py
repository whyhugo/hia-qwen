"""Joint projector + LoRA model for stage-2 HIA-Qwen training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn

from .data import HIA_PHONE_TOKEN, HIA_SPECIAL_TOKENS, HIA_UTT_TOKEN, HIA_WORD_TOKEN
from .hia_features import HiaFeatureOutput
from .modeling import (
    HiaProjectors,
    ProjectorConfig,
    _get_text_embedding_layer,
    _single_token_id,
    make_chat_text,
    merge_hia_soft_tokens,
    tokenize_with_labels,
)


def is_stage2_trainable_name(name: str) -> bool:
    lowered = name.lower()
    return name.startswith("projectors.") or "lora_" in lowered or ".modules_to_save." in lowered


def tokenize_prompts(tokenizer: Any, prompts: Sequence[str]) -> Dict[str, torch.Tensor]:
    texts = [make_chat_text(tokenizer, prompt, None) for prompt in prompts]
    return tokenizer(texts, return_tensors="pt", padding=True)


class HiaQwenJointModel(nn.Module):
    """Qwen2-Audio with frozen HIA inputs, trainable projectors, and LoRA adapters."""

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
        llm_dim = int(_get_text_embedding_layer(self.llm).embedding_dim)
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
    ) -> "HiaQwenJointModel":
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2AudioForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            base_model,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        tokenizer = getattr(processor, "tokenizer", processor)
        tokenizer.add_special_tokens({"additional_special_tokens": HIA_SPECIAL_TOKENS})
        kwargs: Dict[str, Any] = {"trust_remote_code": True, "local_files_only": local_files_only}
        if torch_dtype is not None:
            kwargs["dtype"] = torch_dtype
        if device_map is not None:
            kwargs["device_map"] = device_map
        if quantization == "4bit":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        llm = Qwen2AudioForConditionalGeneration.from_pretrained(base_model, **kwargs)
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

    def configure_lora(
        self,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
        target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
        prepare_kbit: bool = False,
    ) -> None:
        from peft import LoraConfig, get_peft_model

        if prepare_kbit:
            from peft import prepare_model_for_kbit_training

            self.llm = prepare_model_for_kbit_training(self.llm)
        cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(target_modules),
        )
        self.llm = get_peft_model(self.llm, cfg)
        for name, param in self.llm.named_parameters():
            param.requires_grad_(("lora_" in name.lower()) or (".modules_to_save." in name.lower()))

    def load_projectors(self, checkpoint: str | Path, strict: bool = True) -> None:
        path = Path(checkpoint)
        if path.is_dir():
            path = path / "projector.pt"
        state = torch.load(path, map_location="cpu")
        projectors = state.get("projectors", state)
        self.projectors.load_state_dict(projectors, strict=strict)

    def projected_features(self, hia_features: HiaFeatureOutput, device: torch.device) -> Dict[str, List[torch.Tensor]]:
        return {
            "utt": [self.projectors.utt_projector(x.to(device)) for x in hia_features.utt_features],
            "word": [self.projectors.word_projector(x.to(device)) for x in hia_features.word_features],
            "phone": [self.projectors.phone_projector(x.to(device)) for x in hia_features.phone_features],
        }

    def merged_inputs(
        self,
        prompts: Sequence[str],
        hia_features: HiaFeatureOutput,
        targets: Sequence[str] | None = None,
    ) -> Dict[str, torch.Tensor]:
        device = next(self.projectors.parameters()).device
        if targets is None:
            tokenized = tokenize_prompts(self.tokenizer, prompts)
            labels = tokenized["input_ids"].new_full(tokenized["input_ids"].shape, -100)
            tokenized["labels"] = labels
        else:
            tokenized = tokenize_with_labels(self.tokenizer, prompts, targets)
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        embeds = _get_text_embedding_layer(self.llm)(tokenized["input_ids"])
        return merge_hia_soft_tokens(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            labels=tokenized["labels"],
            token_embeds=embeds,
            special_token_ids=self.special_token_ids,
            projected=self.projected_features(hia_features, device),
        )

    def forward(
        self,
        prompts: Sequence[str],
        targets: Sequence[str],
        hia_features: HiaFeatureOutput,
    ) -> Any:
        return self.llm(**self.merged_inputs(prompts, hia_features, targets))

    @torch.no_grad()
    def generate_json(
        self,
        prompts: Sequence[str],
        hia_features: HiaFeatureOutput,
        max_new_tokens: int = 768,
    ) -> List[str]:
        merged = self.merged_inputs(prompts, hia_features, targets=None)
        generated_ids = self.greedy_decode_from_inputs_embeds(
            inputs_embeds=merged["inputs_embeds"],
            attention_mask=merged["attention_mask"],
            max_new_tokens=max_new_tokens,
        )
        return self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    @torch.no_grad()
    def greedy_decode_from_inputs_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """Greedy decode for models whose `.generate()` rejects `inputs_embeds`.

        Qwen2AudioForConditionalGeneration.generate() currently raises when
        called with HIA-injected `inputs_embeds`. A manual first forward pass
        preserves soft-token conditioning, then subsequent steps use
        `past_key_values` and ordinary token ids.
        """
        if max_new_tokens <= 0:
            return torch.empty((inputs_embeds.shape[0], 0), dtype=torch.long, device=inputs_embeds.device)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = getattr(outputs, "past_key_values", None)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = [next_token]
        finished = torch.zeros(next_token.shape[0], dtype=torch.bool, device=next_token.device)
        if eos_token_id is not None:
            finished |= next_token.squeeze(1).eq(int(eos_token_id))

        running_attention = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
            dim=1,
        )
        for _ in range(max_new_tokens - 1):
            if bool(finished.all()):
                break
            outputs = self.llm(
                input_ids=next_token,
                attention_mask=running_attention,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = getattr(outputs, "past_key_values", None)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, int(eos_token_id)),
                    next_token,
                )
                finished |= next_token.squeeze(1).eq(int(eos_token_id))
            generated.append(next_token)
            running_attention = torch.cat(
                [running_attention, running_attention.new_ones((running_attention.shape[0], 1))],
                dim=1,
            )
        return torch.cat(generated, dim=1)

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

    def save_joint(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.save_projectors(output)
        if hasattr(self.llm, "save_pretrained"):
            self.llm.save_pretrained(output)
